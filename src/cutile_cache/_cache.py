# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SQLite storage for the cuTile compilation cache."""

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, asdict
from typing import Optional, ClassVar
from . import _remarks

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    key       TEXT PRIMARY KEY,
    blob      BLOB NOT NULL,
    blob_size INTEGER NOT NULL,
    atime     REAL NOT NULL,
    metadata  TEXT
)
"""

_CACHE_FILENAME = "cache.db"
_CACHE_ENTRY_BATCH_SIZE = 100


def _close(conn):
    if conn:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute(_CREATE_TABLE_SQL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
    if "metadata" not in columns:
        try:
            conn.execute("ALTER TABLE cache ADD COLUMN metadata TEXT")
        except sqlite3.OperationalError:
            # Another process may have migrated the shared cache concurrently.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
            if "metadata" not in columns:
                raise
    conn.execute("DROP INDEX IF EXISTS idx_cache_atime")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_atime_key on cache(atime, key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_blob_size on cache(blob_size)")
    return conn


def _connect(cache_dir: str) -> sqlite3.Connection:
    os.makedirs(cache_dir, exist_ok=True)
    db_path = os.path.join(cache_dir, _CACHE_FILENAME)
    try:
        return _open_db(db_path)
    except sqlite3.Error:
        logger.debug("cache db corrupt, recreating %s", db_path,
                     exc_info=True)
        try:
            os.unlink(db_path)
        except OSError:
            pass
        return _open_db(db_path)


_CACHE_VERSION = b''


@dataclass
class MetadataV1:
    """Versioned schema for compilation-cache metadata."""

    kernel_names: list[str] | None = None
    compiler_version: str | None = None
    compilation_timestamp: float | None = None
    compilation_time_seconds: float | None = None
    remarks: str | None = None
    version: ClassVar[int] = 1

    def _validate(self, field_name: str, expected_type):
        val = getattr(self, field_name)
        if val is not None and not isinstance(val, expected_type):
            raise TypeError(f"Expect {field_name} to be a {expected_type} "
                            f"but got {type(val)}")

    def __post_init__(self):
        self._validate("kernel_names", list)
        if self.kernel_names is not None:
            try:
                idx, item = next((i, x) for i, x in
                                 enumerate(self.kernel_names)
                                 if not isinstance(x, str))
                raise TypeError(f"Expect kernel_names to be a list[str], "
                                f"but {idx}th item is {type(item)}")
            except StopIteration:
                pass
        self._validate("compiler_version", str)
        self._validate("compilation_timestamp", float)
        self._validate("compilation_time_seconds", float)
        self._validate("remarks", str)

    def to_dict(self):
        ret = asdict(self)
        if ret['remarks'] is not None:
            ret['remarks'] = _remarks.cleanup(ret['remarks'])
        # ClassVar is not serialized
        ret["version"] = self.version
        return ret


def cache_key(compiler_version: str, sm_arch: str, opt_level: int,
              bytecode: bytes, device_debug: bool = False) -> str:

    def encode_uint(x: int):
        return int.to_bytes(x, 4, byteorder='big', signed=False)

    version = compiler_version.encode()
    arch = sm_arch.encode()

    h = hashlib.sha256()
    h.update(_CACHE_VERSION)
    h.update(encode_uint(len(version)))
    h.update(version)
    h.update(encode_uint(len(arch)))
    h.update(arch)
    h.update(encode_uint(opt_level | (int(device_debug) << 8)))
    h.update(encode_uint(len(bytecode)))
    h.update(bytecode)
    return h.hexdigest()


def cache_lookup(cache_dir: str, key: str) -> Optional[bytes]:
    conn = None
    try:
        conn = _connect(cache_dir)
        row = conn.execute(
            "SELECT blob FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE cache SET atime = ? WHERE key = ?",
            (time.time(), key)
        )
        conn.commit()
        blob = row[0]
        return blob
    except (sqlite3.Error, OSError):
        logger.debug("cache lookup failed for %s", key, exc_info=True)
        return None
    finally:
        _close(conn)


def cache_store(cache_dir: str, key: str, cubin: bytes,
                metadata: dict[str, object] | None = None) -> None:
    conn = None
    try:
        conn = _connect(cache_dir)
        metadata_json = None if metadata is None else json.dumps(metadata, sort_keys=True)
        conn.execute(
            "INSERT OR IGNORE INTO cache"
            " (key, blob, blob_size, atime, metadata) VALUES (?, ?, ?, ?, ?)",
            (key, cubin, len(cubin), time.time(), metadata_json)
        )
        conn.commit()
    except (sqlite3.Error, OSError):
        logger.debug("cache store failed for %s", key, exc_info=True)
    finally:
        _close(conn)


@dataclass(frozen=True)
class CacheEntry:
    key: str
    size: int
    atime: float
    metadata: MetadataV1


def cache_entries(cache_dir: str, max_count: int | None = None) -> Iterator[CacheEntry]:
    """Lazily iterate over cache entries in most-recently-accessed order."""
    conn = None
    try:
        conn = _connect(cache_dir)
        # treat negative max_count as unlimited
        remaining = None if max_count is not None and max_count < 0 else max_count
        last_atime = None
        last_key = None
        while remaining is None or remaining > 0:
            batch_size = _CACHE_ENTRY_BATCH_SIZE
            if remaining is not None:
                batch_size = min(batch_size, remaining)

            query = "SELECT key, blob_size, atime, metadata FROM cache "
            params = ()
            if last_atime is not None:
                query += "WHERE (atime, key) < (?, ?) "
                params = (last_atime, last_key)
            query += "ORDER BY atime DESC, key DESC LIMIT ?"
            params += (batch_size,)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                break

            for key, size, atime, metadata_json in rows:
                metadata = MetadataV1()
                if metadata_json is not None:
                    try:
                        value = json.loads(metadata_json)
                        if isinstance(value, dict):
                            ver = value.get('version')
                            if ver == MetadataV1.version:
                                value.pop("version")
                                metadata = MetadataV1(**value)
                            else:
                                logger.debug(
                                    "invalid cache metadata version %s for %s",
                                    ver, key,
                                )
                        else:
                            raise TypeError("Expect a json dict")
                    except (json.JSONDecodeError, TypeError):
                        logger.debug(
                            "invalid cache metadata for %s", key,
                            exc_info=True,
                        )
                yield CacheEntry(key, size, atime, metadata)

            last_key, _, last_atime, _ = rows[-1]
            if remaining is not None:
                remaining -= len(rows)
    finally:
        _close(conn)


def evict_lru(cache_dir: str, size_limit: int) -> None:
    conn = None
    try:
        conn = _connect(cache_dir)
        row_limit = 100
        while True:
            res = conn.execute("""
            DELETE FROM cache WHERE key IN (SELECT key FROM
                (SELECT key, SUM(blob_size) OVER (ORDER BY atime, key) as cumul_size
                    FROM cache ORDER BY atime, key limit ?)
                WHERE cumul_size <= (SELECT SUM(blob_size) - ? FROM cache)
            )
            """, (row_limit, size_limit))
            if res.rowcount < row_limit:
                break
            row_limit *= 10
        conn.commit()
    except sqlite3.Error:
        logger.debug("cache evict failed", exc_info=True)
    finally:
        _close(conn)
