# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import time
from pathlib import Path

import pytest

from cutile_cache import _cache
from cutile_cache._cache import (
    MetadataV1,
    cache_entries,
    cache_key,
    cache_lookup,
    cache_store,
    evict_lru,
)


def test_cache_key_equal():
    k1 = cache_key("v1", "sm_90", 3, b"data")
    k2 = cache_key("v1", "sm_90", 3, b"data")
    assert k1 == k2


def test_cache_key_differs():
    base = cache_key("v1", "sm_90", 3, b"data")
    assert cache_key("v2", "sm_90", 3, b"data") != base
    assert cache_key("v1", "sm_80", 3, b"data") != base
    assert cache_key("v1", "sm_90", 2, b"data") != base
    assert cache_key("v1", "sm_90", 3, b"other") != base


def test_cache_key_device_debug_differs():
    base = cache_key("v1", "sm_90", 0, b"data", False)
    assert cache_key("v1", "sm_90", 0, b"data", True) != base


def test_metadata():
    first = "--- !Passed\nName: First\n...\n"
    second = "--- !Failure\nName: Second\n...\n"
    dup = "--- !Failure\nName: Second\n...\n"
    remarks = first + second + dup

    metadata = MetadataV1(
        kernel_names=["kernel_Knew"],
        compiler_version="tileiras 13.4",
        compilation_timestamp=1.0,
        compilation_time_seconds=0.125,
        remarks=remarks,
    ).to_dict()

    assert metadata["version"] == 1
    assert metadata["remarks"] == first + second


@pytest.fixture
def cache_env(tmp_path):
    cache_dir = str(tmp_path / "cache")
    return cache_dir, tmp_path


def test_store_then_lookup(cache_env):
    cache_dir, tmp_path = cache_env
    key = cache_key("v1", "sm_90", 3, b"data")
    content = b"\x7fELF_fake_cubin_data"

    cache_store(cache_dir, key, content)

    result = cache_lookup(cache_dir, key)
    assert result is not None
    assert result == content


def test_existing_database_is_migrated(cache_env):
    cache_dir, _ = cache_env
    Path(cache_dir).mkdir()
    db_path = Path(cache_dir) / "cache.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE cache (
            key TEXT PRIMARY KEY,
            blob BLOB NOT NULL,
            blob_size INTEGER NOT NULL,
            atime REAL NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO cache (key, blob, blob_size, atime) VALUES (?, ?, ?, ?)",
        ("old", b"old", 3, 1.0),
    )
    conn.commit()
    conn.close()

    metadata = MetadataV1(compiler_version="foo")
    cache_store(cache_dir, "new", b"new", metadata.to_dict())

    entries = {entry.key: entry for entry in cache_entries(cache_dir)}
    assert entries["old"].metadata == MetadataV1()
    assert entries["new"].metadata == metadata


def test_cache_entries_does_not_block_writes(cache_env, monkeypatch):
    cache_dir, _ = cache_env
    cache_store(cache_dir, "first", b"first")
    cache_store(cache_dir, "second", b"second")
    monkeypatch.setattr(_cache, "_CACHE_ENTRY_BATCH_SIZE", 1)

    entries = cache_entries(cache_dir)
    next(entries)
    try:
        db_path = Path(cache_dir) / "cache.db"
        conn = sqlite3.connect(db_path, timeout=0)
        try:
            conn.execute(
                "INSERT INTO cache (key, blob, blob_size, atime, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                ("new", b"new", 3, time.time(), None),
            )
            conn.commit()
        finally:
            conn.close()
    finally:
        entries.close()


def test_lookup_updates_atime(cache_env):
    cache_dir, tmp_path = cache_env
    key = cache_key("v1", "sm_90", 3, b"data")

    cache_store(cache_dir, key, b"data")

    # Manually set old atime in DB
    import os
    db_path = os.path.join(cache_dir, "cache.db")
    old_time = time.time() - 1000
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE cache SET atime = ? WHERE key = ?", (old_time, key))
    conn.commit()
    conn.close()

    cache_lookup(cache_dir, key)

    conn = sqlite3.connect(db_path)
    atime = conn.execute(
        "SELECT atime FROM cache WHERE key = ?", (key,)
    ).fetchone()[0]
    conn.close()
    assert atime > old_time


def test_lookup_miss(cache_env):
    cache_dir, _ = cache_env

    result = cache_lookup(cache_dir, "a" * 64)
    assert result is None


def test_evict_lru(cache_env):
    cache_dir, tmp_path = cache_env
    import os
    db_path = os.path.join(cache_dir, "cache.db")

    # Populate 5 entries (1000 bytes each, 5000 total)
    keys = []
    for i in range(5):
        key = cache_key(str(i), "sm_90", 3, b"data")
        keys.append(key)
        cache_store(cache_dir, key, b"x" * 1000)

    # Set controlled atimes so eviction order is deterministic
    conn = sqlite3.connect(db_path)
    for i, key in enumerate(keys):
        conn.execute(
            "UPDATE cache SET atime = ? WHERE key = ?",
            (float(i), key)
        )
    conn.commit()
    conn.close()

    # Evict to keep 3000 bytes; newest 3 survive (indices 2, 3, 4)
    evict_lru(cache_dir, 3000)

    remaining = [k for k in keys
                 if cache_lookup(cache_dir, k) is not None]
    assert remaining == keys[2:]
