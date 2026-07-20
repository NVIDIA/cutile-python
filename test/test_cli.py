# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from pathlib import Path

from cutile_cache._cache import MetadataV1, cache_store
from cutile_cache import _cli


def _store_entry(cache_dir: Path, key: str, atime: float, ctime: float, kernel: str) -> None:
    metadata = MetadataV1(
        kernel_names=[kernel],
        compiler_version=(
            "tileiras: NVIDIA (R) Cuda Tile IR optimizing assembler\n"
            "Copyright (c) 2005-2026 NVIDIA Corporation\n"
            "Cuda compilation tools, release 13.3, V13.3.99\n"
        ),
        compilation_timestamp=ctime,
        compilation_time_seconds=0.125,
        remarks=("--- !Passed\nName: RemarkTensorCoreMMA\n" * 2),
    )
    cache_store(str(cache_dir), key, b"x" * 2048, metadata.to_dict())
    conn = sqlite3.connect(cache_dir / "cache.db")
    conn.execute("UPDATE cache SET atime = ? WHERE key = ?", (atime, key))
    conn.commit()
    conn.close()


def test_cache_log(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    ctime = 1_700_000_000.0
    atime = ctime + 1
    _store_entry(cache_dir, "a" * 64, atime, ctime, "vector_add_Kfoo")
    output = []
    monkeypatch.setenv("CUDA_TILE_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        _cli, "_page_entries",
        lambda entries: output.append("\n\n".join(_cli._format_entry(e) for e in entries)),
    )

    assert _cli.main(["log"]) == 0

    [text] = output
    expected = f"""commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Kernel:     vector_add_Kfoo
Compiler:   tileiras 13.3.99
Date:       {_cli.format_date(ctime)}
Last used:  {_cli.format_date(atime)}
Duration:   125.0 ms
CUBIN:      2,048 bytes (2.0 KiB)

Remarks:
    --- !Passed
    Name: RemarkTensorCoreMMA"""

    assert text.strip() == expected
