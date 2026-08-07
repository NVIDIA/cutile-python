# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import zipfile


def _run_crash_dump():
    import torch

    import cuda.tile as ct
    import cuda.tile._compile as tile_compile
    from cuda.tile._exception import TileCompilerExecutionError

    def fail_compilation(*args, **kwargs):
        raise TileCompilerExecutionError(
            1, "error: compiler error", "--compiler-flag", "13.3"
        )

    tile_compile.compile_cubin = fail_compilation

    @ct.kernel
    def kernel():
        pass

    try:
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    except TileCompilerExecutionError:
        pass
    else:
        raise AssertionError("Expected tile compiler failure")


def test_crash_dump_enabled_by_env(tmp_path):
    env = os.environ.copy()
    env["CUDA_TILE_ENABLE_CRASH_DUMP"] = "1"
    env["CUDA_TILE_TEMP_DIR"] = str(tmp_path / "temp")
    env["CUDA_TILE_CACHE_DIR"] = "0"
    subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        check=True,
        cwd=tmp_path,
        env=env,
    )

    [dump_path] = tmp_path.glob("crash_dump_*.zip")
    with zipfile.ZipFile(dump_path) as dump:
        artifacts = dump.namelist()
        assert "debug_info.txt" in artifacts
        assert any(name.endswith(".bytecode") for name in artifacts)
        assert any(name.endswith(".cutileir") for name in artifacts)


if __name__ == "__main__":
    _run_crash_dump()
