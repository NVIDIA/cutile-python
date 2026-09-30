# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from io import BytesIO

import cuda.tile as ct
import pytest
import torch

from cuda.tile import _compile
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileCompilerExecutionError
from cuda.tile.compilation import CallingConvention, KernelSignature, export_kernel
from conftest import requires_tileiras


@ct.kernel
def _empty_kernel():
    pass


def _check_dump(tmp_path, capsys, bin_available):
    dumps = list(tmp_path.glob("*.tileir"))
    stderr = capsys.readouterr().err
    if bin_available:
        assert len(dumps) == 1
        text = dumps[0].read_text()
        assert "entry @_empty_kernel" in text
        assert text in stderr
    else:
        assert dumps == []
        assert "Failed to disassemble TileIR: tileirdisasm is unavailable" in stderr


@pytest.mark.parametrize("bin_available", [
    pytest.param(True, marks=requires_tileiras(BytecodeVersion.V_13_4)),
    False,
])
def test_launch_dumps_tileir(monkeypatch, tmp_path, capsys, bin_available):
    if not bin_available:
        def unavailable():
            raise FileNotFoundError("tileirdisasm is unavailable")
        monkeypatch.setattr(_compile, "_find_tileirdisasm_bin", unavailable)
    monkeypatch.setattr(_compile, "CUDA_TILE_DUMP_TILEIR", str(tmp_path))
    monkeypatch.setattr(_compile.default_tile_context.config, "log_tileir", True)
    monkeypatch.setattr(_compile.default_tile_context.config, "cache_dir", None)

    kernel = ct.kernel(_empty_kernel._pyfunc)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
    torch.cuda.synchronize()

    _check_dump(tmp_path, capsys, bin_available)


@pytest.mark.parametrize("bin_available", [
    pytest.param(True, marks=requires_tileiras(BytecodeVersion.V_13_4)),
    False,
])
def test_export_dumps_tileir(monkeypatch, tmp_path, capsys, bin_available):
    if not bin_available:
        def unavailable():
            raise FileNotFoundError("tileirdisasm is unavailable")
        monkeypatch.setattr(_compile, "_find_tileirdisasm_bin", unavailable)
    monkeypatch.setattr(_compile, "CUDA_TILE_DUMP_TILEIR", str(tmp_path))
    monkeypatch.setattr(_compile.default_tile_context.config, "log_tileir", True)

    output = BytesIO()
    export_kernel(
        _empty_kernel,
        [KernelSignature([], CallingConvention.cutile_python_v1())],
        output,
        gpu_code="sm_100",
        output_format="tileir_bytecode",
        bytecode_version="13.1",
    )

    assert output.getvalue() != b""
    _check_dump(tmp_path, capsys, bin_available)


def test_disassembler_error(monkeypatch, tmp_path):
    monkeypatch.setattr(_compile, "_find_tileirdisasm_bin", lambda: "/tool/bin/tileirdisasm")
    monkeypatch.setattr(
        _compile.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "bad bytecode"),
    )

    with pytest.raises(TileCompilerExecutionError,
                       match="Return code 1\\ntileirdisasm: bad bytecode") as exc_info:
        _compile._bytecode_to_mlir_text(b"invalid", preview_features=("simt",),
                                        temp_dir=str(tmp_path))
    assert exc_info.value.compiler_flags == "--preview=simt"
