# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import cuda.lang._compile as compile_module
from cuda.lang._ir import hir, ir
from cuda.lang._logging import LoggingConfig
from cuda.lang.compilation import KernelSignature


def _disable_environment_logs(monkeypatch):
    monkeypatch.setattr(compile_module, "get_log_flags", LoggingConfig)


def test_compile_simt_does_not_keep_ir_by_default(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(kernel, [KernelSignature([])])

    assert result.hir is None
    assert result.final_ir is None
    assert result.mlir is None
    assert result.ptx is None
    assert isinstance(result.cubin, bytes)


def test_compile_simt_keeps_ir_without_logging(monkeypatch, capsys):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(
        kernel,
        [KernelSignature([])],
        keep_hir=True,
        keep_final_ir=True,
        keep_mlir=True,
        keep_ptx=True,
    )

    assert isinstance(result.hir, hir.Function)
    assert isinstance(result.final_ir, ir.Region)
    assert isinstance(result.mlir, str)
    assert isinstance(result.ptx, str)
    assert isinstance(result.cubin, bytes)
    assert capsys.readouterr().err == ""


def test_compile_simt_logs_ir_without_keeping(monkeypatch, capsys):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(
        kernel,
        [KernelSignature([])],
        log_hir=True,
        log_ir=True,
        log_mlir=True,
        log_ptx=True,
    )

    assert result.hir is None
    assert result.final_ir is None
    assert result.mlir is None
    assert result.ptx is None

    stderr = capsys.readouterr().err
    assert "cuda.lang HIR dump" in stderr
    assert "cuda.lang IR (pre-transforms) dump" in stderr
    assert "cuda.lang IR (post-transforms) dump" in stderr
    assert "cuda.lang MLIR dump" in stderr
    assert "cuda.lang PTX dump" in stderr
