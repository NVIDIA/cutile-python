# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import cuda.tile as ct
import cuda.lang._compile as compile_module
import cuda.lang._logging as logging_module
import pytest
from cuda.lang._compiler_options import CompilerOptions
from cuda.tile._exception import UnsupportedCallError
from cuda.tile._execution import (is_function_allowed_in, stub)

from cuda.lang._ir import hir, ir
from cuda.lang._logging import LoggingConfig
from cuda.lang._timing import CompilationTimings
from cuda.lang.compilation import KernelSignature


def _disable_environment_logs(monkeypatch):
    monkeypatch.setattr(compile_module, "get_log_flags", LoggingConfig)


def test_function_execution_spaces():
    @cl.function
    def simt_function():
        pass

    assert is_function_allowed_in(simt_function, "host")
    assert is_function_allowed_in(simt_function, "device", device_execution_space="simt")
    assert not is_function_allowed_in(simt_function, "device", device_execution_space="tile")

    @cl.function(simt=False)
    def host_function():
        pass

    assert is_function_allowed_in(host_function, "host")
    assert not is_function_allowed_in(host_function, "device")
    assert not is_function_allowed_in(host_function, "device", device_execution_space="simt")


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("tile, simt", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("compiled_host", [False, True])
def test_stub_execution_spaces(host, tile, simt, compiled_host):
    @stub(host=host, tile=tile, simt=simt, compiled_host=compiled_host)
    def helper():
        return 42

    assert is_function_allowed_in(helper, "host") == compiled_host
    assert is_function_allowed_in(helper, "device", device_execution_space="tile") == tile
    assert is_function_allowed_in(helper, "device", device_execution_space="simt") == simt
    if host:
        assert helper() == 42
    else:
        with pytest.raises(RuntimeError, match="Device functions can only be called"):
            helper()


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("tile", [False, True])
def test_ct_function_execution_spaces(host, tile):
    @ct.function(host=host, tile=tile)
    def helper():
        return 42

    assert is_function_allowed_in(helper, "host")
    assert is_function_allowed_in(helper, "device", device_execution_space="tile") == tile
    assert not is_function_allowed_in(helper, "device", device_execution_space="simt")
    if host:
        assert helper() == 42
    else:
        with pytest.raises(
            RuntimeError, match="Device functions can only be called from device code"
        ):
            helper()


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("tile, simt", [(False, False), (False, True), (True, False), (True, True)])
def test_cl_function_execution_spaces(host, tile, simt):
    @cl.function(host=host, tile=tile, simt=simt)
    def helper():
        return 42

    assert is_function_allowed_in(helper, "host")
    assert is_function_allowed_in(helper, "device", device_execution_space="tile") == tile
    assert is_function_allowed_in(helper, "device", device_execution_space="simt") == simt
    if host:
        assert helper() == 42
    else:
        with pytest.raises(
            RuntimeError, match="Device functions can only be called from device code"
        ):
            helper()


def test_compile_simt_honors_execution_space():
    @cl.function(host=True, simt=False)
    def host_only():
        pass

    def kernel():
        host_only()

    with pytest.raises(
        UnsupportedCallError,
        match=r"host_only\(\) is not supported in simt code",
    ):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_compile_simt_does_not_keep_ir_by_default(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(kernel, [KernelSignature([])])

    assert result.hir is None
    assert result.final_ir is None
    assert result.mlir is None
    assert result.nvvm is None
    assert result.ptx is None
    assert isinstance(result.cubin, bytes)
    assert result.timings is None


def test_compile_simt_keeps_timings(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(
        kernel,
        [KernelSignature([])],
        keep_timings=True,
    )

    assert isinstance(result.timings, CompilationTimings)
    assert all(value >= 0 for value in result.timings.phases_ns.values())


def test_compile_simt_logs_timings_from_environment(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_LANG_LOGS", "timings")
    monkeypatch.setattr(logging_module, "_config", None)

    def kernel():
        pass

    result = cl.compile_simt(kernel, [KernelSignature([])])

    stderr = capsys.readouterr().err
    assert "cuda.lang Timings dump" in stderr
    assert result.timings is None


def test_compile_simt_keeps_line_information(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(
        kernel,
        [KernelSignature([])],
        compiler_options=CompilerOptions(debug_info="line"),
        keep_mlir=True,
        keep_nvvm=True,
        keep_ptx=True,
    )

    assert result.mlir is not None
    assert __file__ in result.mlir
    assert "loc(" in result.mlir
    assert result.nvvm is not None
    assert "!DICompileUnit" in result.nvvm
    assert "!DILocation" in result.nvvm
    assert result.ptx is not None
    assert ".file" in result.ptx
    assert ".loc" in result.ptx


def test_compile_simt_keeps_call_site_information(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def callee():
        return cl.thread_index(0)

    def kernel():
        print(callee())

    result = cl.compile_simt(
        kernel,
        [KernelSignature([])],
        compiler_options=CompilerOptions(debug_info="line"),
        keep_mlir=True,
        keep_nvvm=True,
    )

    assert result.mlir is not None
    assert "loc(callsite(" in result.mlir
    assert result.nvvm is not None
    assert "inlinedAt:" in result.nvvm


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
        keep_nvvm=True,
        keep_ptx=True,
    )

    assert isinstance(result.hir, hir.Function)
    assert isinstance(result.final_ir, ir.Region)
    assert isinstance(result.mlir, str)
    assert isinstance(result.nvvm, str)
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
        log_nvvm=True,
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
    assert "cuda.lang NVVM dump" in stderr
    assert "cuda.lang PTX dump" in stderr


def test_compile_simt_logs_nvvm_without_keeping(monkeypatch, capsys):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(kernel, [KernelSignature([])], log_nvvm=True)

    assert result.nvvm is None
    stderr = capsys.readouterr().err
    assert "cuda.lang NVVM dump" in stderr
    assert 'target triple = "nvptx64-nvidia-cuda"' in stderr


def test_compile_simt_keeps_nvvm(monkeypatch):
    _disable_environment_logs(monkeypatch)

    def kernel():
        pass

    result = cl.compile_simt(kernel, [KernelSignature([])], keep_nvvm=True)

    assert isinstance(result.nvvm, str)
    assert 'target triple = "nvptx64-nvidia-cuda"' in result.nvvm
