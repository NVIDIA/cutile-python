# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
from cuda.lang.compilation import KernelSignature

from .filecheck_utils import filecheck, get_source
from .ir_utils import (
    get_ir,
    make_symbolic_scalar,
    make_symbolic_tensor,
)


def require_blackwell_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (10, 0),
        reason="feature requires Blackwell or newer",
    )


def require_blackwell_cc100():
    cc = get_compute_capability()
    return pytest.mark.skipif(
        cc.major != 10,
        reason="feature requires Blackwell with compute capability 100",
    )


def require_hopper_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (9, 0),
        reason="feature requires Hopper or newer",
    )


def compile_kernel(
    kernel,
    signature=KernelSignature([]),
    assert_in_ptx=None,
    assert_not_in_ptx=None,
    assert_in_mlir=None,
    assert_not_in_mlir=None,
    assert_in_nvvm=None,
    assert_not_in_nvvm=None,
    filecheck_ptx=None,
    filecheck_mlir=None,
    raises=None,
    **compile_simt_kwargs,
):
    if raises is not None:
        message = "If the `raises` argument was passed, then we can't check the IR"
        assert assert_in_ptx is None, message
        assert assert_not_in_ptx is None, message
        assert assert_in_mlir is None, message
        assert assert_not_in_mlir is None, message
        assert assert_in_nvvm is None, message
        assert assert_not_in_nvvm is None, message
        assert filecheck_ptx is None, message
        assert filecheck_mlir is None, message

        with raises:
            cl.compile_simt(kernel, [signature], **compile_simt_kwargs)

        return

    compiled = cl.compile_simt(
        kernel,
        [signature],
        keep_ptx=True,
        keep_mlir=True,
        keep_nvvm=assert_in_nvvm is not None or assert_not_in_nvvm is not None,
        **compile_simt_kwargs,
    )
    assert compiled.mlir
    assert compiled.ptx
    if assert_in_nvvm is not None or assert_not_in_nvvm is not None:
        assert compiled.nvvm

    def tuple_or_str_check(check, scrutinee, predicate=lambda x, y: x in y):
        match check:
            case None:
                pass
            case str():
                assert predicate(check, scrutinee), (
                    f"{predicate=} failed with\n{check=}\n{scrutinee}"
                )
            case tuple() | list():
                for single_check in check:
                    assert predicate(single_check, scrutinee), (
                        f"{predicate=} failed with\n{single_check=}\n{scrutinee}"
                    )
            case _:
                assert False, "expected check to be str or iterable of str"

    def is_in(x, y):
        return x in y

    def is_not_in(x, y):
        return x not in y

    tuple_or_str_check(assert_in_ptx, compiled.ptx, is_in)
    tuple_or_str_check(assert_not_in_ptx, compiled.ptx, is_not_in)

    tuple_or_str_check(assert_in_mlir, compiled.mlir, is_in)
    tuple_or_str_check(assert_not_in_mlir, compiled.mlir, is_not_in)

    tuple_or_str_check(assert_in_nvvm, compiled.nvvm, is_in)
    tuple_or_str_check(assert_not_in_nvvm, compiled.nvvm, is_not_in)

    if filecheck_ptx is not None:
        assert isinstance(filecheck_ptx, str)
        filecheck(compiled.ptx, filecheck_ptx)

    if filecheck_mlir is not None:
        assert isinstance(filecheck_mlir, str)
        filecheck(compiled.mlir, filecheck_mlir)

    return compiled


__all__ = (
    "filecheck",
    "get_source",
    "get_ir",
    "make_symbolic_scalar",
    "make_symbolic_tensor",
    "compile_kernel",
    "require_hopper_or_newer",
    "require_blackwell_or_newer",
)
