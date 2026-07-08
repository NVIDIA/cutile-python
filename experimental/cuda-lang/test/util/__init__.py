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
    filecheck_ptx=None,
    filecheck_mlir=None,
    raises=None,
    **compile_simt_kwargs,
):
    if raises is not None:
        assert assert_in_ptx is None
        assert assert_not_in_ptx is None
        assert assert_in_mlir is None
        assert assert_not_in_mlir is None
        assert filecheck_ptx is None
        assert filecheck_mlir is None
        with raises:
            cl.compile_simt(kernel, [signature], **compile_simt_kwargs)
        return

    compiled = cl.compile_simt(
        kernel,
        [signature],
        keep_ptx=True,
        keep_mlir=True,
        **compile_simt_kwargs,
    )
    assert compiled.mlir
    assert compiled.ptx

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
                assert False, "expected assert_in_ptx to be str or iterable of str"

    tuple_or_str_check(assert_in_ptx, compiled.ptx)
    tuple_or_str_check(assert_not_in_ptx, compiled.ptx, lambda x, y: x not in y)
    tuple_or_str_check(assert_in_mlir, compiled.mlir)
    tuple_or_str_check(assert_not_in_mlir, compiled.mlir, lambda x, y: x not in y)

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
