# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
from test.util import compile_kernel, require_hopper_or_newer
import pytest


@require_hopper_or_newer()
@pytest.mark.parametrize(
    "kwarg",
    (
        dict(max_blocks_per_cluster=-1),
        dict(max_threads_per_block=-1),
        dict(max_registers_per_thread=-1),
        dict(min_blocks_per_sm=-1),
        dict(max_blocks_per_cluster=1.0),
        dict(max_threads_per_block=1.0),
        dict(max_registers_per_thread=1.0),
        dict(min_blocks_per_sm=1.0),
    ),
)
def test_bad_kernel_attribute_types(kwarg):
    with pytest.raises(TypeCheckingError):

        @cl.kernel(**kwarg)
        def foo():
            pass


def test_max_threads_per_block_tuple_needs_extension_1():
    @cl.kernel(max_threads_per_block=(8,))
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .maxntid 8
        """,
    )


def test_max_threads_per_block_tuple_needs_extension_2():
    @cl.kernel(max_threads_per_block=(8, 16))
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .maxntid 8, 16
        """,
    )


def test_max_threads_per_block():
    @cl.kernel(max_threads_per_block=(2, 3, 4))
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .maxntid 2, 3, 4
        """,
    )


@require_hopper_or_newer()
def test_max_blocks_per_cluster():
    @cl.kernel(max_blocks_per_cluster=2)
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .maxclusterrank 2
        """,
    )


def test_max_registers_per_thread():
    @cl.kernel(max_registers_per_thread=64)
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .maxnreg 64
        """,
    )


def test_min_blocks_per_sm():
    @cl.kernel(min_blocks_per_sm=2)
    def foo():
        pass

    compile_kernel(
        foo,
        compiler_options=foo._compiler_options,
        filecheck_ptx="""
        CHECK: .entry foo
        CHECK-NEXT: .minnctapersm 2
        """,
    )
