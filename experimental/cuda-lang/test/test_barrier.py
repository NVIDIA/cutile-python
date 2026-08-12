# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._exception import CompilerExecutionError
from cuda.tile import TileStaticAssertionError
from .util import compile_kernel
import pytest

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}


def barrier_sync_block_cases():
    for number_of_threads in (None, 5, 32):
        for aligned in (True, False, 3):
            if aligned == 3:
                raises = pytest.raises(
                    Exception,
                    match="Expected constant of type bool",
                )
                yield number_of_threads, aligned, None, raises
            elif number_of_threads == 5:
                raises = pytest.raises(
                    CompilerExecutionError,
                    match=(
                        "Number of threads participating in barrier must be "
                        "in multiple of warp size"
                    ),
                )
                yield number_of_threads, aligned, None, raises
            else:
                ptx = ("barrier" if not aligned else "bar") + ".sync"
                yield number_of_threads, aligned, ptx, None


@pytest.mark.parametrize(
    "number_of_threads, aligned, expect, raises", barrier_sync_block_cases()
)
def test_barrier_sync_block(number_of_threads, aligned, expect, raises):
    def kernel():
        cl.barrier_sync_block(number_of_threads, 7, aligned=aligned)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


def barrier_arrive_block_cases():
    for number_of_threads in (None, 5, 32):
        for aligned in (True, False, 3):
            if not isinstance(aligned, bool):
                raises = pytest.raises(
                    Exception,
                    match="Expected constant of type bool",
                )
                yield number_of_threads, aligned, None, raises
            elif number_of_threads is None:
                raises = pytest.raises(
                    Exception,
                    match="Expected a scalar value, but given value has type None",
                )
                yield number_of_threads, aligned, None, raises
            elif number_of_threads == 5:
                raises = pytest.raises(
                    CompilerExecutionError,
                    match=(
                        "Number of threads participating in barrier must "
                        "be in multiple of warp size"
                    ),
                )
                yield number_of_threads, aligned, None, raises
            else:
                ptx = ("barrier" if not aligned else "bar") + ".arrive"
                yield number_of_threads, aligned, ptx, None


@pytest.mark.parametrize(
    "number_of_threads, aligned, expect, raises", barrier_arrive_block_cases()
)
def test_barrier_arrive_block(number_of_threads, aligned, expect, raises):
    def kernel():
        cl.barrier_arrive_block(number_of_threads, 7, aligned=aligned)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


def barrier_reduce_block_cases():
    for op in (*cl.BarrierReductionKind, None, 5):
        for predicate in (True, False, None, 5):
            for number_of_threads in (None, 5, 32):
                for aligned in (True, False, 3):
                    if not isinstance(op, cl.BarrierReductionKind):
                        raises = pytest.raises(
                            Exception,
                            match="BarrierReductionKind",
                        )
                        yield op, predicate, number_of_threads, aligned, None, raises
                    elif not isinstance(predicate, bool):
                        raises = pytest.raises(
                            Exception,
                            match="Expected (a scalar|boolean scalar)",
                        )
                        yield op, predicate, number_of_threads, aligned, None, raises
                    elif not isinstance(aligned, bool):
                        raises = pytest.raises(
                            Exception,
                            match="Expected (constant of type bool|a boolean constant)",
                        )
                        yield op, predicate, number_of_threads, aligned, None, raises
                    elif number_of_threads == 5:
                        raises = pytest.raises(
                            CompilerExecutionError,
                            match=(
                                "Number of threads participating in barrier "
                                "must be in multiple of warp size"
                            ),
                        )
                        yield op, predicate, number_of_threads, aligned, None, raises
                    else:
                        ptx_op = {
                            cl.BarrierReductionKind.POP_COUNT: "popc",
                            cl.BarrierReductionKind.AND: "and",
                            cl.BarrierReductionKind.OR: "or",
                        }[op]
                        ptx = ("barrier" if not aligned else "bar") + f".red.{ptx_op}"
                        yield op, predicate, number_of_threads, aligned, ptx, None


@pytest.mark.parametrize(
    "op, predicate, number_of_threads, aligned, expect, raises",
    barrier_reduce_block_cases(),
)
def test_barrier_reduce_block(
    op, predicate, number_of_threads, aligned, expect, raises
):
    def kernel():
        cl.barrier_reduce_block(op, predicate, number_of_threads, 7, aligned=aligned)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


def barrier_arrive_cluster_cases():
    valid_orders = (cl.MemoryOrder.RELEASE, cl.MemoryOrder.RELAXED)
    for aligned in (True, False, 3):
        for order in (*tuple(cl.MemoryOrder), 5, None):
            if not isinstance(aligned, bool):
                raises = pytest.raises(
                    Exception,
                    match="Expected constant of type bool",
                )
                yield aligned, order, None, raises
            elif order not in tuple(cl.MemoryOrder):
                raises = pytest.raises(
                    Exception,
                    match="Expected enum constant of type MemoryOrder",
                )
                yield aligned, order, None, raises
            elif order not in valid_orders:
                raises = pytest.raises(
                    TileStaticAssertionError,
                    match="memory_order must be MemoryOrder.RELEASE or MemoryOrder.RELAXED",
                )
                yield aligned, order, None, raises
            else:
                expect = "barrier.cluster.arrive"
                expect += ".relaxed" if order == cl.MemoryOrder.RELAXED else ""
                expect += ".aligned" if aligned else ""
                yield aligned, order, expect, None


@pytest.mark.parametrize(
    "aligned, order, expect, raises", barrier_arrive_cluster_cases()
)
def test_barrier_arrive_cluster(aligned, order, expect, raises):
    def kernel():
        cl.barrier_arrive_cluster(aligned=aligned, memory_order=order)

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        raises=raises,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "aligned, expect, raises",
    (
        (True, "barrier.cluster.wait.aligned", None),
        (False, "barrier.cluster.wait", None),
        (None, None, pytest.raises(Exception, match="Expected constant of type bool")),
        (5, None, pytest.raises(Exception, match="Expected constant of type bool")),
    ),
)
def test_barrier_wait_cluster(aligned, expect, raises):
    def kernel():
        cl.barrier_wait_cluster(aligned=aligned)

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        raises=raises,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "aligned, expect, raises",
    (
        (
            True,
            ("barrier.cluster.arrive.aligned", "barrier.cluster.wait.aligned"),
            None,
        ),
        (
            False,
            ("barrier.cluster.arrive", "barrier.cluster.wait"),
            None,
        ),
        (None, None, pytest.raises(Exception, match="Expected constant of type bool")),
        (5, None, pytest.raises(Exception, match="Expected constant of type bool")),
    ),
)
def test_barrier_sync_cluster(aligned, expect, raises):
    def kernel():
        cl.barrier_sync_cluster(aligned=aligned)

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        raises=raises,
        **HOPPER_TARGET,
    )


def test_barrier_sync_warp():
    def kernel():
        cl.barrier_sync_warp(32)

    compile_kernel(kernel, assert_in_ptx="bar.warp.sync")
