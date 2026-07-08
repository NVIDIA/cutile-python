# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import cuda.lang as cl
from cuda.lang._ir.ops import RawNVVMIntrinsic
from cuda.lang.compilation import KernelSignature

from .util import compile_kernel, require_hopper_or_newer


@require_hopper_or_newer()
def test_cluster_barriers():
    '''
    Allocate an mbarrier and get a pointer to the rank-0 CTA's barrier after
    a cluster-wide sync.
    Arrive at rank 0's mbarrier, and then rank 0 observes it's completion.
    '''

    @cl.kernel()
    def kernel(out):
        rank = cl.block_in_cluster_index(0)
        cdx = cl.block_in_cluster_count(0)
        tx = cl.thread_index(0)
        bdx = cl.thread_count(0)
        mbar = cl.shared_array(shape=(), dtype=cl.mbarrier, alignment=8)
        mbar = mbar.get_base_pointer()

        if tx == 0:
            cl.mbarrier_initialize(mbar, cdx * bdx)

        cl._nvvm.fence_mbarrier_init_release_cluster()
        cl.barrier_sync_block()
        cl._nvvm.barrier_cluster_arrive_aligned()
        cl._nvvm.barrier_cluster_wait_aligned()

        mbar0 = cl.map_shared_to_cluster(mbar, 0)
        cl.mbarrier_arrive(mbar0, scope=cl.MbarrierScope.CLUSTER)

        if rank == 0 and tx == 0:
            while not cl.mbarrier_test_wait_parity(mbar, 0):
                pass
            out[0] = 1

    out = torch.zeros(1, dtype=torch.int32).cuda()
    # Grid == cluster so there's exactly one cluster of 2 CTAs. 32 threads/CTA
    # gives 64 total arrives at rank 0's mbarrier.
    # Initialize cdx * bdx barrier participants.
    cl.launch(
        torch.cuda.current_stream(),
        (2, 1, 1),
        (32, 1, 1),
        kernel,
        (out,),
        block_in_cluster_count=(2, 1, 1),
    )
    assert out.cpu().tolist() == [1]


# compile-only tests that cover the full api


SCOPES = [cl.MbarrierScope.BLOCK, cl.MbarrierScope.CLUSTER]


def _get_intrinsics(kernel):
    result = compile_kernel(
        kernel, signature=KernelSignature(()), keep_final_ir=True
    )
    return [
        op.intrinsic
        for block in result.final_ir.blocks
        for op in block.traverse()
        if isinstance(op, RawNVVMIntrinsic)
    ]


@require_hopper_or_newer()
def test_initialize_and_invalidate():
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        cl.mbarrier_initialize(mbar, 32)
        cl.mbarrier_invalidate(mbar)

    names = _get_intrinsics(kernel)
    assert "llvm.nvvm.mbarrier.init.shared" in names
    assert "llvm.nvvm.mbarrier.inval.shared" in names


ARRIVE_ORDERINGS = [cl.MemoryOrder.RELEASE, cl.MemoryOrder.RELAXED]
WAIT_ORDERINGS = [cl.MemoryOrder.ACQUIRE, cl.MemoryOrder.RELAXED]


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("ordering", ARRIVE_ORDERINGS)
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("expect_transaction", [False, True])
def test_arrive_intrinsic_name(expect_transaction, drop, ordering, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if expect_transaction:
            cl.mbarrier_arrive_expect_transaction(
                mbar, 128, drop=drop, scope=scope, memory_order=ordering
            )
        else:
            cl.mbarrier_arrive(mbar, 1, drop=drop, scope=scope, memory_order=ordering)

    expected = "llvm.nvvm.mbarrier.arrive"
    if drop:
        expected += ".drop"
    if expect_transaction:
        expected += ".expect.tx"
    if ordering is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize(
    ("operation", "intrinsic"),
    (
        (cl.mbarrier_expect_transaction, "llvm.nvvm.mbarrier.expect.tx"),
        (cl.mbarrier_complete_transaction, "llvm.nvvm.mbarrier.complete.tx"),
    ),
)
def test_expect_complete_transaction_intrinsic_name(operation, intrinsic, scope):

    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        operation(mbar, 64, scope=scope)

    expected = intrinsic + f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("ordering", WAIT_ORDERINGS)
@pytest.mark.parametrize("parity", [False, True])
def test_test_wait_intrinsic_name(parity, ordering, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if parity:
            cl.mbarrier_test_wait_parity(mbar, 0, scope=scope, memory_order=ordering)
        else:
            cl.mbarrier_test_wait(mbar, cl.uint64(0), scope=scope, memory_order=ordering)

    expected = "llvm.nvvm.mbarrier.test.wait"
    if parity:
        expected += ".parity"
    if ordering is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("time_hint", [None, 1000])
@pytest.mark.parametrize("ordering", WAIT_ORDERINGS)
@pytest.mark.parametrize("parity", [False, True])
def test_try_wait_intrinsic_name(parity, ordering, time_hint, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if parity:
            cl.mbarrier_try_wait_parity(
                mbar, 0, time_hint=time_hint, scope=scope, memory_order=ordering
            )
        else:
            cl.mbarrier_try_wait(
                mbar, cl.uint64(0), time_hint=time_hint, scope=scope, memory_order=ordering
            )

    expected = "llvm.nvvm.mbarrier.try.wait"
    if parity:
        expected += ".parity"
    if time_hint is not None:
        expected += ".tl"
    if ordering is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)
