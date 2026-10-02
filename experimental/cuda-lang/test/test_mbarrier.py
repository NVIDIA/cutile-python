# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ops import RawLLVMIntrinsic
from cuda.lang.compilation import KernelSignature

from .util import compile_kernel, make_symbolic_tensor, require_hopper_or_newer

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}


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
        mbar = mbar.pointer()

        if tx == 0:
            cl.mbarrier_initialize(mbar, cdx * bdx)

        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=cl.FenceRestriction.mbarrier_initialize(),
        )
        cl.barrier_sync_block_aligned()
        cl._nvvm.barrier_cluster_arrive_aligned()
        cl._nvvm.barrier_cluster_wait_aligned()

        mbar0 = cl.map_shared_to_cluster(mbar, 0)
        cl.mbarrier_arrive(mbar0, scope=cl.MbarrierScope.CLUSTER)

        if rank == 0 and tx == 0:
            while not cl.mbarrier_test_wait_parity(mbar, 0):
                pass
            out[0] = 1

    out = torch.zeros(1, dtype=torch.int32).cuda(0)
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
        kernel,
        signature=KernelSignature(()),
        keep_final_ir=True,
        **HOPPER_TARGET,
    )
    return [
        op.intrinsic
        for block in result.final_ir.blocks
        for op in block.traverse()
        if isinstance(op, RawLLVMIntrinsic)
    ]


@pytest.mark.parametrize("layout", (None, "default"))
def test_initialize_and_invalidate(layout):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        if cl.ensure_constant(cl.static_eval(layout == "default")):
            cl.mbarrier_initialize(mbar, 32)
        else:
            cl.mbarrier_initialize(mbar, 32, layout=layout)
        cl.mbarrier_invalidate(mbar)

    compile_kernel(
        kernel,
        assert_in_ptx=(
            "mbarrier.init.shared::cta.b64",
            "mbarrier.inval.shared.b64",
        ),
        assert_not_in_ptx="mbarrier.init.layout",
        gpu_name="sm_80",
        arch="compute_80",
    )


@pytest.mark.parametrize("layout", (*cl.MbarrierLayout, "V0", "V1"))
@require_hopper_or_newer()
def test_mbarrier_layout(layout):
    layout_version = cl.MbarrierLayout[layout] if isinstance(layout, str) else layout
    other_layout = (
        cl.MbarrierLayout.V1
        if layout_version is cl.MbarrierLayout.V0
        else cl.MbarrierLayout.V0
    )

    @cl.kernel
    def kernel(out):
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        cl.mbarrier_initialize(mbar, 1, layout=layout)
        out[0] = cl.mbarrier_has_layout(mbar, layout)
        out[1] = cl.mbarrier_has_layout(mbar, other_layout)
        cl.mbarrier_invalidate(mbar)

    initialize = f"mbarrier.init.layout::v{layout_version.value}.shared::cta.b64"
    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor([2], cl.int32)]),
        assert_in_ptx=(
            initialize,
            "mbarrier.check_layout.layout::v0",
            "mbarrier.check_layout.layout::v1",
        ),
        **HOPPER_TARGET,
    )
    out = torch.tensor([False, True], dtype=torch.bool).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [True, False]


@pytest.mark.parametrize("layout", (0, 1, "V2", "v0"))
@pytest.mark.parametrize("operation", (cl.mbarrier_initialize, cl.mbarrier_has_layout))
def test_mbarrier_bad_layout(operation, layout):
    initialize = operation is cl.mbarrier_initialize

    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        if initialize:
            operation(mbar, 1, layout=layout)
        else:
            operation(mbar, layout)

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="MbarrierLayout"),
        **HOPPER_TARGET,
    )


ARRIVE_MEMORY_ORDERS = [cl.MemoryOrder.RELEASE, cl.MemoryOrder.RELAXED]
WAIT_MEMORY_ORDERS = [cl.MemoryOrder.ACQUIRE, cl.MemoryOrder.RELAXED]


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("memory_order", ARRIVE_MEMORY_ORDERS)
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("expect_transaction", [False, True])
def test_arrive_intrinsic_name(expect_transaction, drop, memory_order, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).pointer()
        if expect_transaction:
            cl.mbarrier_arrive_expect_transaction(
                mbar, 128, drop=drop, scope=scope, memory_order=memory_order
            )
        else:
            cl.mbarrier_arrive(mbar, 1, drop=drop, scope=scope, memory_order=memory_order)

    expected = "llvm.nvvm.mbarrier.arrive"
    if drop:
        expected += ".drop"
    if expect_transaction:
        expected += ".expect.tx"
    if memory_order is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


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
        ).pointer()
        operation(mbar, 64, scope=scope)

    expected = intrinsic + f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("memory_order", WAIT_MEMORY_ORDERS)
@pytest.mark.parametrize("parity", [False, True])
def test_test_wait_intrinsic_name(parity, memory_order, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).pointer()
        if parity:
            cl.mbarrier_test_wait_parity(mbar, 0, scope=scope, memory_order=memory_order)
        else:
            cl.mbarrier_test_wait(mbar, cl.uint64(0), scope=scope, memory_order=memory_order)

    expected = "llvm.nvvm.mbarrier.test.wait"
    if parity:
        expected += ".parity"
    if memory_order is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("time_hint", [None, 1000])
@pytest.mark.parametrize("memory_order", WAIT_MEMORY_ORDERS)
@pytest.mark.parametrize("parity", [False, True])
def test_try_wait_intrinsic_name(parity, memory_order, time_hint, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).pointer()
        if parity:
            cl.mbarrier_try_wait_parity(
                mbar, 0, time_hint=time_hint, scope=scope, memory_order=memory_order
            )
        else:
            cl.mbarrier_try_wait(
                mbar, cl.uint64(0), time_hint=time_hint, scope=scope, memory_order=memory_order
            )

    expected = "llvm.nvvm.mbarrier.try.wait"
    if parity:
        expected += ".parity"
    if time_hint is not None:
        expected += ".tl"
    if memory_order is cl.MemoryOrder.RELAXED:
        expected += ".relaxed"
    expected += f".scope.{scope.value}.space.cta"
    assert expected in _get_intrinsics(kernel)
