# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
import torch
import pytest

from cuda.lang._exception import TypeCheckingError

cc = get_compute_capability()

clc_bytes = cl.cluster_launch_control_token.bitwidth // 8


@cl.kernel
def bad_clc_memspace_1():
    smem = cl.shared_array(1, cl.mbarrier).get_base_pointer()
    with cl.local_array(1, cl.cluster_launch_control_token) as larr:
        cl.cluster_launch_control_try_cancel(larr.get_base_pointer(), smem)


@cl.kernel
def bad_clc_memspace_2():
    smem = cl.shared_array(1, cl.cluster_launch_control_token).get_base_pointer()
    with cl.local_array(1, cl.mbarrier) as larr:
        cl.cluster_launch_control_try_cancel(smem, larr.get_base_pointer())


@cl.kernel
def bad_clc_type():
    smem = cl.shared_array(2, cl.int64).get_base_pointer()
    cl.cluster_launch_control_is_canceled(smem)


@pytest.mark.parametrize(
    "kernel,match",
    (
        pytest.param(
            bad_clc_memspace_1,
            "Expected a pointer to a cluster launch control token in shared memory",
            id="bad clc token memory space",
        ),
        pytest.param(
            bad_clc_memspace_2,
            "Expected a pointer to an mbarrier in shared memory",
            id="bad mbarrier memory space",
        ),
        pytest.param(
            bad_clc_type,
            "Expected a cluster_launch_control_token",
            id="bad clc token type",
        ),
    ),
)
def test_invalid_usage(kernel, match):
    with pytest.raises(TypeCheckingError, match=match):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def compute():
    return 5.0


@cl.kernel
def worksteal(data, n: cl.Constant[int], stolen):
    clc_resp = cl.shared_array(1, cl.cluster_launch_control_token, alignment=16).get_base_pointer()
    mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

    tx = cl.thread_index(0)
    bdx = cl.thread_count(0)
    bx = cl.block_index(0)
    phase = 0

    if tx == 0:
        cl.mbarrier_initialize(mbar, 1)

    alpha = compute()
    while True:
        cl.barrier_sync_block()

        if tx == 0:
            cl.fence(
                cl.MemoryOrder.ACQUIRE,
                cl.MemoryScope.CLUSTER,
                from_proxy=cl.FenceProxy.GENERIC,
                to_proxy=cl.FenceProxy.ASYNC,
                restriction=cl.FenceRestriction.shared_cluster(),
            )
            cl.cluster_launch_control_try_cancel(clc_resp, mbar)
            cl.mbarrier_arrive_expect_transaction(
                mbar,
                clc_bytes,
                memory_order=cl.MemoryOrder.RELAXED,
            )

        i = bx * bdx + tx
        if i < n:
            data[i] *= alpha

        cl.mbarrier_wait_parity(mbar, phase)
        phase ^= 1

        tok = clc_resp.load()
        if not cl.cluster_launch_control_is_canceled(tok):
            break

        if tx == 0:
            cl.atomic_add(stolen.get_element_pointer(0), 1)
        bx = cl.cluster_launch_control_get_first_block_index(tok, axis=0)
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )


@cl.kernel
def worksteal_cluster(data, n: cl.Constant[int], stolen):
    clc_resp = cl.shared_array(1, cl.cluster_launch_control_token, alignment=16).get_base_pointer()
    mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

    tx = cl.thread_index(0)
    bdx = cl.thread_count(0)
    bx = cl.block_index(0)
    local_block = cl.block_in_cluster_index(0)
    phase = 0

    if tx == 0:
        cl.mbarrier_initialize(mbar, 1)
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=cl.FenceRestriction.mbarrier_initialize(),
        )

    alpha = compute()

    while True:
        cl._nvvm.barrier_cluster_arrive_aligned()
        cl._nvvm.barrier_cluster_wait_aligned()

        if local_block == 0 and tx == 0:
            cl.fence(
                cl.MemoryOrder.ACQUIRE,
                cl.MemoryScope.CLUSTER,
                from_proxy=cl.FenceProxy.GENERIC,
                to_proxy=cl.FenceProxy.ASYNC,
                restriction=cl.FenceRestriction.shared_cluster(),
            )
            cl.cluster_launch_control_try_cancel(clc_resp, mbar, multicast=True)

        if tx == 0:
            cl.mbarrier_arrive_expect_transaction(
                mbar,
                clc_bytes,
                scope=cl.MbarrierScope.CLUSTER,
                memory_order=cl.MemoryOrder.RELAXED,
            )

        i = bx * bdx + tx
        if i < n:
            data[i] *= alpha

        cl.mbarrier_wait_parity(mbar, phase, scope=cl.MbarrierScope.CLUSTER)
        phase ^= 1

        token = clc_resp.load()
        if not cl.cluster_launch_control_is_canceled(token):
            break  # no more work to steal

        if local_block == 0 and tx == 0:
            cl.atomic_add(stolen.get_element_pointer(0), 1)

        bx = cl.cluster_launch_control_get_first_block_index(token, axis=0) + local_block
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )


def launch_configs():
    for gscale, bscale in (
        (1, 1),
        (2, 1),
        (4, 2),
    ):
        grid = gscale * 8 * 1024
        block = bscale * 128
        yield pytest.param((grid,), (block,), id=f"{grid=},{block=}")


@pytest.mark.parametrize(
    "kernel,block_in_cluster_count",
    (
        pytest.param(worksteal, None, id="worksteal"),
        pytest.param(worksteal_cluster, (2, 1, 1), id="worksteal-cluster"),
    ),
)
@pytest.mark.parametrize("grid,block", launch_configs())
@pytest.mark.skipif(
    cc < (10, 0),
    reason="Cluster Launch Control requires compute capability 10.0 or newer",
)
def test_worksteal(kernel, block_in_cluster_count, grid, block):
    # Adapted from programming guide examples with added output tensor
    # to track number of stolen jobs.
    # https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#use-case-thread-blocks
    n = grid[0] * block[0]
    data = torch.arange(n, dtype=torch.float32).cuda()
    stolen = torch.zeros(1, dtype=torch.int64).cuda()
    expect = (data * compute()).cpu()
    cl.launch(
        torch.cuda.current_stream(),
        grid,
        block,
        kernel,
        (data, n, stolen),
        block_in_cluster_count=block_in_cluster_count,
    )
    torch.testing.assert_close(data.cpu(), expect)
    stolen = stolen.cpu().item()
    assert stolen > 0, (
        f"With {grid=} larger than possible active CTAs, expected at least one stolen job"
    )
