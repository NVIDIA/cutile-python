# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._exception import TileCompilerExecutionError
import torch
import pytest

from cuda.lang.compilation import KernelSignature

from .util import (
    compile_for_arguments,
    filecheck,
    make_symbolic_tensor,
    require_blackwell_or_newer,
    require_hopper_or_newer,
)


def test_arch_specific_kernel_failure():
    @cl.kernel(arch="compute_80", gpu_name="sm_80")
    def kernel():
        cl.elect_sync()

    with pytest.raises(
        TileCompilerExecutionError,
        match="Cannot select: intrinsic %llvm.nvvm.elect.sync",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_coop_launch():
    @cl.kernel
    def kernel():
        pass

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (),
        cooperative=True,
    )


@require_hopper_or_newer()
def test_cluster_dim_launch_updates_cluster_registers():
    @cl.kernel
    def kernel(out):
        if cl.thread_index(0) != 0:
            return

        bx, by = cl.block_index(0), cl.block_index(1)
        slot = bx + by * cl.block_count(0)

        out[slot, 0] = cl.cluster_index(0)
        out[slot, 1] = cl.cluster_index(1)
        out[slot, 2] = cl.cluster_index(2)
        out[slot, 3] = cl.cluster_count(0)
        out[slot, 4] = cl.cluster_count(1)
        out[slot, 5] = cl.cluster_count(2)
        out[slot, 6] = cl.block_in_cluster_index(0)
        out[slot, 7] = cl.block_in_cluster_index(1)
        out[slot, 8] = cl.block_in_cluster_index(2)
        out[slot, 9] = cl.block_in_cluster_count(0)
        out[slot, 10] = cl.block_in_cluster_count(1)
        out[slot, 11] = cl.block_in_cluster_count(2)

    out = torch.zeros(8, 12, dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (4, 2),
        (1,),
        kernel,
        (out,),
        block_in_cluster_count=(2, 1, 1),
    )

    expected = [
        [0, 0, 0, 2, 2, 1, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1],
        [1, 0, 0, 2, 2, 1, 0, 0, 0, 2, 1, 1],
        [1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1],
        [0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 1],
        [0, 1, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1],
        [1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 1, 1],
        [1, 1, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1],
    ]
    assert out.cpu().tolist() == expected


@require_hopper_or_newer()
def test_cluster_dim_launch_requires_grid_multiple():
    @cl.kernel
    def kernel():
        pass

    with pytest.raises(RuntimeError, match="Failed to launch cuTile kernel"):
        cl.launch(
            torch.cuda.current_stream(),
            (3,),
            (1,),
            kernel,
            (),
            block_in_cluster_count=(2, 1, 1),
        )


@require_blackwell_or_newer()
def test_preferred_cluster_dim_launch_requires_multiple_of_cluster_dim():
    @cl.kernel
    def kernel():
        pass

    with pytest.raises(RuntimeError, match="Failed to launch cuTile kernel"):
        cl.launch(
            torch.cuda.current_stream(),
            (6,),
            (1,),
            kernel,
            (),
            block_in_cluster_count=(2, 1, 1),
            preferred_block_in_cluster_count=(3, 1, 1),
        )


@require_hopper_or_newer()
def test_cluster_dims():
    @cl.kernel
    def kernel():
        pass

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (),
        block_in_cluster_count=(1, 1, 1),
    )


@require_blackwell_or_newer()
def test_preferred_cluster_dims():
    @cl.kernel
    def kernel():
        pass

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (),
        block_in_cluster_count=(1, 1, 1),
        preferred_block_in_cluster_count=(1, 1, 1),
    )


def test_invalid_cluster_config():
    @cl.kernel
    def kernel():
        pass

    with pytest.raises(
        ValueError,
        match="Keyword argument preferred_block_in_cluster_count"
        " requires that block_in_cluster_count is also passed",
    ):
        cl.launch(
            torch.cuda.current_stream(),
            (1,),
            (1,),
            kernel,
            (),
            # preferred cluster config without "regular" cluster config
            # block_in_cluster_count=(1, 1, 1),
            preferred_block_in_cluster_count=(1, 1, 1),
        )


@require_hopper_or_newer()
def test_setmaxregister():

    @cl.kernel
    def kernel():
        cl.setmaxregister_increase(64)
        cl.setmaxregister_decrease(32)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_tid():
    @cl.kernel
    def kernel(A):
        tidx, tidy, tidz = cl.thread_index(0), cl.thread_index(1), cl.thread_index(2)
        A[tidx, tidy, tidz] = tidx + tidy + tidz

    A = torch.zeros(3, 3, 3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (3, 3, 3), kernel, (A,))
    expected = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
        ],
        device="cpu",
        dtype=torch.int32,
    )
    assert (expected == A.cpu()).all()


def test_bid():
    @cl.kernel
    def kernel(A):
        bidx, bidy, bidz = cl.block_index(0), cl.block_index(1), cl.block_index(2)
        A[bidx, bidy, bidz] = bidx + bidy + bidz

    A = torch.zeros(3, 3, 3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (3, 3, 3), (1,), kernel, (A,))
    expected = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
        ],
        device="cpu",
        dtype=torch.int32,
    )
    assert (expected == A.cpu()).all()


def test_thread_count():
    @cl.kernel
    def kernel(out):
        tidx = cl.thread_index(0)
        if tidx == 0:
            out[0], out[1], out[2] = (
                cl.thread_count(0),
                cl.thread_count(1),
                cl.thread_count(2),
            )

    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (4, 3, 2), kernel, (out,))
    assert (out.cpu() == torch.tensor([4, 3, 2], dtype=torch.int32)).all()


def test_grid_dim():
    @cl.kernel
    def kernel(out):
        tidx = cl.thread_index(0)
        if tidx == 0:
            out[0], out[1], out[2] = (
                cl.block_count(0),
                cl.block_count(1),
                cl.block_count(2),
            )

    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (5, 6, 7), (1,), kernel, (out,))
    assert (out.cpu() == torch.tensor([5, 6, 7], dtype=torch.int32)).all()


@require_hopper_or_newer()
def test_elect_sync(capsys):
    @cl.kernel()
    def kernel(out):
        tx, ty, tz = cl.thread_index(0), cl.thread_index(1), cl.thread_index(2)
        if cl.elect_sync():
            out[tx, ty, tz] = 1

    out = torch.zeros(3, 3, 3, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (3, 3, 3), kernel, (out,))
    assert sum(out.cpu().ravel().tolist()) == 1


def test_lane_count_full_mask_and_ptx_comment(log_ptx):
    ptx_comment = "FOOBARBAZ"

    @cl.kernel
    def kernel(out):
        tidx = cl.thread_index(0)
        cl.ptx_comment(ptx_comment)
        value = cl.shfl_sync(tidx, 7)
        if tidx == 0:
            out[0] = cl.lane_count()
            out[1] = value

    compiled = compile_for_arguments(kernel, [make_symbolic_tensor((2,), cl.int32)])
    assert compiled.ptx is not None
    assert ptx_comment in compiled.ptx

    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
    assert (out.cpu() == torch.tensor([32, 7], dtype=torch.int32)).all()


def test_full_mask_is_unsigned():
    @cl.kernel
    def kernel(out):
        out[0] = cl.uint64(cl.full_mask())
        out[1] = cl.uint64(cl.int32(0xFFFFFFFF))

    out = torch.zeros(2, dtype=torch.uint64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    expect = [
        0xFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ]
    assert out.cpu().tolist() == expect


def test_full_mask_i32_shift():
    @cl.kernel
    def kernel(out):
        out[0] = cl.full_mask() >> cl.int64(32)

    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().item() == (cl.full_mask() >> 32)


def test_lane_index():
    @cl.kernel
    def kernel(out):
        tidx = cl.thread_index(0)
        out[tidx] = cl.lane_index()

    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
    expected = torch.tensor(list(range(32)) * 2, dtype=torch.int32)
    assert (out.cpu() == expected).all()


def test_warp_index():
    @cl.kernel
    def kernel(out):
        tidx = cl.thread_index(0)
        out[tidx] = cl.warp_index()

    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
    expected = torch.tensor([0] * 32 + [1] * 32, dtype=torch.int32)
    assert (out.cpu() == expected).all()


def test_saxpy():
    """
    Taken from https://developer.nvidia.com/blog/six-ways-saxpy/

    __global__
    void saxpy(int n, float a, float * restrict x, float * restrict y)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) y[i] = a*x[i] + y[i];
    }
    """

    @cl.kernel
    def kernel(N: cl.Constant[int], a: cl.Constant[float], X, Y):
        tidx = cl.thread_index(0)
        bidx = cl.block_index(0)
        thread_count_x = cl.thread_count(0)
        idx = tidx + bidx * thread_count_x
        if idx < N:
            Y[idx] = a * X[idx] + Y[idx]

    N = 256
    alpha = 2.0
    X = torch.ones(N, dtype=torch.float32, device="cuda")
    Y = torch.ones(N, dtype=torch.float32, device="cuda")
    expected = (alpha * X + Y).cpu()
    cl.launch(torch.cuda.current_stream(), (64,), (64,), kernel, (N, alpha, X, Y))
    assert torch.allclose(expected, Y.cpu())


class TestSyncwarp:
    """
    _CUDA C++ Programming Guide 10.6 Synchronization Functions_ describes
    these functions and _20.6.2. Independent Thread Scheduling_ has example programs
    using them. These tests are based on those examples.
    """

    def test_syncwarp(self):
        """
        __syncwarp(mask) waits until all named lanes execute the same warp barrier.
        """

        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            lane = cl.thread_index(0)

            shmem[lane] = lane
            cl.barrier_sync_warp()
            out[lane] = shmem[lane ^ 1]

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [lane ^ 1 for lane in range(32)],
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_syncwarp_reduction(self):
        """
        warp lanes can communicate via memory by storing, synchronizing with barrier_sync_warp
        and then reading peers' values.
        """

        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            lane = cl.thread_index(0)

            shmem[lane] = 1
            cl.barrier_sync_warp()

            if lane < 16:
                shmem[lane] = shmem[lane] + shmem[lane + 16]
            cl.barrier_sync_warp()
            if lane < 8:
                shmem[lane] = shmem[lane] + shmem[lane + 8]
            cl.barrier_sync_warp()
            if lane < 4:
                shmem[lane] = shmem[lane] + shmem[lane + 4]
            cl.barrier_sync_warp()
            if lane < 2:
                shmem[lane] = shmem[lane] + shmem[lane + 2]
            cl.barrier_sync_warp()
            if lane < 1:
                shmem[lane] = shmem[lane] + shmem[lane + 1]
            cl.barrier_sync_warp()

            if lane == 0:
                out[0] = shmem[0]

        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        assert (out.cpu() == torch.tensor([32], dtype=torch.int32)).all()

    def test_syncwarp_half_mask(self):
        """
        each calling thread must have its own bit set in the mask, and only lanes
        named in the mask participate in the warp barrier.
        """

        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(16,), dtype=cl.int32)
            lane = cl.thread_index(0)
            mask = 0x0000FFFF

            if lane < 16:
                shmem[lane] = lane + 100
                cl.barrier_sync_warp(mask)
                out[lane] = shmem[lane ^ 1]
            else:
                out[lane] = -1

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [
                101,
                100,
                103,
                102,
                105,
                104,
                107,
                106,
                109,
                108,
                111,
                110,
                113,
                112,
                115,
                114,
            ]
            + [-1] * 16,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()


class TestShuffle:
    """
    Shuffle tests based on the CUDA sample
    Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu and direct
    single-warp sanity checks.
    """

    def test_shfl_up_scan_width_8(self):
        @cl.kernel
        def kernel(inp, out):
            tid = cl.thread_index(0)
            lane = tid % 32
            sublane = lane % 8
            value = cl.int32(inp[tid])
            width = cl.int32(8)
            mask = cl.int32(0xFFFFFFFF)

            delta = cl.int32(1)
            for _ in range(4):
                other = cl.shfl_up_sync(value, delta, width, mask=mask)
                if sublane >= delta:
                    value += other
                delta *= 2

            out[tid] = value

        inp = torch.ones(32, dtype=torch.int32, device="cuda")
        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (inp, out))
        expected = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8] * 4,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_shfl_sync_idx(self):
        @cl.kernel
        def kernel(out):
            lane = cl.thread_index(0)
            out[lane] = cl.shfl_sync(lane, 4)

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.full((32,), 4, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_shfl_down_sync(self):
        @cl.kernel
        def kernel(out):
            lane = cl.thread_index(0)
            out[lane] = cl.shfl_down_sync(lane, 4)

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.arange(32, dtype=torch.int32)
        expected[:-4] += 4
        assert (out.cpu() == expected).all()

    def test_shfl_xor_sync(self):
        @cl.kernel
        def kernel(out):
            lane = cl.thread_index(0)
            out[lane] = cl.shfl_xor_sync(lane, 16, mask=cl.int32(0xFFFFFFFF))

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor([lane ^ 16 for lane in range(32)], dtype=torch.int32)
        assert (out.cpu() == expected).all()

    @pytest.mark.parametrize(
        "op", (cl.shfl_down_sync, cl.shfl_xor_sync, cl.shfl_sync, cl.shfl_up_sync)
    )
    def test_shfl_primitive_mask_persists(self, op):
        @cl.kernel
        def kernel(tensor):
            i1 = op(1, 1, 1, mask=1234)
            i2 = op(1, 1, 1, mask=4321)
            i2 = op(1, 1, 1)  # omitted, ensure it's FULL_MASK
            tensor[0] = i1 + i2

        cres = cl.compile_simt(
            kernel, [KernelSignature([make_symbolic_tensor(1, cl.int32)])]
        )

        filecheck(
            cres.mlir,
            """
            CHECK: 1234
            CHECK: 4321
            CHECK: -1
            """,
        )


class TestBarrierSync:
    """
    Barrier tests based on the CUDA guide's block synchronization semantics and
    PTX named barrier examples.
    """

    def test_barrier_sync_shared_exchange(self):
        @cl.kernel
        def kernel(out):
            lane = cl.thread_index(0)
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)

            shmem[lane] = lane * 2
            cl._nvvm.barrier_cta_sync_all(cl.int32(0))

            if lane < 16:
                out[lane] = shmem[lane + 16]
            else:
                out[lane] = shmem[lane - 16]

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [2 * (lane + 16) for lane in range(16)]
            + [2 * (lane - 16) for lane in range(16, 32)],
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_barrier_sync_count_warp_subset(self):
        @cl.kernel
        def kernel(out):
            lane = cl.thread_index(0)
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)

            if lane < 32:
                shmem[lane] = lane + 100
                cl._nvvm.barrier_cta_sync_count(1, 32)
                out[lane] = shmem[lane ^ 1]
            else:
                out[lane] = -1

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor(
            [
                101,
                100,
                103,
                102,
                105,
                104,
                107,
                106,
                109,
                108,
                111,
                110,
                113,
                112,
                115,
                114,
                117,
                116,
                119,
                118,
                121,
                120,
                123,
                122,
                125,
                124,
                127,
                126,
                129,
                128,
                131,
                130,
            ]
            + [-1] * 32,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()


class TestVoteSync:
    """
    Vote tests based on simpleVoteIntrinsics and the CUDA guide's warp vote
    semantics.
    """

    def test_all_sync(self):
        @cl.kernel
        def kernel(out):
            tid = cl.thread_index(0)
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = True
            else:
                pred = lane < 16

            if cl._nvvm.vote_all_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor([1] * 32 + [0] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_any_sync(self):
        @cl.kernel
        def kernel(out):
            tid = cl.thread_index(0)
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = False
            else:
                pred = lane == 0

            if cl._nvvm.vote_any_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor([0] * 32 + [1] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_uni_sync(self):
        @cl.kernel
        def kernel(out):
            tid = cl.thread_index(0)
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = True
            elif tid < 64:
                pred = False
            else:
                pred = lane < 16

            if cl._nvvm.vote_uni_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(96, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (96,), kernel, (out,))
        expected = torch.tensor([1] * 64 + [0] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_ballot_sync(self):
        @cl.kernel
        def kernel(out):
            tid = cl.thread_index(0)
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = lane < 8
            else:
                pred = (lane % 2) == 0

            out[tid] = cl._nvvm.vote_ballot_sync(mask, pred)

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor(
            [0x000000FF] * 32 + [0x55555555] * 32,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()
