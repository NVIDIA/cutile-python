# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError, UnsupportedFeatureError
from cuda.lang.compilation import CallingConvention, KernelSignature, TensorMapConstraint
from cuda.tile._cext import cconv_v3_enabled

from .util import (
    compile_kernel,
    make_symbolic_scalar,
    make_symbolic_tensor,
    require_hopper_or_newer,
)


HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}
SM100_TARGET = {"gpu_name": "sm_100a", "arch": "compute_100a"}


COPY_ASYNC_ARRIVALS = {
    cl.copy_async_mbarrier_arrive: "cp.async.mbarrier.arrive",
    cl.copy_async_mbarrier_arrive_no_increment: "cp.async.mbarrier.arrive.noinc",
}


@pytest.mark.parametrize(("operation", "instruction"), COPY_ASYNC_ARRIVALS.items())
def test_copy_async_mbarrier_arrive_ptx(operation, instruction):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        operation(mbar)

    compile_kernel(
        kernel,
        assert_in_ptx=instruction,
        gpu_name="sm_80",
        arch="compute_80",
    )


@pytest.mark.parametrize("operation", COPY_ASYNC_ARRIVALS.keys())
def test_copy_async_mbarrier_arrive_unsupported_target(operation):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        operation(mbar)

    compile_kernel(
        kernel,
        raises=pytest.raises(UnsupportedFeatureError, match="copy_async_mbarrier_arrive"),
        gpu_name="sm_75",
        arch="compute_75",
    )


@pytest.mark.parametrize("arrival_operation", COPY_ASYNC_ARRIVALS.keys())
@pytest.mark.parametrize("arrivals", (1, 2))
@require_hopper_or_newer()
def test_copy_async_mbarrier_arrive_completion(arrival_operation, arrivals):
    participants = 32
    if arrival_operation is cl.copy_async_mbarrier_arrive_no_increment:
        participants += arrivals
    size = arrivals * 4

    @cl.kernel
    def kernel(src, out):
        tid = cl.thread_index(0)
        memory = cl.shared_array(size, cl.int32, alignment=16)
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        if tid == 0:
            cl.mbarrier_initialize(mbar, participants)
        cl.barrier_sync_block_aligned()

        if tid == 0:
            for i in range(arrivals):
                ptr = src.pointer(4 * i)
                ptr = cl.bitcast(ptr, cl.opaque_pointer_dtype('GLOBAL'))
                cl._nvvm.cp_async_ca_shared_global_16(memory.pointer(4 * i), ptr)
                arrival_operation(mbar)

        state = cl.mbarrier_arrive(mbar)
        cl.mbarrier_wait(mbar, state)
        out[tid] = memory[tid % size]

    src = torch.arange(1, size + 1, dtype=torch.int32).cuda(0)
    out = torch.zeros(32, dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (src, out))
    expected = src.cpu()[torch.arange(32) % size]
    assert torch.equal(out.cpu(), expected)


class CopyAsyncPtxTestBase:
    @staticmethod
    def signature():
        return KernelSignature(
            [
                TensorMapConstraint(),
                make_symbolic_scalar(cl.bool_),
                make_symbolic_scalar(cl.int32),
                make_symbolic_scalar(cl.int32),
                32,
                8,
            ],
            calling_convention=CallingConvention.cutile_python_v3(),
        )


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Tensor-map arguments require cconv3")
class TestG2S(CopyAsyncPtxTestBase):
    def test_l2_cache_hint(self):
        @cl.kernel
        def kernel(tensor_map, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
            cache_hint = cl.create_fractional_cache_policy(
                cl.CachePolicy.L2_EVICT_FIRST
            )

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.pointer(),
                mbar,
                l2_cache_hint=cache_hint,
            )

        compile_kernel(
            kernel,
            signature=KernelSignature([
                TensorMapConstraint(),
                make_symbolic_scalar(cl.int32),
                make_symbolic_scalar(cl.int32),
                32,
                8,
            ], calling_convention=CallingConvention.cutile_python_v3()),
            assert_in_ptx="cp.async.bulk.tensor.2d.shared::cta.global",
            **HOPPER_TARGET,
        )

    def test_shared_cluster_group_with_predicate_and_multicast(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                multicast_mask=0x3,
                cta_group=cl.CTAGroup.CTA_2,
                predicate=pred,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            assert_in_ptx=(
                "cp.async.bulk.tensor.2d.shared::cluster.global",
                "multicast::cluster",
            ),
            **SM100_TARGET,
        )

    def test_shared_cluster_mbarrier_address_space(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
            mbar = cl.map_shared_to_cluster(mbar, 0)

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                multicast_mask=0x3,
                cta_group=cl.CTAGroup.CTA_2,
            )

        match = (
            "Expected pointer memory space to be MemorySpace.SHARED "
            "but got MemorySpace.SHARED_CLUSTER"
        )
        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(TypeCheckingError, match=match),
        )

    def k1(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
        smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

        cl.copy_async_bulk_tensor_global_to_shared(
            tensor_map,
            (i, j),
            smem.pointer(),
            mbar,
            predicate=pred,
        )

    def k2(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
        smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

        cl.copy_async_bulk_tensor_global_to_shared(
            tensor_map,
            (i, j),
            smem.pointer(),
            mbar,
            multicast_mask=0xFF,
        )

    def k3(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
        smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

        cl.copy_async_bulk_tensor_global_to_shared(
            tensor_map,
            (i, j),
            smem.pointer(),
            mbar,
            cta_group=cl.CTAGroup.CTA_1,
        )

    @pytest.mark.parametrize("kernel", (k1, k2, k3))
    def test_unsupported_kwargs_for_cta_mode(self, kernel):
        match = (
            "When the destination memory is in shared memory, the "
            "predicate, multicast mask, and cta_group arguments are invalid."
        )
        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(
                TypeCheckingError,
                match=match,
            ),
        )

    @pytest.mark.parametrize("cluster", (True, False))
    def test_im2col_offsets_without_required_load_mode(self, cluster):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = smem.pointer()
            if cluster:
                smem = cl.map_shared_to_cluster(smem, 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map, (i, j), smem, mbar, im2col_offsets=(0, 1)
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(
                TypeCheckingError, match="TILE mode does not accept im2col_offsets"
            ),
        )

    @pytest.mark.parametrize(
        "mode",
        (cl.TMALoadMode.IM2COL, cl.TMALoadMode.IM2COL_W, cl.TMALoadMode.IM2COL_W_128),
    )
    def test_im2col_load_modes_require_offsets(self, mode):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.pointer(),
                mbar,
                mode=mode,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(
                TypeCheckingError,
                match=f"{mode.name} mode requires im2col_offsets",
            ),
        )

    def test_im2col_load_mode_rank3(self):
        @cl.kernel
        def kernel(
            tensor_map,
            pred,
            i,
            j,
            k,
            D: cl.Constant[int],
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j, k),
                smem.pointer(),
                mbar,
                im2col_offsets=(0,),
                mode=cl.TMALoadMode.IM2COL,
            )

        compile_kernel(
            kernel,
            signature=KernelSignature(
                [
                    TensorMapConstraint(),
                    make_symbolic_scalar(cl.bool_),
                    make_symbolic_scalar(cl.int32),
                    make_symbolic_scalar(cl.int32),
                    make_symbolic_scalar(cl.int32),
                    4,
                    32,
                    8,
                ],
                calling_convention=CallingConvention.cutile_python_v3(),
            ),
            assert_in_ptx="cp.async.bulk.tensor.3d.shared::cta.global.im2col",
            **HOPPER_TARGET,
        )

    def test_tile_gather4_rejects_im2col_offsets(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.pointer(),
                mbar,
                im2col_offsets=(0, 1),
                mode=cl.TMALoadMode.TILE_GATHER4,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(
                TypeCheckingError,
                match="TILE_GATHER4 mode does not accept im2col_offsets",
            ),
        )

    def test_invalid_tensor_map_pointer(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                smem.pointer(),
                (i, j),
                smem.pointer(),
                mbar,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(
                TypeCheckingError,
                match="Expected a tensor-map descriptor pointer",
            ),
        )


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Tensor-map arguments require cconv3")
class TestS2G(CopyAsyncPtxTestBase):
    def test_l2_cache_hint(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            cache_hint = cl.create_fractional_cache_policy(
                cl.CachePolicy.L2_EVICT_FIRST
            )

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.pointer(),
                tensor_map,
                (i, j),
                l2_cache_hint=cache_hint,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            assert_in_ptx="cp.async.bulk.tensor.2d.global.shared::cta",
            **HOPPER_TARGET,
        )

    def test_predicate(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.pointer(),
                tensor_map,
                (i, j),
                predicate=pred,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            assert_in_ptx="cp.async.bulk.tensor.2d.global.shared::cta",
            **HOPPER_TARGET,
        )

    def test_im2col_store_mode_rank2_is_rejected(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.pointer(),
                tensor_map,
                (i, j),
                mode=cl.TMAStoreMode.IM2COL,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            raises=pytest.raises(Exception, match="im2col|IM2COL|expected"),
            **HOPPER_TARGET,
        )

    def test_im2col_store_mode_rank3(self):
        @cl.kernel
        def kernel(
            tensor_map,
            pred,
            i,
            j,
            k,
            D: cl.Constant[int],
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.pointer(),
                tensor_map,
                (i, j, k),
                mode=cl.TMAStoreMode.IM2COL,
            )

        compile_kernel(
            kernel,
            signature=KernelSignature(
                [
                    TensorMapConstraint(),
                    make_symbolic_scalar(cl.bool_),
                    make_symbolic_scalar(cl.int32),
                    make_symbolic_scalar(cl.int32),
                    make_symbolic_scalar(cl.int32),
                    4,
                    32,
                    8,
                ],
                calling_convention=CallingConvention.cutile_python_v3(),
            ),
            assert_in_ptx="cp.async.bulk.tensor.3d.global.shared::cta.im2col",
            **HOPPER_TARGET,
        )


def test_copy_async_bulk_wait_group_read():
    def k():
        cl.copy_async_bulk_wait_group(0, read=False)
        cl.copy_async_bulk_wait_group(1, read=False)

    compile_kernel(
        k,
        filecheck_ptx="""
        CHECK-NOT: cp.async.bulk.wait_group.read
        CHECK: cp.async.bulk.wait_group 0
        CHECK-NEXT: cp.async.bulk.wait_group 1
        """,
        **HOPPER_TARGET,
    )


def test_copy_async_bulk_wait_group():
    def k():
        cl.copy_async_bulk_wait_group(0, read=True)
        cl.copy_async_bulk_wait_group(1, read=True)

    compile_kernel(
        k,
        filecheck_ptx="""
        CHECK: cp.async.bulk.wait_group.read 0
        CHECK-NEXT: cp.async.bulk.wait_group.read 1
        """,
        **HOPPER_TARGET,
    )


def test_copy_async_bulk_commit_group():
    def k():
        cl.copy_async_bulk_commit_group()

    compile_kernel(
        k,
        assert_in_ptx="cp.async.bulk.commit_group",
        **HOPPER_TARGET,
    )


def test_copy_async_bulk_wait_group_non_immediate_group():
    def k(input):
        cl.copy_async_bulk_wait_group(input[0])

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_tensor(1, dtype=cl.int32)]),
        raises=pytest.raises(Exception, match="Expected an integer constant"),
    )


def test_copy_async_bulk_wait_group_non_immediate_read():
    def k(input):
        cl.copy_async_bulk_wait_group(0, read=input[0] > 0)

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_tensor(1, dtype=cl.int32)]),
        raises=pytest.raises(Exception, match="Expected a boolean constant"),
    )
