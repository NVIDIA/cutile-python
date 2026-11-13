# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from test.util import filecheck
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
    require_rubin_family,
    require_Ampere_or_newer,
)

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}
SM100_TARGET = {"gpu_name": "sm_100a", "arch": "compute_100a"}
RUBIN_TARGET = {"gpu_name": "sm_107f", "arch": "compute_107f"}


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


@require_rubin_family()
@pytest.mark.skipif(not cconv_v3_enabled(), reason="Tensor-map arguments require cconv3")
@pytest.mark.parametrize(
    ("report", "dtype", "pattern"),
    (
        (cl.AsyncCopyReportMechanism.DISABLED, torch.uint8, 0xff),
        (cl.AsyncCopyReportMechanism.VALIDITY_PER_16_BYTES_80000000, torch.int32, -(1 << 31)),
        (cl.AsyncCopyReportMechanism.VALIDITY_PER_16_BYTES_8000, torch.uint16, 0x8000),
        (cl.AsyncCopyReportMechanism.VALIDITY_PER_16_BYTES_80, torch.uint8, 0x80),
        (cl.AsyncCopyReportMechanism.VALIDITY_PER_16_BYTES_8, torch.uint8, 0x88),
        (cl.AsyncCopyReportMechanism.VALIDITY_PER_ELEMENT_FF, torch.uint8, 0xff),
    ),
)
@pytest.mark.parametrize("cluster", (False, True))
@pytest.mark.parametrize("match_first", (False, True))
def test_copy_report_completion(report, dtype, pattern, cluster, match_first):
    rows, columns = 16, 32
    transaction_bytes = rows * columns * torch.empty((), dtype=dtype).element_size()

    @cl.kernel
    def kernel(desc: cl.TensorMap, out, reports):
        tid = cl.thread_index(0)
        dst = cl.shared_array(rows * columns, out.dtype, alignment=512)
        mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        if tid == 0:
            cl.mbarrier_initialize(mbar, 1, layout=cl.MbarrierLayout.V1)
            cl.fence(
                "RELEASE",
                "CLUSTER",
                restriction=cl.FenceRestriction.mbarrier_initialize(),
            )
        cl.barrier_sync_block_aligned()
        copy_dst = dst.pointer()
        if cl.ensure_constant(cluster):
            copy_dst = cl.map_shared_to_cluster(copy_dst, 0)
        for phase in range(2):
            if tid == 0:
                cl.mbarrier_arrive_expect_transaction(mbar, transaction_bytes)
                cl.copy_async_bulk_tensor_global_to_shared(
                    desc, (0, phase * rows), copy_dst, mbar, report_mechanism=report
                )
            report_predicate, report_value = cl.mbarrier_wait_report_parity(mbar, phase)
            if tid == 0:
                reports[phase, 0] = report_predicate
                reports[phase, 1] = report_value
            for row in range(rows):
                out[phase * rows + row, tid] = dst[row * columns + tid]
            cl.barrier_sync_block_aligned()
        if tid == 0:
            cl.mbarrier_invalidate(mbar)

    src = torch.zeros((2 * rows, columns), dtype=dtype).cuda(0)
    if match_first:
        src[:rows] = pattern
    else:
        src[rows:] = pattern
    out = torch.empty_like(src)
    reports = torch.full((2, 2), -1, dtype=torch.int32).cuda(0)
    desc = cl.tensor_map_tiled(src, (columns, rows), order=(1, 0))
    cl.launch(torch.cuda.current_stream(), (1,), (columns,), kernel, (desc, out, reports))
    assert torch.equal(out.cpu(), src.cpu())
    enabled = report is not cl.AsyncCopyReportMechanism.DISABLED
    assert reports.cpu().tolist() == [
        [int(enabled and match_first), 0],
        [int(enabled and not match_first), 0],
    ]


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

    @staticmethod
    def make_report_kernel(report, cluster: bool):
        @cl.kernel
        def k(
            desc: cl.TensorMap,
            _,
            i32_0,
            i32_1,
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            dst = cl.shared_array(H * W, cl.int32, alignment=512).pointer()
            mbar = cl.shared_array(1, cl.mbarrier, alignment=16).pointer()
            if cluster:
                dst = cl.map_shared_to_cluster(dst, 0)
            cl.copy_async_bulk_tensor_global_to_shared(
                desc, (i32_0, i32_1), dst, mbar, report_mechanism=report
            )
        return k

    @pytest.mark.parametrize("target", ("90", "100a", "103f", "107", "110f", "120a"))
    def test_copy_report_target(self, target):
        compile_kernel(
            TestG2S.make_report_kernel('VALIDITY_PER_ELEMENT_FF', False),
            signature=CopyAsyncPtxTestBase.signature(),
            gpu_name=f"sm_{target}", arch=f"compute_{target}",
            raises=pytest.raises(UnsupportedFeatureError, match="reporting requires"),
        )

    @pytest.mark.parametrize("target", ("90", "100a", "103f", "107"))
    @pytest.mark.parametrize("cluster", (False, True))
    def test_copy_report_disabled_on_older_targets(self, target, cluster):
        compile_kernel(
            TestG2S.make_report_kernel(cl.AsyncCopyReportMechanism.DISABLED, cluster),
            signature=self.signature(),
            gpu_name=f"sm_{target}",
            arch=f"compute_{target}",
            assert_in_ptx="cp.async.bulk.tensor.2d",
            assert_not_in_ptx="mbarrier::report",
        )

    @pytest.mark.parametrize("cluster", (False, True))
    def test_copy_report_with_options(self, cluster):
        @cl.kernel
        def k(
            desc: cl.TensorMap,
            _,
            i32_0,
            i32_1,
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            dst = cl.shared_array(H * W, cl.int32, alignment=512).pointer()
            mbar = cl.shared_array(1, cl.mbarrier, alignment=16).pointer()
            hint = cl.create_fractional_cache_policy(cl.CachePolicy.L2_EVICT_FIRST)
            if cl.ensure_constant(cluster):
                dst = cl.map_shared_to_cluster(dst, 0)
                cl.copy_async_bulk_tensor_global_to_shared(
                    desc, (i32_0, i32_1), dst, mbar,
                    l2_cache_hint=hint,
                    multicast_mask=0x3,
                    cta_group='CTA_2',
                    report_mechanism='VALIDITY_PER_ELEMENT_FF',
                )
            else:
                cl.copy_async_bulk_tensor_global_to_shared(
                    desc, (i32_0, i32_1), dst, mbar,
                    l2_cache_hint=hint,
                    report_mechanism='VALIDITY_PER_ELEMENT_FF',
                )

        compiled = compile_kernel(
            k,
            signature=self.signature(),
            **RUBIN_TARGET,
        )
        assert compiled and compiled.ptx
        filecheck(
            compiled.ptx,
            r"""
            CHECK-NOT: cp.async.bulk.tensor
            CHECK: {{^[ \t]*}}cp.async.bulk.tensor.2d
            LOCAL-SAME: .shared::cta.global
            CLUSTER-SAME: .shared::cluster.global
            CLUSTER-SAME: .multicast::cluster
            CHECK-SAME: .L2::cache_hint
            CHECK-SAME: .mbarrier::report::validity::per_element::ff
            CLUSTER-SAME: .cta_group::2
            CHECK-NOT: cp.async.bulk.tensor
            """,
            check_prefixes=('CHECK', 'CLUSTER' if cluster else 'LOCAL'),
        )

    @pytest.mark.parametrize('report', tuple(cl.AsyncCopyReportMechanism))
    @pytest.mark.parametrize('cluster', (True, False))
    @pytest.mark.parametrize('target', ('107f', '107a'))
    def test_report_mechanism(self, report, cluster, target):
        k = TestG2S.make_report_kernel(report, cluster)
        if report is cl.AsyncCopyReportMechanism.DISABLED:
            ptx = None
            not_ptx = '.mbarrier::report::'
        else:
            ptx = ".mbarrier::report::" + report.name.lower().replace(
                "validity_", "validity::"
            ).replace("per_16_bytes_", "per_16bytes::").replace(
                "per_element_", "per_element::"
            )
            not_ptx = None
        compile_kernel(
            k,
            signature=self.signature(),
            gpu_name=f"sm_{target}",
            arch=f"compute_{target}",
            assert_in_ptx=ptx,
            assert_not_in_ptx=not_ptx,
        )

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

    def test_shared_cluster_group_with_multicast(self):
        @cl.kernel
        def kernel(tensor_map, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(H * W, cl.int32, alignment=512).pointer()
            smem = cl.map_shared_to_cluster(smem, 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map, (i, j), smem, mbar,
                multicast_mask=0x3,
                cta_group=cl.CTAGroup.CTA_2,
            )

        compile_kernel(
            kernel,
            signature=self.signature(),
            assert_in_ptx=(
                "cp.async.bulk.tensor.2d.shared::cluster.global",
                "multicast::cluster",
                "cta_group::2",
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

    @pytest.mark.parametrize("kernel", (k2, k3))
    def test_unsupported_kwargs_for_cta_mode(self, kernel):
        match = (
            "When the destination memory is in shared memory, the "
            "multicast mask and cta_group arguments are invalid."
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


@require_Ampere_or_newer()
def test_copy_async_global_to_shared():
    @cl.kernel
    def kernel(x, out):
        tid = cl.thread_index(0)
        smem = cl.shared_array(256, cl.int32, alignment=16)
        index = tid * 4
        dst = smem.pointer(index)
        src = x.pointer(index)

        cl.copy_async_global_to_shared(src, dst, dst_size=16, cache_hint=cl.CacheHint.CA)
        cl.copy_async_commit_group()
        cl.copy_async_wait_groups(0)
        cl.syncthreads()

        index_from_end = 255 - index
        for i in range(4):
            out[index + i] = smem[index_from_end - i]

    x = torch.arange(256, dtype=torch.int32, device="cuda:0")
    out = torch.empty_like(x)
    cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (x, out))
    assert out.tolist() == x.tolist()[::-1]


@require_Ampere_or_newer()
def test_copy_async_global_to_shared_wait():
    @cl.kernel
    def kernel(x, out):
        tid = cl.thread_index(0)
        smem = cl.shared_array(8, cl.int32, alignment=4)
        index = tid * 2
        dst = smem.pointer(index)
        src = x.pointer(index)

        cl.copy_async_global_to_shared(src, dst, dst_size=4)
        cl.copy_async_commit_group()

        cl.copy_async_global_to_shared(src + 1, dst + 1, dst_size=4)
        cl.copy_async_commit_group()

        cl.copy_async_wait_groups(1)

        smem[index] = 99

        cl.copy_async_wait_groups(0)
        cl.syncthreads()

        index_from_end = 7 - index
        out[index] = smem[index_from_end]
        out[index + 1] = smem[index_from_end - 1]

    x = torch.arange(8, dtype=torch.int32, device="cuda:0")
    out = torch.empty_like(x)
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (x, out))
    assert out.tolist() == [7, 99, 5, 99, 3, 99, 1, 99]


@require_Ampere_or_newer()
@pytest.mark.parametrize(
    "bytes_to_copy,result",
    [(None, [7, 6, 5, 4, 3, 2, 1, 0]), (0, [0, 0, 0, 0, 0, 0, 0, 0]),
     (4, [0, 6, 0, 4, 0, 2, 0, 0]), (8, [7, 6, 5, 4, 3, 2, 1, 0])]
)
def test_copy_async_global_to_shared_src_size_constant(bytes_to_copy, result):
    @cl.kernel
    def kernel(x, out, bytes_to_copy: cl.Constant[int | None]):
        tid = cl.thread_index(0)
        smem = cl.shared_array(8, cl.int32, alignment=8)
        index = tid * 2
        dst = smem.pointer(index)
        src = x.pointer(index)

        cl.copy_async_global_to_shared(src, dst, dst_size=8, src_size=bytes_to_copy)
        cl.copy_async_commit_group()
        cl.copy_async_wait_groups(0)
        cl.syncthreads()

        index_from_end = 7 - index
        out[index] = smem[index_from_end]
        out[index + 1] = smem[index_from_end - 1]

    x = torch.arange(8, dtype=torch.int32, device="cuda:0")
    out = torch.empty_like(x)
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (x, out, bytes_to_copy))
    assert out.tolist() == result


@require_Ampere_or_newer()
@pytest.mark.parametrize(
    "bytes_to_copy,result",
    [(0, [0, 0, 0, 0, 0, 0, 0, 0]), (4, [0, 6, 0, 4, 0, 2, 0, 0]), (8, [7, 6, 5, 4, 3, 2, 1, 0])]
)
def test_copy_async_global_to_shared_src_size_runtime(bytes_to_copy, result):
    @cl.kernel
    def kernel(x, out, bytes_to_copy):
        tid = cl.thread_index(0)
        smem = cl.shared_array(8, cl.int32, alignment=8)
        index = tid * 2
        dst = smem.pointer(index)
        src = x.pointer(index)

        cl.copy_async_global_to_shared(src, dst, dst_size=8, src_size=bytes_to_copy)
        cl.copy_async_commit_group()
        cl.copy_async_wait_groups(0)
        cl.syncthreads()

        index_from_end = 7 - index
        out[index] = smem[index_from_end]
        out[index + 1] = smem[index_from_end - 1]

    x = torch.arange(8, dtype=torch.int32, device="cuda:0")
    out = torch.empty_like(x)
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (x, out, bytes_to_copy))
    assert out.tolist() == result


@require_Ampere_or_newer()
def test_copy_async_global_to_shared_reject_src_mem():
    @cl.kernel
    def kernel(x):
        tid = cl.thread_index(0)
        smem = cl.shared_array(8, cl.int32, alignment=8)
        dst = smem.pointer(tid)
        src = x.pointer(tid)

        cl.copy_async_global_to_shared(dst, src, dst_size=8)

    x = torch.arange(8, dtype=torch.int32, device="cuda:0")
    with pytest.raises(TypeCheckingError, match="Expected pointer memory space to be "
                                                "MemorySpace.SHARED but got MemorySpace.GENERIC"):
        cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (x,))


@require_Ampere_or_newer()
def test_copy_async_global_to_shared_reject_bytes():
    @cl.kernel
    def kernel(x):
        tid = cl.thread_index(0)
        smem = cl.shared_array(8, cl.int32, alignment=8)
        dst = smem.pointer(tid)
        src = x.pointer(tid)

        cl.copy_async_global_to_shared(src, dst, dst_size=8, cache_hint=cl.CacheHint.CG)

    x = torch.arange(8, dtype=torch.int32, device="cuda:0")
    with pytest.raises(TypeCheckingError, match="dst_size must be 16 for CG, got 8"):
        cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (x,))
