# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Lang port of the CuTe DSL ``fp16_gemm_3.py`` tutorial.

Computes ``C = A @ B.T`` with an optional FP16 row bias. Two CTAs collaborate
on each 256x256 output tile. GEMM3 adds warp specialization, persistent tile
scheduling, a six-stage A/B SMEM pipeline, and a two-stage TMEM accumulator
pipeline: epilogue warps 0-3 store C, warp 4 issues MMA, and warp 5 issues TMA.

Additional CuTe DSL -> CUDA Lang notes beyond ``fp16_gemm_2.py``:

* CuTe DSL uses ``StaticPersistentTileScheduler``. This sample shows that a
  similar abstraction can be built from CUDA Lang's ordinary Python
  primitives: a frozen dataclass holding the configuration plus methods that
  CUDA Lang inlines into the kernel. The instance uses the source's no-swizzle,
  M-raster, one-batch, (2, 1, 1)-cluster configuration. CUDA Lang does not
  support dataclass ``__post_init__``, so configuration validation is an
  explicit host-side method call.
* The source queries ``utils.HardwareInfo().get_max_active_clusters`` using a
  compiled dummy kernel and the CUDA occupancy API. PyTorch does not expose
  that occupancy query, so at launch time this sample queries the current
  device's SM count and divides by the number of CTAs per cluster. This gives
  the same value for the source's two-CTA cluster on B200, without adding a
  dummy compile step.
The implementation otherwise retains the source's warp roles, cluster shape,
pipeline stages, data movement, barrier phases, CTA_2 MMA sequence, staged TMEM
epilogue, peer deallocation synchronization, and vector stores.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch


WARP_SIZE = 32
BLOCK_THREADS = 6 * WARP_SIZE
CLUSTER_M = 2
CTA_M = 128
CTA_N = 128
TILE_M = CLUSTER_M * CTA_M
TILE_N = CLUSTER_M * CTA_N
BLOCK_K = 64
MMA_K = 16
VEC_BYTES = 32
AB_STAGES = 6
ACC_STAGES = 2
EPILOGUE_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
ALLOCATOR_WARP = 0
TMEM_COLS = 512
TMEM_BARRIER_ID = 2
TMEM_BARRIER_THREADS = (EPILOGUE_WARPS + 1) * WARP_SIZE
DEALLOC_BARRIER_ID = 3
A_STAGE_ELEMS = CTA_M * BLOCK_K
B_STAGE_ELEMS = CTA_N * BLOCK_K
TMA_COPY_BYTES = (A_STAGE_ELEMS + B_STAGE_ELEMS) * 2 * CLUSTER_M

_DEFAULT_MNK = (256, 256, 256)
_DEFAULT_TOLERANCE = 1.0e-1


def _as_float32_vector(regs):
    """Interpret 32 TMEM load registers as one FP32 vector."""
    return cl.Vector(
        *tuple(
            cl.bitcast(regs[i], cl.float32)
            for i in cl.static_iter(range(32))
        ),
        dtype=cl.float32,
    )


def _to_float16_vector(values, base, vsize):
    """Convert one FP32 vector slice to FP16."""
    return cl.Vector(
        *tuple(
            values[base + i]
            for i in cl.static_iter(range(vsize))
        ),
        dtype=cl.float32,
    ).astype(cl.float16)


@dataclass(frozen=True)
class _StaticPersistentTileScheduler:
    """Local scheduler abstraction built from CUDA Lang Python primitives.

    The configuration is immutable and therefore compile-time visible when
    the methods are called from a kernel. CUDA Lang inlines the ordinary
    Python methods while their M/N/work-index operands remain device values.

    This tutorial implements both raster directions without swizzling. A
    swizzled scheduler can use the same interface, but its padded layout and
    index mapping are outside what the source tutorial exercises.
    """

    cta_tile_shape_mn: tuple[int, int]
    cluster_shape_mnk: tuple[int, int, int]
    swizzle_size: int = 1
    raster_along_m: bool = True
    batch_count: int = 1

    def validate(self) -> None:
        if min(*self.cta_tile_shape_mn, *self.cluster_shape_mnk) <= 0:
            raise ValueError("scheduler tile and cluster shapes must be positive")
        if self.cluster_shape_mnk[2] != 1:
            raise NotImplementedError("cluster_shape_k != 1 is not supported")
        if self.swizzle_size != 1:
            raise NotImplementedError("scheduler swizzling is not implemented")
        if self.batch_count < 1:
            raise ValueError("scheduler batch_count must be positive")

    def cluster_count_m(self, m):
        cta_tiles_m = cl.cdiv(m, self.cta_tile_shape_mn[0])
        return cl.cdiv(cta_tiles_m, self.cluster_shape_mnk[0])

    def cluster_count_n(self, n):
        cta_tiles_n = cl.cdiv(n, self.cta_tile_shape_mn[1])
        return cl.cdiv(cta_tiles_n, self.cluster_shape_mnk[1])

    def problem_cluster_count(self, m, n):
        return self.cluster_count_m(m) * self.cluster_count_n(n) * self.batch_count

    def initial_work_index(self):
        return cl.block_index(2)

    def persistent_stride(self):
        return cl.block_count(2)

    def is_valid(self, work_idx, m, n):
        return work_idx < self.problem_cluster_count(m, n)

    def tile_mnl(self, work_idx, m, n):
        """Map a linear cluster index to an M/N/batch tile coordinate."""
        clusters_m = self.cluster_count_m(m)
        clusters_n = self.cluster_count_n(n)
        if self.raster_along_m:
            tile_m = work_idx % clusters_m
            remainder = work_idx // clusters_m
            tile_n = remainder % clusters_n
            tile_l = remainder // clusters_n
        else:
            tile_n = work_idx % clusters_n
            remainder = work_idx // clusters_n
            tile_m = remainder % clusters_m
            tile_l = remainder // clusters_m
        return tile_m, tile_n, tile_l

    def advance(self, work_idx):
        return work_idx + self.persistent_stride()

    def max_active_clusters(self) -> int:
        """Derive the launch-time cluster count from PyTorch device properties."""
        cluster_ctas = self.cluster_shape_mnk[0] * self.cluster_shape_mnk[1]
        device = torch.cuda.current_device()
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        return max(1, sm_count // cluster_ctas)

    def host_grid(self, m: int, n: int) -> tuple[int, int, int]:
        problem_clusters = self.problem_cluster_count(m, n)
        persistent_clusters = min(problem_clusters, self.max_active_clusters())
        return (*self.cluster_shape_mnk[:2], persistent_clusters)


# The scheduler abstraction supports other unswizzled tile and cluster shapes,
# but this kernel's CTA_2 MMA, TMA partitioning, and epilogue are specialized to
# the source tutorial's exact shapes.
_TILE_SCHEDULER = _StaticPersistentTileScheduler(
    cta_tile_shape_mn=(CTA_M, TILE_N),
    cluster_shape_mnk=(CLUSTER_M, 1, 1),
    swizzle_size=1,
    raster_along_m=True,
    batch_count=1,
)
# Keep validation outside __post_init__: CUDA Lang supports frozen dataclasses
# in kernels, but rejects dataclass types that define __post_init__.
_TILE_SCHEDULER.validate()


@cl.kernel
def _kernel(a, b, c, bias, k: cl.Constant[int], has_bias: cl.Constant[bool]):
    """Warp-specialized persistent two-CTA FP16 tcgen05 GEMM kernel."""
    m, n = c.shape
    cl.static_assert(k % 8 == 0, "K must be divisible by 8 for TMA alignment")

    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    rank = cl.block_in_cluster_index(0)
    is_leader = rank == 0

    a_tmap = cl.tensor_map_tiled(
        a,
        (BLOCK_K, CTA_M),
        order="F",
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (BLOCK_K, CTA_N),
        order="F",
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )

    if warp == MMA_WARP:
        cl.prefetch_tensor_map(a_tmap)
        cl.prefetch_tensor_map(b_tmap)

    ab_full = cl.shared_array(AB_STAGES, cl.mbarrier, alignment=8)
    ab_empty = cl.shared_array(AB_STAGES, cl.mbarrier, alignment=8)
    acc_empty = cl.shared_array(ACC_STAGES, cl.mbarrier, alignment=8)
    acc_full = cl.shared_array(ACC_STAGES, cl.mbarrier, alignment=8)
    tmem_dealloc = cl.shared_array(1, cl.mbarrier, alignment=8)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
    )
    a_smem = cl.shared_array(
        (AB_STAGES, A_STAGE_ELEMS),
        cl.float16,
        alignment=128,
    )
    b_smem = cl.shared_array(
        (AB_STAGES, B_STAGE_ELEMS),
        cl.float16,
        alignment=128,
    )

    tmem_dealloc_ptr = tmem_dealloc.get_base_pointer()
    tmem_storage_ptr = tmem_storage.get_base_pointer()

    if warp == 0 and cl.elect_sync():
        cl.mbarrier_initialize(tmem_dealloc_ptr, WARP_SIZE)
        for stage in cl.static_iter(range(ACC_STAGES)):
            cl.mbarrier_initialize(acc_empty.get_element_pointer(stage), 8)
            cl.mbarrier_initialize(acc_full.get_element_pointer(stage), 1)
        for stage in cl.static_iter(range(AB_STAGES)):
            cl.mbarrier_initialize(ab_full.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(ab_empty.get_element_pointer(stage), 1)

    cl.fence_mbarrier_initialize()
    cl.barrier_arrive_cluster(
        aligned=False, memory_order=cl.MemoryOrder.RELAXED
    )

    instruction = cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        n=TILE_N,
        m=TILE_M,
    ).encode()

    cl.barrier_wait_cluster(aligned=False)

    if warp == TMA_WARP:
        ab_stage_idx = 0
        ab_empty_phase = 1
        work_idx = _TILE_SCHEDULER.initial_work_index()
        while _TILE_SCHEDULER.is_valid(work_idx, m, n):
            mma_tile_m, mma_tile_n, _ = _TILE_SCHEDULER.tile_mnl(
                work_idx, m, n
            )

            for k_tile in range(cl.cdiv(k, BLOCK_K)):
                current_ab_stage = ab_stage_idx
                ab_full_stage = ab_full.get_element_pointer(current_ab_stage)
                ab_empty_stage = ab_empty.get_element_pointer(current_ab_stage)
                current_ab_empty_phase = ab_empty_phase

                ab_stage_idx += 1
                if ab_stage_idx == AB_STAGES:
                    ab_stage_idx = 0
                    ab_empty_phase = ab_empty_phase ^ 1

                a_stage = a_smem.get_element_pointer((current_ab_stage, 0))
                b_stage = b_smem.get_element_pointer((current_ab_stage, 0))
                coord_k = k_tile * BLOCK_K
                coord_m = mma_tile_m * TILE_M + rank * CTA_M
                coord_n = mma_tile_n * TILE_N + rank * CTA_N

                cl.mbarrier_wait_parity(ab_empty_stage, current_ab_empty_phase)

                if is_leader and cl.elect_sync():
                    cl.mbarrier_arrive_expect_transaction(
                        ab_full_stage,
                        TMA_COPY_BYTES,
                        scope=cl.MbarrierScope.BLOCK,
                    )

                if cl.elect_sync():
                    a_dst = cl.address_space_cast(
                        a_stage, cl.MemorySpace.SHARED_CLUSTER
                    )
                    arrive_bar = cl.map_shared_to_leader_block(ab_full_stage)
                    tma_mask = cl.int16(1 << rank)
                    cl.copy_async_bulk_tensor_global_to_shared(
                        a_tmap,
                        (coord_k, coord_m),
                        a_dst,
                        arrive_bar,
                        multicast_mask=tma_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )

                if cl.elect_sync():
                    b_dst = cl.address_space_cast(
                        b_stage, cl.MemorySpace.SHARED_CLUSTER
                    )
                    arrive_bar = cl.map_shared_to_leader_block(ab_full_stage)
                    tma_mask = cl.int16(1 << rank)
                    cl.copy_async_bulk_tensor_global_to_shared(
                        b_tmap,
                        (coord_k, coord_n),
                        b_dst,
                        arrive_bar,
                        multicast_mask=tma_mask,
                        cta_group=cl.CTAGroup.CTA_2,
                    )

            work_idx = _TILE_SCHEDULER.advance(work_idx)

        tail_stage = ab_stage_idx
        tail_phase = ab_empty_phase
        for _ in cl.static_iter(range(AB_STAGES - 1)):
            tail_stage += 1
            if tail_stage == AB_STAGES:
                tail_stage = 0
                tail_phase = tail_phase ^ 1
        if cl.elect_sync():
            cl.mbarrier_wait_parity(
                ab_empty.get_element_pointer(tail_stage), tail_phase
            )

    elif warp == MMA_WARP:
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )
        tmem_base = tmem_storage[0]

        ab_stage_idx = 0
        ab_full_phase = 0
        acc_stage_idx = 0
        acc_empty_phase = 1
        work_idx = _TILE_SCHEDULER.initial_work_index()

        while _TILE_SCHEDULER.is_valid(work_idx, m, n):
            current_acc_stage = acc_stage_idx
            acc_empty_stage = acc_empty.get_element_pointer(current_acc_stage)
            acc_full_stage = acc_full.get_element_pointer(current_acc_stage)
            current_acc_empty_phase = acc_empty_phase

            acc_stage_idx += 1
            if acc_stage_idx == ACC_STAGES:
                acc_stage_idx = 0
                acc_empty_phase = acc_empty_phase ^ 1

            if is_leader:
                cl.mbarrier_wait_parity(
                    acc_empty_stage, current_acc_empty_phase
                )

                tmem_for_mma = cl.tcgen05_tmem_offset(
                    tmem_base,
                    column_offset=current_acc_stage * TILE_N,
                )
                scale_d = False
                for k_tile in range(cl.cdiv(k, BLOCK_K)):
                    current_ab_stage = ab_stage_idx
                    ab_full_stage = ab_full.get_element_pointer(current_ab_stage)
                    ab_empty_stage = ab_empty.get_element_pointer(current_ab_stage)
                    current_ab_full_phase = ab_full_phase

                    ab_stage_idx += 1
                    if ab_stage_idx == AB_STAGES:
                        ab_stage_idx = 0
                        ab_full_phase = ab_full_phase ^ 1

                    cl.mbarrier_wait_parity(
                        ab_full_stage, current_ab_full_phase
                    )

                    a_stage = a_smem.get_element_pointer((current_ab_stage, 0))
                    b_stage = b_smem.get_element_pointer((current_ab_stage, 0))
                    a_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=a_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()
                    b_desc = cl.Tcgen05SharedMemoryDescriptor(
                        matrix_start_address=b_stage,
                        leading_dimension_byte_offset=16,
                        stride_dimension_byte_offset=8 * 128,
                        swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                    ).encode()

                    for kk in cl.static_iter(range(BLOCK_K // MMA_K)):
                        if cl.elect_sync():
                            cl.tcgen05_mma(
                                cl.Tcgen05MMAKind.F16,
                                tmem_for_mma,
                                cl.int64(a_desc + 2 * kk),
                                cl.int64(b_desc + 2 * kk),
                                cl.int32(instruction),
                                accumulate=scale_d,
                                cta_group=cl.CTAGroup.CTA_2,
                            )
                        scale_d = True

                    if cl.elect_sync():
                        cl.tcgen05_commit(
                            ab_empty_stage,
                            multicast_mask=0b11,
                            cta_group=cl.CTAGroup.CTA_2,
                        )

                if cl.elect_sync():
                    cl.tcgen05_commit(
                        acc_full_stage,
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )

            work_idx = _TILE_SCHEDULER.advance(work_idx)

        tail_stage = acc_stage_idx
        tail_phase = acc_empty_phase
        if is_leader:
            for _ in cl.static_iter(range(ACC_STAGES - 1)):
                tail_stage += 1
                if tail_stage == ACC_STAGES:
                    tail_stage = 0
                    tail_phase = tail_phase ^ 1
            if cl.elect_sync():
                cl.mbarrier_wait_parity(
                    acc_empty.get_element_pointer(tail_stage), tail_phase
                )

    elif warp < MMA_WARP:
        if warp == ALLOCATOR_WARP:
            cl.tcgen05_allocate(
                tmem_storage_ptr,
                TMEM_COLS,
                cta_group=cl.CTAGroup.CTA_2,
            )
            cl.tcgen05_relinquish_allocation_permit(
                cta_group=cl.CTAGroup.CTA_2
            )

        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )
        tmem_base = tmem_storage[0]

        vsize = VEC_BYTES // 2  # sizeof(float16) == 2
        acc_stage_idx = 0
        acc_full_phase = 0
        work_idx = _TILE_SCHEDULER.initial_work_index()

        while _TILE_SCHEDULER.is_valid(work_idx, m, n):
            current_acc_stage = acc_stage_idx
            acc_full_stage = acc_full.get_element_pointer(current_acc_stage)
            acc_empty_stage = acc_empty.get_element_pointer(current_acc_stage)
            current_acc_full_phase = acc_full_phase

            acc_stage_idx += 1
            if acc_stage_idx == ACC_STAGES:
                acc_stage_idx = 0
                acc_full_phase = acc_full_phase ^ 1

            mma_tile_m, mma_tile_n, _ = _TILE_SCHEDULER.tile_mnl(
                work_idx, m, n
            )
            coordc_m = mma_tile_m * TILE_M + rank * CTA_M
            coordc_n = mma_tile_n * TILE_N

            cl.mbarrier_wait_parity(acc_full_stage, current_acc_full_phase)

            row = coordc_m + tid
            for subtile in cl.static_iter(range(TILE_N // 32)):
                column = subtile * 32
                tmem = cl.tcgen05_tmem_offset(
                    tmem_base,
                    lane_offset=warp * WARP_SIZE,
                    column_offset=current_acc_stage * TILE_N + column,
                )
                regs = cl.tcgen05_load(
                    cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                    tmem,
                    count=32,
                )
                accumulators = _as_float32_vector(regs)
                if has_bias:
                    accumulators = accumulators + cl.float32(bias[row])

                if row < m:
                    # No partial vector stores: run() requires N divisible by
                    # vsize, and this guard suppresses only whole tail vectors.
                    for j in cl.static_iter(range(32 // vsize)):
                        col_j = coordc_n + column + j * vsize
                        if col_j + vsize <= n:
                            packed = _to_float16_vector(
                                accumulators, j * vsize, vsize
                            )
                            dst = c.get_element_pointer((row, col_j))
                            dst.store(packed, alignment=VEC_BYTES)

            if cl.elect_sync():
                empty_bar = cl.map_shared_to_cluster(acc_empty_stage, 0)
                cl.mbarrier_arrive(empty_bar, scope=cl.MbarrierScope.BLOCK)

            work_idx = _TILE_SCHEDULER.advance(work_idx)

        cl.barrier_sync_block(
            number_of_threads=EPILOGUE_WARPS * WARP_SIZE,
            barrier_id=DEALLOC_BARRIER_ID,
        )

        if warp == ALLOCATOR_WARP:
            peer_rank = rank ^ 1
            peer_mbar = cl.map_shared_to_cluster(tmem_dealloc_ptr, peer_rank)
            cl.mbarrier_arrive(peer_mbar, scope=cl.MbarrierScope.BLOCK)
            cl.mbarrier_wait_parity(tmem_dealloc_ptr, 0)
            cl.tcgen05_deallocate(
                tmem_base,
                TMEM_COLS,
                cta_group=cl.CTAGroup.CTA_2,
            )


def _validate_mnk(mnk: tuple[int, int, int]) -> None:
    if len(mnk) != 3:
        raise ValueError("MNK must contain exactly three values")
    m, n, k = mnk
    if min(m, n, k) <= 0:
        raise ValueError("MNK values must be positive")
    vsize = VEC_BYTES // 2  # sizeof(float16) == 2
    if n % vsize != 0:
        raise ValueError(f"N must be divisible by {vsize} (got n={n})")
    k_align_elems = 16 // 2  # 16-byte TMA stride alignment for float16
    if k % k_align_elems != 0:
        raise ValueError(
            f"K must be divisible by {k_align_elems} for TMA alignment (got k={k})"
        )


def FLOPS_FORMULA(m: int, n: int, k: int, has_bias: bool = False, **_) -> int:
    return 2 * m * n * k + (m * n if has_bias else 0)


def prepare_tensors(
    m: int, n: int, k: int, has_bias: bool = False, **_
) -> dict[str, torch.Tensor]:
    _validate_mnk((m, n, k))
    torch.manual_seed(1111)

    def _make(rows: int, cols: int) -> torch.Tensor:
        return (
            torch.empty(rows, cols, dtype=torch.int32)
            .random_(-2, 2)
            .to(device="cuda", dtype=torch.float16)
        )

    tensors = {"a": _make(m, k), "b": _make(n, k)}
    tensors["c"] = torch.empty((m, n), device="cuda", dtype=torch.float16)
    if has_bias:
        tensors["bias"] = torch.randn(m, device="cuda", dtype=torch.float16)
    return tensors


def run(tensors: dict[str, torch.Tensor], stream=None) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    m, k = a.shape
    if b.ndim != 2 or b.shape[1] != k:
        raise ValueError("B must have shape (N, K) with the same K as A")
    n = b.shape[0]
    if c.shape != (m, n):
        raise ValueError("C must have shape (M, N)")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("A and B must have dtype torch.float16")
    if c.dtype != torch.float16:
        raise ValueError("C must have dtype torch.float16")
    _validate_mnk((m, n, k))

    bias = tensors.get("bias")
    if bias is not None and bias.shape != (m,):
        raise ValueError("bias must have shape (M,)")
    if bias is not None and bias.dtype != torch.float16:
        raise ValueError("bias must have dtype torch.float16")

    bias_arg = c.reshape(-1) if bias is None else bias
    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    cl.launch(
        cuda_stream,
        _TILE_SCHEDULER.host_grid(m, n),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (a, b, c, bias_arg, k, bias is not None),
        block_in_cluster_count=_TILE_SCHEDULER.cluster_shape_mnk,
    )


def verify_output(
    tensors: dict[str, torch.Tensor], tolerance: float = 1.0e-4, **_
) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    reference = torch.einsum("mk,nk->mn", a.float(), b.float())
    bias = tensors.get("bias")
    if bias is not None:
        reference = reference + bias.float()[:, None]
    torch.testing.assert_close(
        c, reference.to(torch.float16), atol=tolerance, rtol=1.0e-5
    )


def verify(
    mnk: tuple[int, int, int] = _DEFAULT_MNK,
    has_bias: bool = False,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> None:
    _validate_mnk(mnk)
    m, n, k = mnk
    tensors = prepare_tensors(m=m, n=n, k=k, has_bias=has_bias)
    run(tensors)
    torch.cuda.synchronize()
    print(f"Run kernel (mnk={mnk}, has_bias={has_bias}) OK", flush=True)
    verify_output(tensors, tolerance=tolerance)
    print(f"verify (mnk={mnk}, has_bias={has_bias}): PASS")


def _parse_mnk(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three MNK values.")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CUDA Lang warp-specialized persistent two-CTA fp16 GEMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mnk", type=_parse_mnk, default=_DEFAULT_MNK, help="M,N,K dimensions"
    )
    parser.add_argument("--has_bias", action="store_true", help="Whether to use bias")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help="Tolerance for validation",
    )
    args = parser.parse_args()
    verify(args.mnk, has_bias=args.has_bias, tolerance=args.tolerance)
