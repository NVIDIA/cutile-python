# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Lang port of the CuTe DSL ``fp16_gemm_1.py`` tutorial.

Computes ``C = A @ B.T`` with an optional FP16 row bias. Two CTAs collaborate
on each 256x256 output tile: each CTA loads one 128-row A slice and one
128-column B slice, the leader issues CTA_2 tcgen05 MMA instructions, and both
CTAs store their 128 output rows.

Additional CuTe DSL -> CUDA Lang notes beyond ``fp16_gemm_0.py``:

* CuTe DSL differentiates CTA-local and cluster TMA with different primitive
  names: ``cp_async_bulk_tensor_shared_cta_global`` in ``fp16_gemm_0.py`` and
  ``cp_async_bulk_tensor_shared_cluster_global`` here. CUDA Lang uses the same
  ``cl.copy_async_bulk_tensor_global_to_shared`` API for both; cluster TMA is
  selected by passing a destination pointer in
  ``cl.MemorySpace.SHARED_CLUSTER`` plus the CTA_2 multicast arguments.
  Completion still has to arrive on the leader CTA's mbarrier, so this sample
  maps the local ``ab_full`` pointer with ``cl.map_shared_to_leader_block``.
* The source kernel takes K as a runtime argument. This sample keeps M, N, and K
  runtime dimensions; only the optional-bias path is specialized.

The implementation otherwise retains the source's cluster shape, work
decomposition, data movement, barrier phases, CTA_2 MMA sequence, TMEM
epilogue, and vector stores.
"""

from __future__ import annotations

import argparse

import cuda.lang as cl
import torch


WARP_SIZE = 32
BLOCK_THREADS = 128
CLUSTER_M = 2
CTA_M = 128
CTA_N = 128
TILE_M = CLUSTER_M * CTA_M
TILE_N = CLUSTER_M * CTA_N
BLOCK_K = 64
MMA_K = 16
VEC_BYTES = 32

_DEFAULT_MNK = (256, 256, 64)
_DEFAULT_TOLERANCE = 1.0e-1


def _as_float32_vector(regs):
    """Interpret the 32 TMEM load registers as one FP32 vector."""
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


@cl.kernel
def _kernel(a, b, c, bias, has_bias: cl.Constant[bool]):
    """Two-CTA FP16 tcgen05 GEMM kernel."""
    m, n = c.shape
    k = a.shape[1]
    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    block_m = cl.block_index(0)
    block_n = cl.block_index(1)
    rank = cl.block_in_cluster_index(0)
    is_leader = rank == 0

    # Each CTA loads one half of A and one half of B. Together the pair forms
    # the 256x256 collective tile consumed by CTA_2 MMA.
    cta_m = rank * CTA_M
    off_m = (block_m // CLUSTER_M) * TILE_M + cta_m
    off_n_b = block_n * TILE_N + rank * CTA_N
    off_n_c = block_n * TILE_N

    # A/B are row-major (rows, K). order="F" makes the tensor-map coordinate
    # order (K, rows), matching the source descriptors.
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

    ab_full = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    ab_empty = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    acc_full = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR),
        alignment=4,
        dynamic=True,
    )
    a_smem = cl.shared_array(
        CTA_M * BLOCK_K, cl.float16, alignment=128, dynamic=True
    )
    b_smem = cl.shared_array(
        CTA_N * BLOCK_K, cl.float16, alignment=128, dynamic=True
    )

    ab_full_ptr = ab_full.get_base_pointer()
    ab_empty_ptr = ab_empty.get_base_pointer()
    acc_full_ptr = acc_full.get_base_pointer()
    tmem_storage_ptr = tmem_storage.get_base_pointer()
    a_smem_ptr = a_smem.get_base_pointer()
    b_smem_ptr = b_smem.get_base_pointer()

    if warp == 0 and cl.elect_sync():
        cl.mbarrier_initialize(ab_full_ptr, 1)
        cl.mbarrier_initialize(ab_empty_ptr, 1)
        cl.mbarrier_initialize(acc_full_ptr, 1)
    cl.fence_mbarrier_initialize()
    cl.barrier_arrive_cluster(
        aligned=False, memory_order=cl.MemoryOrder.RELAXED
    )

    # Warp 0 in both CTAs participates in the CTA_2 allocation.
    if warp == 0:
        cl.tcgen05_allocate(
            tmem_storage_ptr,
            512,
            cta_group=cl.CTAGroup.CTA_2,
        )

    cl.barrier_wait_cluster(aligned=False)
    cl.barrier_sync_block()
    tmem_base = tmem_storage[0]

    if warp == 0:
        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            n=TILE_N,
            m=TILE_M,
        ).encode()
        ab_empty_phase = 1
        ab_full_phase = 0
        scale_d = False

        for k_tile in range(cl.cdiv(k, BLOCK_K)):
            cl.mbarrier_wait_parity(ab_empty_ptr, ab_empty_phase)
            ab_empty_phase = ab_empty_phase ^ 1

            # Both CTAs contribute their A and B transfers to the leader's
            # full barrier. Each transfer targets only its issuing CTA.
            if is_leader and cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    ab_full_ptr,
                    2 * (CTA_M + CTA_N) * BLOCK_K * 2,
                    scope=cl.MbarrierScope.BLOCK,
                )

            if cl.elect_sync():
                a_dst = cl.address_space_cast(
                    a_smem_ptr, cl.MemorySpace.SHARED_CLUSTER
                )
                arrive_bar = cl.map_shared_to_leader_block(ab_full_ptr)
                tma_mask = cl.int16(1 << rank)
                cl.copy_async_bulk_tensor_global_to_shared(
                    a_tmap,
                    (k_tile * BLOCK_K, off_m),
                    a_dst,
                    arrive_bar,
                    multicast_mask=tma_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )
            if cl.elect_sync():
                b_dst = cl.address_space_cast(
                    b_smem_ptr, cl.MemorySpace.SHARED_CLUSTER
                )
                arrive_bar = cl.map_shared_to_leader_block(ab_full_ptr)
                tma_mask = cl.int16(1 << rank)
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (k_tile * BLOCK_K, off_n_b),
                    b_dst,
                    arrive_bar,
                    multicast_mask=tma_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )

            if is_leader:
                cl.mbarrier_wait_parity(ab_full_ptr, ab_full_phase)
                ab_full_phase = ab_full_phase ^ 1

                a_desc = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=a_smem_ptr,
                    leading_dimension_byte_offset=16,
                    stride_dimension_byte_offset=8 * 128,
                    swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                ).encode()
                b_desc = cl.Tcgen05SharedMemoryDescriptor(
                    matrix_start_address=b_smem_ptr,
                    leading_dimension_byte_offset=16,
                    stride_dimension_byte_offset=8 * 128,
                    swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
                ).encode()

                for kk in cl.static_iter(range(BLOCK_K // MMA_K)):
                    if cl.elect_sync():
                        cl.tcgen05_mma(
                            cl.Tcgen05MMAKind.F16,
                            tmem_base,
                            cl.int64(a_desc + 2 * kk),
                            cl.int64(b_desc + 2 * kk),
                            cl.int32(instruction),
                            accumulate=scale_d,
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    scale_d = True

                if cl.elect_sync():
                    cl.tcgen05_commit(
                        ab_empty_ptr,
                        multicast_mask=0b11,
                        cta_group=cl.CTAGroup.CTA_2,
                    )

        if is_leader and cl.elect_sync():
            cl.tcgen05_commit(
                acc_full_ptr,
                multicast_mask=0b11,
                cta_group=cl.CTAGroup.CTA_2,
            )

    if warp == 0:
        cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_2)

    cl.mbarrier_wait_parity(acc_full_ptr, 0)

    # Each CTA stores its own 128-row half. Its TMEM base is already the base
    # for that CTA's half of the CTA_2 allocation.
    vsize = VEC_BYTES // 2  # sizeof(float16) == 2
    row = off_m + tid
    # No partial vector stores: run() requires N to be divisible by vsize.
    for column in cl.static_iter(range(0, TILE_N, 32)):
        tmem = cl.tcgen05_tmem_offset(
            tmem_base,
            lane_offset=warp * WARP_SIZE,
            column_offset=column,
        )
        regs = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B, tmem, count=32
        )
        accumulators = _as_float32_vector(regs)
        if has_bias:
            accumulators = accumulators + cl.float32(bias[row])
        if row < m:
            for j in cl.static_iter(range(32 // vsize)):
                col_j = off_n_c + column + j * vsize
                if col_j + vsize <= n:
                    packed = _to_float16_vector(accumulators, j * vsize, vsize)
                    dst = c.get_element_pointer((row, col_j))
                    dst.store(packed, alignment=VEC_BYTES)

    cl.barrier_sync_block()
    if warp == 0:
        cl.tcgen05_deallocate(
            tmem_base, 512, cta_group=cl.CTAGroup.CTA_2
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
    _validate_mnk((m, n, k))

    bias = tensors.get("bias")
    if bias is not None and bias.shape != (m,):
        raise ValueError("bias must have shape (M,)")

    # The signature always includes a tensor for bias; the bias-free
    # specialization ignores this same-device placeholder.
    bias_arg = c.reshape(-1) if bias is None else bias
    grid_m = cl.cdiv(cl.cdiv(m, CTA_M), CLUSTER_M) * CLUSTER_M
    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    cl.launch(
        cuda_stream,
        (grid_m, cl.cdiv(n, TILE_N), 1),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (a, b, c, bias_arg, bias is not None),
        block_in_cluster_count=(CLUSTER_M, 1, 1),
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
        description="CUDA Lang two-CTA fp16 GEMM - verify correctness",
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
