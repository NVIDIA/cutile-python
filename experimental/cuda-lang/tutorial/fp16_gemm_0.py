# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Lang port of the CuTe DSL ``fp16_gemm_0.py`` tutorial.

Computes ``C = A @ B.T`` with an optional FP16 row bias. The kernel retains
the source tutorial's 128x128x64 single-CTA baseline: one shared-memory stage,
explicit TMA and mbarrier operations, tcgen05 MMA into TMEM, FP32 accumulation,
and an FP16 epilogue.

CuTe DSL -> CUDA Lang API mapping used here:

* ``@cute.kernel`` maps to ``@cl.kernel``. CuTe's ``warp_idx``, thread/block
  indices, ``ceil_div``, and ``range_constexpr`` map to the corresponding
  ``cl`` index operations, ``cl.cdiv``, and ``cl.static_iter``.
* ``cutlass.Array(..., space=cutlass.AddressSpace.smem)`` maps to
  ``cl.shared_array``. The A/B arrays are flattened in CUDA Lang, but retain
  the source sizes, 128-byte alignment, and physical shared-memory layout.
* ``cuda.create_tensor_map_tiled_from_view`` maps to ``cl.tensor_map_tiled``.
  A and B remain row-major, K-contiguous 2-D tensors. CUDA Lang expresses the
  TMA coordinate order explicitly as ``(K, M)`` / ``(K, N)`` using tile shapes
  ``(BLOCK_K, BLOCK_M)`` / ``(BLOCK_K, BLOCK_N)`` and ``order="F"``; this is
  equivalent to CuTe DSL automatically selecting K as the leading mode.
* ``prims.cp_async_bulk_tensor_shared_cta_global`` maps to
  ``cl.copy_async_bulk_tensor_global_to_shared``. The expected transaction
  byte count and the TMA-completion mbarrier are unchanged.
* ``prims.mbarrier_init``, ``mbarrier_arrive_expect_tx``, and
  ``mbarrier_try_wait_parity`` map to ``cl.mbarrier_initialize``,
  ``cl.mbarrier_arrive_expect_transaction``, and
  ``cl.mbarrier_wait_parity``. The full/empty/accumulator barrier topology
  and phase progression are preserved.
* ``prims.Tcgen05InstrDesc.build`` and ``Tcgen05SmemDesc.build`` map to
  ``cl.Tcgen05InstructionDescriptor`` and
  ``cl.Tcgen05SharedMemoryDescriptor``, each followed by ``encode()``. Both
  builders take byte-valued leading/stride offsets. Once encoded, the per-MMA
  K-step is added in descriptor units of 16 bytes, hence ``2 * kk`` for each
  32-byte F16 MMA K-step.
* ``prims.tcgen05_alloc``, ``tcgen05_mma``, ``tcgen05_commit``,
  ``tcgen05_ld``, and ``tcgen05_dealloc`` map directly to the corresponding
  ``cl.tcgen05_*`` operations. Allocation size, MMA accumulation predicate,
  commit barriers, 32-column TMEM loads, and deallocation size match the
  source.
* The CuTe ``@cute.jit`` host function builds tensor maps and launches the
  device kernel. CUDA Lang instead declares tensor maps in ``@cl.kernel``;
  compilation hoists their construction into the generated host program, so
  both public launch interfaces still accept ordinary 2-D tensors.
* ``cute.compile`` has no explicit CUDA Lang counterpart here. ``run()`` calls
  ``cl.launch`` directly; the first launch compiles the kernel and later
  launches reuse CUDA Lang's internal cache. K and ``has_bias`` are
  ``cl.Constant`` specialization keys, while M and N remain runtime dimensions
  of C.

The CLI retains the source tutorial's correctness-checking interface. CUDA
Lang-specific restrictions are validated explicitly.
"""

from __future__ import annotations

import argparse

import cuda.lang as cl
import torch


WARP_SIZE = 32
BLOCK_THREADS = 128
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
MMA_K = 16
VEC_BYTES = 32

_DEFAULT_MNK = (128, 128, 64)
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
def _kernel(
    a,
    b,
    c,
    bias,
    k: cl.Constant[int],
    has_bias: cl.Constant[bool],
):
    """Single-CTA FP16 tcgen05 GEMM kernel."""
    cl.static_assert(k % 8 == 0, "K must be divisible by 8 for TMA alignment")
    m, n = c.shape
    tid = cl.thread_index(0)
    warp = tid // WARP_SIZE
    tile_m = cl.block_index(0)
    tile_n = cl.block_index(1)
    off_m = tile_m * BLOCK_M
    off_n = tile_n * BLOCK_N

    # A/B are ordinary row-major (rows, K) arrays. ``order="F"`` reverses the
    # tensor-map modes to (K, rows), making K the contiguous TMA coordinate.
    # This is the explicit CUDA Lang equivalent of CuTe DSL's leading-mode
    # detection in create_tensor_map_tiled_from_view.
    a_tmap = cl.tensor_map_tiled(
        a,
        (BLOCK_K, BLOCK_M),
        order="F",
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (BLOCK_K, BLOCK_N),
        order="F",
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )

    ab_full = cl.shared_array(1, cl.mbarrier, alignment=8)
    ab_empty = cl.shared_array(1, cl.mbarrier, alignment=8)
    acc_full = cl.shared_array(1, cl.mbarrier, alignment=8)
    tmem_storage = cl.shared_array(
        1, cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR), alignment=4
    )
    a_smem = cl.shared_array(BLOCK_M * BLOCK_K, cl.float16, alignment=128)
    b_smem = cl.shared_array(BLOCK_N * BLOCK_K, cl.float16, alignment=128)

    if warp == 0 and cl.elect_sync():
        cl.mbarrier_initialize(ab_full.get_base_pointer(), 1)
        cl.mbarrier_initialize(ab_empty.get_base_pointer(), 1)
        cl.mbarrier_initialize(acc_full.get_base_pointer(), 1)
    cl.fence_mbarrier_initialize()
    cl.barrier_sync_block()

    # Match the source tutorial's full TMEM allocation.
    if warp == 0:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(), 512, cta_group=cl.CTAGroup.CTA_1
        )
    cl.barrier_sync_block()
    tmem_base = tmem_storage[0]

    if warp == 0:
        instruction = cl.Tcgen05InstructionDescriptor(
            d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
            a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
            n=BLOCK_N,
            m=BLOCK_M,
        ).encode()
        ab_empty_phase = 1
        ab_full_phase = 0
        scale_d = False
        for k_tile in range(cl.cdiv(k, BLOCK_K)):
            cl.mbarrier_wait_parity(ab_empty.get_base_pointer(), ab_empty_phase)
            ab_empty_phase = ab_empty_phase ^ 1

            # The elected lane issues both TMA loads and contributes the single
            # expected arrival. TMA completes the transaction bytes.
            if cl.elect_sync():
                cl.mbarrier_arrive_expect_transaction(
                    ab_full.get_base_pointer(),
                    (BLOCK_M + BLOCK_N) * BLOCK_K * 2,
                    scope=cl.MbarrierScope.BLOCK,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    a_tmap,
                    (k_tile * BLOCK_K, off_m),
                    a_smem.get_base_pointer(),
                    ab_full.get_base_pointer(),
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (k_tile * BLOCK_K, off_n),
                    b_smem.get_base_pointer(),
                    ab_full.get_base_pointer(),
                )

            cl.mbarrier_wait_parity(ab_full.get_base_pointer(), ab_full_phase)
            ab_full_phase = ab_full_phase ^ 1

            a_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=a_smem,
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            b_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=b_smem,
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
                        cta_group=cl.CTAGroup.CTA_1,
                    )
                scale_d = True

            if cl.elect_sync():
                cl.tcgen05_commit(
                    ab_empty.get_base_pointer(), cta_group=cl.CTAGroup.CTA_1
                )

        if cl.elect_sync():
            cl.tcgen05_commit(
                acc_full.get_base_pointer(), cta_group=cl.CTAGroup.CTA_1
            )

        cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)

    cl.mbarrier_wait_parity(acc_full.get_base_pointer(), 0)

    # Each thread owns one row. Match the source's four 32-column TMEM loads.
    # This epilogue has no partial-vector store fallback, so the host requires
    # N to be divisible by vsize.
    vsize = VEC_BYTES // 2  # sizeof(float16) == 2
    row = off_m + tid
    bias_value = cl.float32(0.0)
    if has_bias:
        bias_value = cl.float32(bias[row])
    for column in cl.static_iter(range(0, BLOCK_N, 32)):
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
            accumulators = accumulators + bias_value
        if row < m:
            for j in cl.static_iter(range(32 // vsize)):
                col_j = off_n + column + j * vsize
                if col_j + vsize <= n:
                    packed = _to_float16_vector(accumulators, j * vsize, vsize)
                    dst = c.get_element_pointer((row, col_j))
                    dst.store(packed, alignment=VEC_BYTES)

    cl.barrier_sync_block()
    if warp == 0:
        cl.tcgen05_deallocate(
            tmem_base, 512, cta_group=cl.CTAGroup.CTA_1
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
    if k % 8 != 0:
        raise ValueError(f"K must be divisible by 8 for TMA alignment (got k={k})")


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

    # The kernel signature always includes a bias tensor. The bias-free
    # specialization ignores this same-device placeholder.
    bias_arg = c.reshape(-1) if bias is None else bias
    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    cl.launch(
        cuda_stream,
        (cl.cdiv(m, BLOCK_M), cl.cdiv(n, BLOCK_N), 1),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (a, b, c, bias_arg, k, bias is not None),
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
        description="CUDA Lang single-CTA fp16 GEMM — verify correctness",
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
