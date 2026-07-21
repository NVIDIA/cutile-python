# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Lang port of the CuTe DSL ``nvfp4_gemm_1_ws.py`` tutorial.

Two CTAs collaborate on each 256x256x256 collective tile.  Each CTA loads a
128-row A slice and a 128-column B slice, while B scale factors are multicast
to both CTAs.  The leader issues CTA_2 MXF4NVF4 tcgen05 MMA instructions and
each CTA writes its own 128 output rows as FP16.

The port retains the six specialized warps, five-stage input pipeline,
cluster TMA transfers, SMEM-to-TMEM scale staging, FP32 accumulation, and
vectorized epilogue used by the CuTe DSL source.
"""

from __future__ import annotations

import argparse
from typing import Any

import cuda.lang as cl
import cuda.tile as ct
import torch


WARP_SIZE = 32
BLOCK_THREADS = 6 * WARP_SIZE
EPILOGUE_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5

CLUSTER_M = 2
CTA_M = 128
CTA_N = 128
BLOCK_M = CLUSTER_M * CTA_M
BLOCK_N = CLUSTER_M * CTA_N
BLOCK_K = 256
PACKED_BLOCK_K = BLOCK_K // 2
MMA_K = 64
SF_VECTOR_SIZE = 16
SF_K_PER_TILE = BLOCK_K // SF_VECTOR_SIZE
AB_STAGES = 5
TMEM_COLUMNS = 512
# CuTe DSL sets descriptor bit 12 (sparsity_version). It is ignored on SM100,
# but preserving it makes the emitted MMA descriptor match the reference PTX.
NVFP4_SPARSITY_VERSION_BIT = 1 << 12

A_STAGE_BYTES = CTA_M * PACKED_BLOCK_K
B_STAGE_BYTES = CTA_N * PACKED_BLOCK_K
SFA_STAGE_BYTES = CTA_M * SF_K_PER_TILE
SFB_CTA_BYTES = CTA_N * SF_K_PER_TILE
SFB_STAGE_BYTES = CLUSTER_M * SFB_CTA_BYTES
# Transaction accounting follows the source: A/B/SFA are one-CTA transfers;
# each CTA's SFB slice is multicast to both CTAs.
TMA_STAGE_BYTES = (
    A_STAGE_BYTES + B_STAGE_BYTES + SFA_STAGE_BYTES + SFB_STAGE_BYTES
) * CLUSTER_M

SFA_TMEM_COLUMN = BLOCK_N
SFB_TMEM_COLUMN = SFA_TMEM_COLUMN + SF_K_PER_TILE
VEC_BYTES = 32
TMEM_BARRIER_ID = 1
TMEM_BARRIER_THREADS = (EPILOGUE_WARPS + 1) * WARP_SIZE

_DEFAULT_MNKL = (256, 256, 256, 1)
_DEFAULT_TOLERANCE = 1.0e-1


def _wait_mbarrier(mbar: cl.Pointer[Any], phase: cl.int32) -> None:
    while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10_000_000):
        pass


def _p3_to_u32(pointer: cl.Pointer[Any]) -> cl.uint32:
    return cl.bitcast(pointer, cl.uint32)


def _tmem_offset(
    base: cl.Pointer[Any], lane_offset: int = 0, column_offset: int = 0
) -> cl.Pointer[Any]:
    return cl.tcgen05_tmem_offset(
        base, lane_offset=lane_offset, column_offset=column_offset
    )


def _tmem_add_columns(
    base: cl.Pointer[Any], column_offset: int
) -> cl.Pointer[Any]:
    """Add columns to an already packed TMEM address, matching CuTe DSL."""
    pointer_dtype = cl.pointer_dtype(base.pointee_dtype, cl.MemorySpace.TENSOR)
    address = cl.bitcast(base, cl.uint32)
    return cl.bitcast(address + column_offset, pointer_dtype)


def _as_float32_vector_128(regs: cl.Vector[Any]) -> cl.Vector[cl.float32]:
    return cl.Vector(
        *tuple(
            cl.bitcast(regs[i], cl.float32)
            for i in cl.static_iter(range(128))
        ),
        dtype=cl.float32,
    )


def _to_float16_vector(
    values: cl.Vector[cl.float32], base: int, count: int
) -> cl.Vector[cl.float16]:
    return cl.Vector(
        *tuple(
            cl.float16(values[base + i])
            for i in cl.static_iter(range(count))
        ),
        dtype=cl.float16,
    )


@cl.kernel
def _kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    problem_m: ct.ScalarInt64,
    problem_n: ct.ScalarInt64,
    k_tiles: ct.ScalarInt64,
) -> None:
    """Warp-specialized CTA_2 NVFP4 block-scaled GEMM."""
    tid = cl.thread_index(0)
    # Broadcast the logical warp index from lane 0 so ptxas can keep the
    # warp-role control flow and TCGEN operands in uniform registers.
    warp = cl.shfl_sync(tid // WARP_SIZE, 0)
    block_m = cl.block_index(0)
    block_n = cl.block_index(1)
    batch = cl.block_index(2)
    rank = cl.block_in_cluster_index(0)
    is_leader = rank == 0

    coord_m = block_m * CTA_M
    coord_n_b = block_n * BLOCK_N + rank * CTA_N
    coord_n_c = block_n * BLOCK_N

    a_tmap = cl.tensor_map_tiled(
        a,
        (PACKED_BLOCK_K, CTA_M, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    b_tmap = cl.tensor_map_tiled(
        b,
        (PACKED_BLOCK_K, CTA_N, 1),
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    sfa_tmap = cl.tensor_map_tiled(sfa, (256, 4, 1, 1))
    sfb_tmap = cl.tensor_map_tiled(sfb, (256, 4, 1, 1))

    ab_full = cl.shared_array(
        AB_STAGES, cl.mbarrier, alignment=8, dynamic=True
    )
    ab_empty = cl.shared_array(
        AB_STAGES, cl.mbarrier, alignment=8, dynamic=True
    )
    acc_full = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    tmem_dealloc = cl.shared_array(1, cl.mbarrier, alignment=8, dynamic=True)
    tmem_storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
        dynamic=True,
    )
    a_smem = cl.shared_array(
        (AB_STAGES, A_STAGE_BYTES), cl.uint8, alignment=128, dynamic=True
    )
    b_smem = cl.shared_array(
        (AB_STAGES, B_STAGE_BYTES), cl.uint8, alignment=128, dynamic=True
    )
    sfa_smem = cl.shared_array(
        (AB_STAGES, SFA_STAGE_BYTES), cl.uint8, alignment=128, dynamic=True
    )
    sfb_smem = cl.shared_array(
        (AB_STAGES, SFB_STAGE_BYTES), cl.uint8, alignment=128, dynamic=True
    )

    if warp == 0 and cl.elect_sync():
        for stage in cl.static_iter(range(AB_STAGES)):
            cl.mbarrier_initialize(ab_full.get_element_pointer(stage), 1)
            cl.mbarrier_initialize(ab_empty.get_element_pointer(stage), 1)
        cl.mbarrier_initialize(acc_full.get_base_pointer(), 1)
        cl.mbarrier_initialize(tmem_dealloc.get_base_pointer(), WARP_SIZE)
    cl.fence_mbarrier_initialize()
    cl.barrier_arrive_cluster(
        aligned=False, memory_order=cl.MemoryOrder.RELAXED
    )
    cl.barrier_wait_cluster(aligned=False)

    # Both MMA warps participate in the CTA_2 allocation.  The named barrier
    # publishes the resulting local TMEM pointer to the epilogue warps without
    # making the independent TMA producer wait.
    if warp == MMA_WARP:
        cl.tcgen05_allocate(
            tmem_storage.get_base_pointer(),
            TMEM_COLUMNS,
            cta_group=cl.CTAGroup.CTA_2,
        )
        cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_2)
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )

    if warp == TMA_WARP:
        empty_phase = 1
        local_mask = cl.int16(1 << rank)
        sfb_mask = cl.int16(0b11)
        leader_barrier_base = cl.map_shared_to_leader_block(
            ab_full.get_base_pointer()
        )

        for k_tile in range(cl.int32(k_tiles)):
            stage = cl.int32(k_tile % AB_STAGES)
            if stage == 0 and k_tile != 0:
                empty_phase = empty_phase ^ 1

            empty_bar = ab_empty.get_element_pointer(stage)
            full_bar = ab_full.get_element_pointer(stage)
            if cl.elect_sync():
                _wait_mbarrier(empty_bar, empty_phase)
                if is_leader:
                    cl.mbarrier_arrive_expect_transaction(
                        full_bar,
                        TMA_STAGE_BYTES,
                        scope=cl.MbarrierScope.BLOCK,
                    )

            arrive_bar = leader_barrier_base + stage
            a_stage = cl.address_space_cast(
                a_smem.get_element_pointer((stage, 0)),
                cl.MemorySpace.SHARED_CLUSTER,
            )
            b_stage = cl.address_space_cast(
                b_smem.get_element_pointer((stage, 0)),
                cl.MemorySpace.SHARED_CLUSTER,
            )
            sfa_stage = cl.address_space_cast(
                sfa_smem.get_element_pointer((stage, 0)),
                cl.MemorySpace.SHARED_CLUSTER,
            )
            sfb_stage = cl.address_space_cast(
                sfb_smem.get_element_pointer(
                    (stage, rank * SFB_CTA_BYTES)
                ),
                cl.MemorySpace.SHARED_CLUSTER,
            )
            packed_k = cl.int32(k_tile * PACKED_BLOCK_K)
            sf_k_atom = cl.int32(k_tile * 4)

            if cl.elect_sync():
                cl.copy_async_bulk_tensor_global_to_shared(
                    a_tmap,
                    (packed_k, coord_m, batch),
                    a_stage,
                    arrive_bar,
                    multicast_mask=local_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    b_tmap,
                    (packed_k, coord_n_b, batch),
                    b_stage,
                    arrive_bar,
                    multicast_mask=local_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    sfa_tmap,
                    (0, sf_k_atom, block_m, batch),
                    sfa_stage,
                    arrive_bar,
                    multicast_mask=local_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )
                cl.copy_async_bulk_tensor_global_to_shared(
                    sfb_tmap,
                    (0, sf_k_atom, block_n * 2 + rank, batch),
                    sfb_stage,
                    arrive_bar,
                    multicast_mask=sfb_mask,
                    cta_group=cl.CTAGroup.CTA_2,
                )
    elif warp == MMA_WARP and is_leader:
        tmem_base = tmem_storage[0]
        instruction = cl.Tcgen05Mxf4InstructionDescriptor(
            a_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
            b_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
            scale_format=cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE4M3,
            n=BLOCK_N,
            m=BLOCK_M,
        ).encode() | NVFP4_SPARSITY_VERSION_BIT
        full_phase = 0
        accumulate = False

        for k_tile in range(cl.int32(k_tiles)):
            stage = cl.int32(k_tile % AB_STAGES)
            if stage == 0 and k_tile != 0:
                full_phase = full_phase ^ 1

            full_bar = ab_full.get_element_pointer(stage)
            empty_bar = ab_empty.get_element_pointer(stage)
            _wait_mbarrier(full_bar, full_phase)

            a_stage = a_smem.get_element_pointer((stage, 0))
            b_stage = b_smem.get_element_pointer((stage, 0))
            sfa_stage = sfa_smem.get_element_pointer((stage, 0))
            sfb_stage = sfb_smem.get_element_pointer((stage, 0))

            sfa_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=_p3_to_u32(sfa_stage),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_NONE,
            ).encode()
            sfb_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=_p3_to_u32(sfb_stage),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_NONE,
            ).encode()
            a_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=_p3_to_u32(a_stage),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()
            b_desc = cl.Tcgen05SharedMemoryDescriptor(
                matrix_start_address=_p3_to_u32(b_stage),
                leading_dimension_byte_offset=16,
                stride_dimension_byte_offset=8 * 128,
                swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
            ).encode()

            # Pack the lane/column fields once for each scale region. CuTe DSL
            # reloads the allocated address here, then advances the packed
            # addresses by constant column offsets inside the K-tile loop.
            scale_tmem_base = tmem_storage[0]
            sfa_tmem_base = _tmem_offset(
                scale_tmem_base, column_offset=SFA_TMEM_COLUMN
            )
            sfb_tmem_base = _tmem_offset(
                scale_tmem_base, column_offset=SFB_TMEM_COLUMN
            )

            for copy_idx in cl.static_iter(range(4)):
                scale_a = _tmem_add_columns(sfa_tmem_base, copy_idx * 4)
                if cl.elect_sync():
                    cl.tcgen05_copy(
                        scale_a,
                        cl.int64(sfa_desc + 32 * copy_idx),
                        shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                        cta_group=cl.CTAGroup.CTA_2,
                        multicast=cl.Tcgen05CopyMulticast.WARPX4,
                    )

            for copy_idx in cl.static_iter(range(8)):
                scale_b = _tmem_add_columns(sfb_tmem_base, copy_idx * 4)
                smem_increment = 32 * (copy_idx // 2) + 128 * (copy_idx % 2)
                if cl.elect_sync():
                    cl.tcgen05_copy(
                        scale_b,
                        cl.int64(sfb_desc + smem_increment),
                        shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                        cta_group=cl.CTAGroup.CTA_2,
                        multicast=cl.Tcgen05CopyMulticast.WARPX4,
                    )

            for k_block in cl.static_iter(range(BLOCK_K // MMA_K)):
                scale_a = _tmem_add_columns(sfa_tmem_base, k_block * 4)
                scale_b = _tmem_add_columns(sfb_tmem_base, k_block * 8)
                if cl.elect_sync():
                    cl.tcgen05_mma_block_scale(
                        cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
                        tmem_base,
                        a_desc + 2 * k_block,
                        b_desc + 2 * k_block,
                        instruction,
                        scale_a,
                        scale_b,
                        accumulate=accumulate,
                        cta_group=cl.CTAGroup.CTA_2,
                        scale_vector_size=cl.Tcgen05MMAScaleVectorSize.BLOCK_16,
                    )
                accumulate = True

            if cl.elect_sync():
                cl.tcgen05_commit(
                    empty_bar,
                    multicast_mask=0b11,
                    cta_group=cl.CTAGroup.CTA_2,
                )
        if cl.elect_sync():
            cl.tcgen05_commit(
                acc_full.get_base_pointer(),
                multicast_mask=0b11,
                cta_group=cl.CTAGroup.CTA_2,
            )

    elif warp < EPILOGUE_WARPS:
        cl.barrier_sync_block(
            number_of_threads=TMEM_BARRIER_THREADS,
            barrier_id=TMEM_BARRIER_ID,
        )
        tmem_base = tmem_storage[0]
        _wait_mbarrier(acc_full.get_base_pointer(), 0)
        row = coord_m + tid
        vsize = VEC_BYTES // 2
        # c.shape metadata is i32; explicit i64 dimensions let NVVM fuse the
        # linear output index into the same 64-bit MAD form used by CuTe DSL.
        output_base = (
            c.get_base_pointer()
            + row * problem_n
            + coord_n_c
            + batch * problem_m * problem_n
        )

        for half in cl.static_iter(range(BLOCK_N // 128)):
            column = half * 128
            tmem = _tmem_offset(
                tmem_base,
                lane_offset=warp * WARP_SIZE,
                column_offset=column,
            )
            regs = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                tmem,
                count=128,
            )
            accumulators = _as_float32_vector_128(regs)

            for vector_idx in cl.static_iter(range(128 // vsize)):
                values = _to_float16_vector(
                    accumulators, vector_idx * vsize, vsize
                )
                (output_base + column + vector_idx * vsize).store(
                    values, alignment=VEC_BYTES
                )

    # Match the source peer-CTA mbarrier handshake before releasing the
    # collective TMEM allocation.
    cl.barrier_sync_block()
    if warp == 0:
        peer_rank = rank ^ 1
        peer_mbar = cl.map_shared_to_cluster(
            tmem_dealloc.get_base_pointer(), peer_rank
        )
        cl.mbarrier_arrive(peer_mbar, scope=cl.MbarrierScope.BLOCK)
        _wait_mbarrier(tmem_dealloc.get_base_pointer(), 0)
        cl.tcgen05_deallocate(
            tmem_storage[0], TMEM_COLUMNS, cta_group=cl.CTAGroup.CTA_2
        )


def _parse_mnkl(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected exactly four MNKL values.")
    return values


def _validate_mnkl(mnkl: tuple[int, int, int, int]) -> None:
    if len(mnkl) != 4:
        raise ValueError("MNKL must contain exactly four values")
    m, n, k, batch = mnkl
    if min(m, n, k, batch) <= 0:
        raise ValueError("MNKL values must be positive")
    if m % BLOCK_M:
        raise ValueError(f"M must be a multiple of {BLOCK_M} (got {m})")
    if n % BLOCK_N:
        raise ValueError(f"N must be a multiple of {BLOCK_N} (got {n})")
    if k % BLOCK_K:
        raise ValueError(f"K must be a multiple of {BLOCK_K} (got {k})")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """Convert natural (MN, K/16) E4M3 scales to the tcgen05 layout."""
    rows, columns = scale.shape
    if rows % 128 or columns % 4:
        raise ValueError("scale rows and columns must be multiples of 128 and 4")
    row_blocks = rows // 128
    column_blocks = columns // 4
    blocks = scale.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )


def _make_scale_tensors(
    batch: int, rows: int, sf_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = (
        torch.randint(1, 3, (batch, rows, sf_k), dtype=torch.int8)
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    blocked = torch.stack(
        [to_blocked(reference[:, :, index]) for index in range(batch)]
    ).contiguous()
    rest_rows = rows // 128
    rest_k = sf_k // 4
    tma = (
        blocked.view(torch.uint16)
        .reshape(batch, rest_rows, rest_k, 256)
        .permute(3, 2, 1, 0)
        .cuda()
    )
    return reference, tma


def prepare_tensors(m: int, n: int, k: int, batch: int = 1, **_):
    _validate_mnkl((m, n, k, batch))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build does not provide float4_e2m1fn_x2")

    torch.manual_seed(1111)
    a_storage = torch.randint(
        0, 2, (batch, m, k // 2), dtype=torch.uint8, device="cuda"
    )
    b_storage = torch.randint(
        0, 2, (batch, n, k // 2), dtype=torch.uint8, device="cuda"
    )
    a = a_storage.permute(2, 1, 0)
    b = b_storage.permute(2, 1, 0)
    a_ref = a_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    b_ref = b_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)

    sf_k = _ceil_div(k, SF_VECTOR_SIZE)
    sfa_ref, sfa = _make_scale_tensors(batch, m, sf_k)
    sfb_ref, sfb = _make_scale_tensors(batch, n, sf_k)
    c = torch.empty((batch, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )
    return {
        "a": a,
        "b": b,
        "a_ref": a_ref,
        "b_ref": b_ref,
        "sfa": sfa,
        "sfb": sfb,
        "sfa_ref": sfa_ref,
        "sfb_ref": sfb_ref,
        "c": c,
    }


def run(tensors: dict[str, torch.Tensor], stream=None) -> None:
    a, b = tensors["a"], tensors["b"]
    sfa, sfb, c = tensors["sfa"], tensors["sfb"], tensors["c"]
    packed_k, m, batch = a.shape
    if b.shape[0] != packed_k or b.shape[2] != batch:
        raise ValueError("B must have shape (K/2, N, L)")
    n = b.shape[1]
    k = packed_k * 2
    if c.shape != (m, n, batch):
        raise ValueError("C must have shape (M, N, L)")
    if c.stride() != (n, 1, m * n):
        raise ValueError("C must use the source GEMM's contiguous MNL layout")
    _validate_mnkl((m, n, k, batch))

    expected_sfa = (256, k // 64, m // 128, batch)
    expected_sfb = (256, k // 64, n // 128, batch)
    if sfa.shape != expected_sfa or sfb.shape != expected_sfb:
        raise ValueError(
            f"invalid scale layouts: expected {expected_sfa} and {expected_sfb}, "
            f"got {tuple(sfa.shape)} and {tuple(sfb.shape)}"
        )

    cuda_stream = torch.cuda.current_stream() if stream is None else stream
    cl.launch(
        cuda_stream,
        (m // CTA_M, n // BLOCK_N, batch),
        (BLOCK_THREADS, 1, 1),
        _kernel,
        (a, b, sfa, sfb, c, m, n, k // BLOCK_K),
        block_in_cluster_count=(CLUSTER_M, 1, 1),
    )


def _unpack_e2m1_bytes_to_float(fp4_bytes: torch.Tensor) -> torch.Tensor:
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    low = fp4_bytes & 0x0F
    high = (fp4_bytes >> 4) & 0x0F
    low_values = lookup[(low & 0x7).long()]
    high_values = lookup[(high & 0x7).long()]
    low_values = torch.where((low & 0x8) != 0, -low_values, low_values)
    high_values = torch.where((high & 0x8) != 0, -high_values, high_values)
    values = torch.empty(
        (*fp4_bytes.shape[:-1], fp4_bytes.shape[-1] * 2),
        dtype=torch.float32,
        device=fp4_bytes.device,
    )
    values[..., 0::2] = low_values
    values[..., 1::2] = high_values
    return values


def verify_output(
    tensors: dict[str, torch.Tensor], tolerance: float = _DEFAULT_TOLERANCE, **_
) -> None:
    a_ref, b_ref, c = tensors["a_ref"], tensors["b_ref"], tensors["c"]
    sfa_ref, sfb_ref = tensors["sfa_ref"], tensors["sfb_ref"]
    m, n, batch = c.shape
    reference = torch.empty_like(c)
    for batch_idx in range(batch):
        a_values = _unpack_e2m1_bytes_to_float(
            a_ref[:, :, batch_idx].view(torch.uint8)
        )
        b_values = _unpack_e2m1_bytes_to_float(
            b_ref[:, :, batch_idx].view(torch.uint8)
        )
        scale_a = sfa_ref[:, :, batch_idx].to(
            device=a_values.device, dtype=torch.float32
        ).repeat_interleave(SF_VECTOR_SIZE, dim=1)
        scale_b = sfb_ref[:, :, batch_idx].to(
            device=b_values.device, dtype=torch.float32
        ).repeat_interleave(SF_VECTOR_SIZE, dim=1)
        a_values *= scale_a
        b_values *= scale_b
        reference[:, :, batch_idx] = a_values @ b_values.T.contiguous()
    torch.testing.assert_close(c, reference, atol=tolerance, rtol=1.0e-2)


def verify(
    mnkl: tuple[int, int, int, int] = _DEFAULT_MNKL,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> None:
    _validate_mnkl(mnkl)
    m, n, k, batch = mnkl
    tensors = prepare_tensors(m=m, n=n, k=k, batch=batch)
    run(tensors)
    torch.cuda.synchronize()
    print(f"Run kernel (mnkl={mnkl}) OK", flush=True)
    verify_output(tensors, tolerance=tolerance)
    print(f"verify (mnkl={mnkl}): PASS")


def run_nvfp4_gemm(mnkl: tuple[int, int, int, int], tolerance: float) -> None:
    verify(mnkl, tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CUDA Lang warp-specialized CTA_2 NVFP4 GEMM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mnkl", type=_parse_mnkl, default=_DEFAULT_MNKL, help="M,N,K,L dimensions"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help="Numerical validation tolerance",
    )
    args = parser.parse_args()
    verify(args.mnkl, args.tolerance)
