# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Task-scheduled CTA_2 NVFP4 block-scaled GEMM for Blackwell.

The default fused path uses four independent five-stage TMA-to-UMMA pipelines
to feed one leader-CTA MMA warp. A one-stage UMMA-to-async pipeline feeds four
epilogue warps per CTA, and two padding warps complete the second
register-allocation warp group.

Each two-CTA cluster computes a 256x256x256 collective tile. CTA-local A, B,
and SFA tiles use self multicast masks; SFB is multicast to both CTAs. Scale
factors are copied from SMEM to TMEM immediately before four block-scaled MMA
instructions per K tile.

The program builder supports either one combined TMA warp or two TMA warps
split between the A/SFA and B/SFB operands. Both variants use direct CTA_2 tile
mapping and fuse the scale-factor copies into the MMA warp.
"""

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts
import cuda.tile as ct
import torch


WARP_SIZE = 32
STORE_WARPS = 4
MMA_WARPS = 1
LOAD_WARPS = 1
STORE_TASK_WARP_IDX = 0
MMA_TASK_WARP_IDX = 4
LOAD_TASK_WARP_IDX = 5
LOAD_A_TASK_WARP_IDX = 5
LOAD_B_TASK_WARP_IDX = 6

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
ACC_STAGES = 1
NVFP4_SPARSITY_VERSION_BIT = 1 << 12

A_STAGE_BYTES = CTA_M * PACKED_BLOCK_K
B_STAGE_BYTES = CTA_N * PACKED_BLOCK_K
SFA_STAGE_BYTES = CTA_M * SF_K_PER_TILE
SFB_CTA_BYTES = CTA_N * SF_K_PER_TILE
SFB_STAGE_BYTES = CLUSTER_M * SFB_CTA_BYTES

A_SMEM_OFFSET_BYTES = 0
B_SMEM_OFFSET_BYTES = A_SMEM_OFFSET_BYTES + AB_STAGES * A_STAGE_BYTES
SFA_SMEM_OFFSET_BYTES = B_SMEM_OFFSET_BYTES + AB_STAGES * B_STAGE_BYTES
SFB_SMEM_OFFSET_BYTES = SFA_SMEM_OFFSET_BYTES + AB_STAGES * SFA_STAGE_BYTES
TMEM_PTR_SMEM_OFFSET_BYTES = SFB_SMEM_OFFSET_BYTES + AB_STAGES * SFB_STAGE_BYTES

ACC_TMEM_COLUMNS = BLOCK_N
SFA_TMEM_COLUMNS = 16
SFB_TMEM_COLUMNS = 32
SFA_TMEM_COLUMN = ACC_TMEM_COLUMNS
# The fused scale-factor values occupy fixed columns after the accumulator.
SFB_TMEM_COLUMN = SFA_TMEM_COLUMN + SFA_TMEM_COLUMNS
TMEM_COLUMNS = 512
TMEM_SYNC_BARRIER = 1

VEC_BYTES = 32

_DEFAULT_MNKL = (256, 256, 256, 1)
_DEFAULT_TOLERANCE = 1.0e-1


@dataclass(frozen=True)
class GemmConfig:
    """Host specialization selecting one or two fused TMA load warps."""

    use_two_tma_warps: bool = False

    @property
    def active_warps(self):
        return STORE_WARPS + MMA_WARPS + (2 if self.use_two_tma_warps else 1)

    @property
    def block_warps(self):
        return ((self.active_warps + 3) // 4) * 4

    @property
    def block_threads(self):
        return self.block_warps * WARP_SIZE


def _block_tile():
    return (
        cl.block_index(0),
        cl.block_index(1),
        cl.block_index(2),
    )


def _smem_array(stage_info, offset, stage_bytes):
    base = stage_info.context.smem_base.get_base_pointer()
    return cl.reinterpret_pointer_as_array(
        base + offset,
        cl.uint8,
        (AB_STAGES, stage_bytes),
    )


def _mma_descriptor(smem_values, stage):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=smem_values.get_element_pointer((stage, 0)),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=8 * 128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )


def _scale_descriptor(smem_values, stage):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=smem_values.get_element_pointer((stage, 0)),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_NONE,
        ).encode()
    )


def _to_float16_vector(values, base, count):
    return values[base:base + count].astype(cl.float16)


# -----------------------------------------------------------------------------
# Host resource model and device work methods
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class GmemAResource(ts.MemoryResource):
    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info):
        tile_m, _, tile_l = _block_tile()
        return (
            stage_info.loop_offset * PACKED_BLOCK_K,
            tile_m * CTA_M,
            tile_l,
        )


@dataclass(kw_only=True, eq=False)
class GmemBResource(ts.MemoryResource):
    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info):
        rank = cl.block_in_cluster_index(0)
        _, tile_n, tile_l = _block_tile()
        return (
            stage_info.loop_offset * PACKED_BLOCK_K,
            tile_n * BLOCK_N + rank * CTA_N,
            tile_l,
        )


@dataclass(kw_only=True, eq=False)
class GmemSfAResource(ts.MemoryResource):
    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info):
        tile_m, _, tile_l = _block_tile()
        return (
            stage_info.loop_offset * 4,
            tile_m,
            tile_l,
        )


@dataclass(kw_only=True, eq=False)
class GmemSfBResource(ts.MemoryResource):
    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info):
        rank = cl.block_in_cluster_index(0)
        _, tile_n, tile_l = _block_tile()
        return (
            stage_info.loop_offset * 4,
            tile_n * CLUSTER_M + rank,
            tile_l,
        )


@dataclass(kw_only=True, eq=False)
class SmemAResource(ts.MemoryResource):
    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptors(stage_info):
        return _smem_array(stage_info, A_SMEM_OFFSET_BYTES, A_STAGE_BYTES)

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info):
        return (
            stage_info.context.cluster_smem_base
            + cl.uint32(A_SMEM_OFFSET_BYTES)
        )

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, cluster_smem_address, coord_k, coord_m, coord_l):
        rank = cl.block_in_cluster_index(0)
        destination_address = (
            cluster_smem_address
            + cl.uint32(stage_info.stage_idx) * cl.uint32(A_STAGE_BYTES)
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                stage_info.context.tasks_inputs.a_map,
                (coord_k, coord_m, coord_l),
                destination,
                stage_info.barrier,
                multicast_mask=cl.int16(1 << rank),
                cta_group=cl.CTAGroup.CTA_2,
            )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_descriptor(stage_info, smem_values):
        return _mma_descriptor(smem_values, stage_info.stage_idx)


@dataclass(kw_only=True, eq=False)
class SmemBResource(ts.MemoryResource):
    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptors(stage_info):
        return _smem_array(stage_info, B_SMEM_OFFSET_BYTES, B_STAGE_BYTES)

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info):
        return (
            stage_info.context.cluster_smem_base
            + cl.uint32(B_SMEM_OFFSET_BYTES)
        )

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, cluster_smem_address, coord_k, coord_n, coord_l):
        rank = cl.block_in_cluster_index(0)
        destination_address = (
            cluster_smem_address
            + cl.uint32(stage_info.stage_idx) * cl.uint32(B_STAGE_BYTES)
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                stage_info.context.tasks_inputs.b_map,
                (coord_k, coord_n, coord_l),
                destination,
                stage_info.barrier,
                multicast_mask=cl.int16(1 << rank),
                cta_group=cl.CTAGroup.CTA_2,
            )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_descriptor(stage_info, smem_values):
        return _mma_descriptor(smem_values, stage_info.stage_idx)


@dataclass(kw_only=True, eq=False)
class SmemSfAResource(ts.MemoryResource):
    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptors(stage_info):
        return _smem_array(stage_info, SFA_SMEM_OFFSET_BYTES, SFA_STAGE_BYTES)

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info):
        return (
            stage_info.context.cluster_smem_base
            + cl.uint32(SFA_SMEM_OFFSET_BYTES)
        )

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, cluster_smem_address, coord_k, coord_m, coord_l):
        rank = cl.block_in_cluster_index(0)
        destination_address = (
            cluster_smem_address
            + cl.uint32(stage_info.stage_idx) * cl.uint32(SFA_STAGE_BYTES)
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                stage_info.context.tasks_inputs.sfa_map,
                (0, coord_k, coord_m, coord_l),
                destination,
                stage_info.barrier,
                multicast_mask=cl.int16(1 << rank),
                cta_group=cl.CTAGroup.CTA_2,
            )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_descriptor(stage_info, smem_values):
        return _scale_descriptor(smem_values, stage_info.stage_idx)


@dataclass(kw_only=True, eq=False)
class SmemSfBResource(ts.MemoryResource):
    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptors(stage_info):
        return _smem_array(stage_info, SFB_SMEM_OFFSET_BYTES, SFB_STAGE_BYTES)

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info):
        return (
            stage_info.context.cluster_smem_base
            + cl.uint32(SFB_SMEM_OFFSET_BYTES)
        )

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, cluster_smem_address, coord_k, coord_n, coord_l):
        rank = cl.block_in_cluster_index(0)
        destination_address = (
            cluster_smem_address
            + cl.uint32(stage_info.stage_idx) * cl.uint32(SFB_STAGE_BYTES)
            + cl.uint32(rank) * cl.uint32(SFB_CTA_BYTES)
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                stage_info.context.tasks_inputs.sfb_map,
                (0, coord_k, coord_n, coord_l),
                destination,
                stage_info.barrier,
                multicast_mask=cl.int16(0b11),
                cta_group=cl.CTAGroup.CTA_2,
            )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_descriptor(stage_info, smem_values):
        return _scale_descriptor(smem_values, stage_info.stage_idx)


@dataclass(kw_only=True, eq=False)
class TmemSfAResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_copy_state(stage_info):
        stage_info.context.tasks_inputs.tmem

    @ts.producer_work
    @staticmethod
    def copy_fused(stage_info, descriptor):
        base = cl.tcgen05_tmem_offset(
            stage_info.context.tasks_inputs.tmem,
            column_offset=SFA_TMEM_COLUMN,
        )
        for copy_idx in cl.static_iter(range(4)):
            if cl.elect_sync():
                cl.tcgen05_copy(
                    cl.tcgen05_tmem_offset(base, column_offset=copy_idx * 4),
                    cl.int64(descriptor + 32 * copy_idx),
                    shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                    cta_group=cl.CTAGroup.CTA_2,
                    multicast=cl.Tcgen05CopyMulticast.WARPX4,
                )
        cl.tcgen05_wait_store()


@dataclass(kw_only=True, eq=False)
class TmemSfBResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_copy_state(stage_info):
        stage_info.context.tasks_inputs.tmem

    @ts.producer_work
    @staticmethod
    def copy_fused(stage_info, descriptor):
        base = cl.tcgen05_tmem_offset(
            stage_info.context.tasks_inputs.tmem,
            column_offset=SFB_TMEM_COLUMN,
        )
        for copy_idx in cl.static_iter(range(8)):
            smem_increment = 32 * (copy_idx // 2) + 128 * (copy_idx % 2)
            if cl.elect_sync():
                cl.tcgen05_copy(
                    cl.tcgen05_tmem_offset(base, column_offset=copy_idx * 4),
                    cl.int64(descriptor + smem_increment),
                    shape=cl.Tcgen05CopyShape.SHAPE_32x128b,
                    cta_group=cl.CTAGroup.CTA_2,
                    multicast=cl.Tcgen05CopyMulticast.WARPX4,
                )
        cl.tcgen05_wait_store()


@dataclass(kw_only=True, eq=False)
class TmemCResource(ts.MemoryResource):
    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_accumulator_state(stage_info):
        stage_info.context.tasks_inputs.tmem
        return (
            cl.Tcgen05Mxf4InstructionDescriptor(
                a_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
                b_type=cl.Tcgen05Mxf4InstructionDescriptor.Type.E2M1,
                scale_format=cl.Tcgen05Mxf4InstructionDescriptor.ScaleFormat.UE4M3,
                n=BLOCK_N,
                m=BLOCK_M,
            ).encode()
            | NVFP4_SPARSITY_VERSION_BIT
        )

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_store_state(stage_info):
        stage_info.context.tasks_inputs.tmem

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_work_tile_state(stage_info):
        stage_info.context.tasks_inputs.tmem

    @ts.producer_work
    @staticmethod
    def mma(
        stage_info,
        a_descriptor,
        b_descriptor,
        idesc,
    ):
        tmem = stage_info.context.tasks_inputs.tmem
        scale_a_base = cl.tcgen05_tmem_offset(
            tmem,
            column_offset=SFA_TMEM_COLUMN,
        )
        scale_b_base = cl.tcgen05_tmem_offset(
            tmem,
            column_offset=SFB_TMEM_COLUMN,
        )
        for k_block in cl.static_iter(range(BLOCK_K // MMA_K)):
            if cl.elect_sync():
                cl.tcgen05_mma_block_scale(
                    cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
                    tmem,
                    cl.int64(a_descriptor + 2 * k_block),
                    cl.int64(b_descriptor + 2 * k_block),
                    cl.int32(idesc),
                    cl.tcgen05_tmem_offset(
                        scale_a_base,
                        column_offset=k_block * 4,
                    ),
                    cl.tcgen05_tmem_offset(
                        scale_b_base,
                        column_offset=k_block * 8,
                    ),
                    accumulate=stage_info.count != 0 or k_block != 0,
                    cta_group=cl.CTAGroup.CTA_2,
                    scale_vector_size=cl.Tcgen05MMAScaleVectorSize.BLOCK_16,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def load_subtile(stage_info, subtile_idx):
        tmem_subtile = cl.tcgen05_tmem_offset(
            stage_info.context.tasks_inputs.tmem,
            lane_offset=stage_info.context.warp_index * WARP_SIZE,
            column_offset=subtile_idx * 128,
        )
        t2r_rmem = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem_subtile,
            element_count=128,
            dtype=cl.float32,
        )
        cl.tcgen05_wait_load()
        return t2r_rmem


@dataclass(kw_only=True, eq=False)
class GmemDResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def store(stage_info, t2r_rmem, subtile_idx):
        values = stage_info.context.tasks_inputs
        tile_m, tile_n, tile_l = _block_tile()
        row = tile_m * CTA_M + cl.thread_index(0)
        column = tile_n * BLOCK_N + subtile_idx * 128
        output_base = (
            values.output.get_base_pointer()
            + row * values.problem_n
            + column
            + tile_l * values.problem_m * values.problem_n
        )
        vector_size = VEC_BYTES // 2
        for vector_idx in cl.static_iter(range(128 // vector_size)):
            converted = _to_float16_vector(
                t2r_rmem,
                vector_idx * vector_size,
                vector_size,
            )
            (output_base + vector_idx * vector_size).store(
                converted,
                alignment=VEC_BYTES,
            )


# -----------------------------------------------------------------------------
# Immutable schedule trees
# -----------------------------------------------------------------------------


@ts.schedule
def load_schedule(
    stage_info,
    gmem_a,
    gmem_b,
    gmem_sfa,
    gmem_sfb,
    smem_a,
    smem_b,
    smem_sfa,
    smem_sfb,
):
    a_smem_address = smem_a.init_load_state()
    b_smem_address = smem_b.init_load_state()
    sfa_smem_address = smem_sfa.init_load_state()
    sfb_smem_address = smem_sfb.init_load_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        a_coord_k, a_coord_m, a_coord_l = gmem_a.compute_coords()
        b_coord_k, b_coord_n, b_coord_l = gmem_b.compute_coords()
        sfa_coord_k, sfa_coord_m, sfa_coord_l = gmem_sfa.compute_coords()
        sfb_coord_k, sfb_coord_n, sfb_coord_l = gmem_sfb.compute_coords()
        smem_a.try_acquire()
        smem_b.try_acquire()
        smem_sfa.try_acquire()
        smem_sfb.try_acquire()
        smem_a.acquire()
        smem_b.acquire()
        smem_sfa.acquire()
        smem_sfb.acquire()
        smem_a.tma_load(a_smem_address, a_coord_k, a_coord_m, a_coord_l)
        smem_b.tma_load(b_smem_address, b_coord_k, b_coord_n, b_coord_l)
        smem_sfa.tma_load(
            sfa_smem_address, sfa_coord_k, sfa_coord_m, sfa_coord_l
        )
        smem_sfb.tma_load(
            sfb_smem_address, sfb_coord_k, sfb_coord_n, sfb_coord_l
        )
        smem_a.commit()
        smem_b.commit()
        smem_sfa.commit()
        smem_sfb.commit()


@ts.schedule
def load_a_sf_schedule(stage_info, gmem_a, gmem_sfa, smem_a, smem_sfa):
    a_smem_address = smem_a.init_load_state()
    sfa_smem_address = smem_sfa.init_load_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        a_coord_k, a_coord_m, a_coord_l = gmem_a.compute_coords()
        sfa_coord_k, sfa_coord_m, sfa_coord_l = gmem_sfa.compute_coords()
        smem_a.try_acquire()
        smem_sfa.try_acquire()
        smem_a.acquire()
        smem_sfa.acquire()
        smem_a.tma_load(a_smem_address, a_coord_k, a_coord_m, a_coord_l)
        smem_sfa.tma_load(
            sfa_smem_address, sfa_coord_k, sfa_coord_m, sfa_coord_l
        )
        smem_a.commit()
        smem_sfa.commit()


@ts.schedule
def load_b_sf_schedule(stage_info, gmem_b, gmem_sfb, smem_b, smem_sfb):
    b_smem_address = smem_b.init_load_state()
    sfb_smem_address = smem_sfb.init_load_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        b_coord_k, b_coord_n, b_coord_l = gmem_b.compute_coords()
        sfb_coord_k, sfb_coord_n, sfb_coord_l = gmem_sfb.compute_coords()
        smem_b.try_acquire()
        smem_sfb.try_acquire()
        smem_b.acquire()
        smem_sfb.acquire()
        smem_b.tma_load(b_smem_address, b_coord_k, b_coord_n, b_coord_l)
        smem_sfb.tma_load(
            sfb_smem_address, sfb_coord_k, sfb_coord_n, sfb_coord_l
        )
        smem_b.commit()
        smem_sfb.commit()


@ts.schedule
def mma_schedule(
    stage_info,
    smem_a,
    smem_b,
    smem_sfa,
    smem_sfb,
    tmem_sfa,
    tmem_sfb,
    tmem_c,
):
    a_smem = smem_a.init_descriptors()
    b_smem = smem_b.init_descriptors()
    sfa_smem = smem_sfa.init_descriptors()
    sfb_smem = smem_sfb.init_descriptors()
    tmem_sfa.init_copy_state()
    tmem_sfb.init_copy_state()
    idesc = tmem_c.init_accumulator_state()
    tmem_c.init_work_tile_state()
    tmem_c.try_acquire()
    tmem_c.acquire()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        smem_a.try_wait()
        smem_b.try_wait()
        smem_sfa.try_wait()
        smem_sfb.try_wait()
        smem_a.wait()
        smem_b.wait()
        smem_sfa.wait()
        smem_sfb.wait()
        a_descriptor = smem_a.build_descriptor(a_smem)
        b_descriptor = smem_b.build_descriptor(b_smem)
        sfa_descriptor = smem_sfa.build_descriptor(sfa_smem)
        sfb_descriptor = smem_sfb.build_descriptor(sfb_smem)
        tmem_sfa.copy_fused(sfa_descriptor)
        tmem_sfb.copy_fused(sfb_descriptor)
        tmem_c.mma(a_descriptor, b_descriptor, idesc)
        smem_a.release()
        smem_b.release()
        smem_sfa.release()
        smem_sfb.release()
    tmem_c.commit()


@ts.schedule
def store_schedule(stage_info, tmem_c, gmem_d):
    tmem_c.init_store_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        pass
    tmem_c.try_wait()
    tmem_c.wait()
    for subtile_idx in range(BLOCK_N // 128):
        t2r_rmem = tmem_c.load_subtile(subtile_idx=subtile_idx)
        gmem_d.store(t2r_rmem, subtile_idx=subtile_idx)
    tmem_c.release()


@ts.schedule
def padding_schedule(stage_info):
    with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
        pass


# -----------------------------------------------------------------------------
# Device inputs and host program construction
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelInputs:
    a_map: object
    b_map: object
    sfa_map: object
    sfb_map: object
    tmem: object
    output: object
    problem_m: object
    problem_n: object
    num_k_tiles: object


@dataclass(frozen=True)
class GemmProgram:
    manager: object
    device_manager: object
    kernel: object
    smem_allocator: object
    tmem_allocator: object
    barrier_allocator: object
    config: GemmConfig
    block_threads: int


def _tma_umma_config(stage_bytes):
    return ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=AB_STAGES,
        num_bytes=stage_bytes * CLUSTER_M,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(1),
        cta_layout_vmnk=(2, 1, 1, 1),
        consumer_signaling_threads=ts.SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=stage_bytes,
    )


def make_gemm_program(use_two_tma_warps=False):
    """Construct and freeze the one- or two-TMA-warp specialization."""
    config = GemmConfig(use_two_tma_warps=use_two_tma_warps)
    a_allocation = ts.SmemAllocation("smem_a", AB_STAGES * A_STAGE_BYTES, alignment=128)
    b_allocation = ts.SmemAllocation("smem_b", AB_STAGES * B_STAGE_BYTES, alignment=128)
    sfa_allocation = ts.SmemAllocation(
        "smem_sfa", AB_STAGES * SFA_STAGE_BYTES, alignment=128
    )
    sfb_allocation = ts.SmemAllocation(
        "smem_sfb", AB_STAGES * SFB_STAGE_BYTES, alignment=128
    )
    tmem_ptr_allocation = ts.SmemAllocation(
        "tmem_ptr",
        4,
        alignment=4,
    )

    gmem_a = GmemAResource(name="GmemA")
    gmem_b = GmemBResource(name="GmemB")
    gmem_sfa = GmemSfAResource(name="GmemSfA")
    gmem_sfb = GmemSfBResource(name="GmemSfB")
    smem_a = SmemAResource(
        name="SmemA",
        pipeline_config=_tma_umma_config(A_STAGE_BYTES),
        smem_requirements=[a_allocation],
    )
    smem_b = SmemBResource(
        name="SmemB",
        pipeline_config=_tma_umma_config(B_STAGE_BYTES),
        smem_requirements=[b_allocation],
    )
    smem_sfa = SmemSfAResource(
        name="SmemSfA",
        pipeline_config=_tma_umma_config(SFA_STAGE_BYTES),
        smem_requirements=[sfa_allocation],
    )
    smem_sfb = SmemSfBResource(
        name="SmemSfB",
        pipeline_config=_tma_umma_config(SFB_STAGE_BYTES),
        smem_requirements=[sfb_allocation],
    )
    tmem_sfa = TmemSfAResource(
        name="TmemSfA",
        pipeline_config=None,
        tmem_requirements=[ts.TmemAllocation("tmem_sfa", SFA_TMEM_COLUMNS)],
    )
    tmem_sfb = TmemSfBResource(
        name="TmemSfB",
        pipeline_config=None,
        tmem_requirements=[ts.TmemAllocation("tmem_sfb", SFB_TMEM_COLUMNS)],
    )
    tmem_config = ts.PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=ACC_STAGES,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(STORE_WARPS * WARP_SIZE * CLUSTER_M),
        cta_layout_vmnk=(2, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.CtaLeader,
    )
    tmem_c = TmemCResource(
        name="TmemC",
        pipeline_config=tmem_config,
        tmem_requirements=[ts.TmemAllocation("tmem_acc", ACC_TMEM_COLUMNS)],
    )
    gmem_d = GmemDResource(name="GmemD")

    load_tasks = []
    if not config.use_two_tma_warps:
        load_tasks.append(
            ts.Task(
                LOAD_TASK_WARP_IDX,
                LOAD_WARPS,
                schedule=load_schedule(
                    gmem_a,
                    gmem_b,
                    gmem_sfa,
                    gmem_sfb,
                    smem_a,
                    smem_b,
                    smem_sfa,
                    smem_sfb,
                ),
                num_registers=40,
                name="LoadTask",
            )
        )
    else:
        load_tasks.extend(
            (
                ts.Task(
                    LOAD_A_TASK_WARP_IDX,
                    1,
                    schedule=load_a_sf_schedule(
                        gmem_a, gmem_sfa, smem_a, smem_sfa
                    ),
                    num_registers=40,
                    name="LoadATask",
                ),
                ts.Task(
                    LOAD_B_TASK_WARP_IDX,
                    1,
                    schedule=load_b_sf_schedule(
                        gmem_b, gmem_sfb, smem_b, smem_sfb
                    ),
                    num_registers=40,
                    name="LoadBTask",
                ),
            )
        )
    mma_task = ts.Task(
        MMA_TASK_WARP_IDX,
        MMA_WARPS,
        schedule=mma_schedule(
            smem_a,
            smem_b,
            smem_sfa,
            smem_sfb,
            tmem_sfa,
            tmem_sfb,
            tmem_c,
        ),
        num_registers=40,
        name="MmaTask",
        run_only_on_cta_id=0,
    )
    store_task = ts.Task(
        STORE_TASK_WARP_IDX,
        STORE_WARPS,
        schedule=store_schedule(tmem_c, gmem_d),
        num_registers=160,
        name="StoreTask",
    )
    max_warp_end = max(task.warp_end for task in [mma_task, store_task, *load_tasks])
    padding_warps = config.block_warps - max_warp_end
    padding_task = None
    if padding_warps:
        padding_task = ts.Task(
            max_warp_end,
            padding_warps,
            schedule=padding_schedule(),
            num_registers=40,
            name="PaddingTask",
        )

    smem_allocator = ts.SmemAllocator(default_add_barriers=False)
    for resource in (smem_a, smem_b, smem_sfa, smem_sfb):
        smem_allocator.add_resource(resource)
    smem_allocator.add(tmem_ptr_allocation)
    smem_allocator.compute_layout()
    expected_offsets = (
        A_SMEM_OFFSET_BYTES,
        B_SMEM_OFFSET_BYTES,
        SFA_SMEM_OFFSET_BYTES,
        SFB_SMEM_OFFSET_BYTES,
        TMEM_PTR_SMEM_OFFSET_BYTES,
    )
    actual_offsets = tuple(
        allocation.offset
        for allocation in (
            a_allocation,
            b_allocation,
            sfa_allocation,
            sfb_allocation,
            tmem_ptr_allocation,
        )
    )
    if actual_offsets != expected_offsets:
        raise ValueError(
            f"SMEM allocator offsets {actual_offsets} do not match {expected_offsets}"
        )

    tmem_allocator = ts.TmemAllocator()
    for resource in (tmem_c, tmem_sfa, tmem_sfb):
        tmem_allocator.add_resource(resource)
    tmem_allocator.compute_layout()

    barrier_allocator = ts.BarrierAllocator()
    barrier_allocator.add(
        ts.BarrierAllocation("tmem_dealloc", 1, WARP_SIZE)
    )
    barrier_resources = [smem_a, smem_b, smem_sfa, smem_sfb, tmem_c]
    for resource in barrier_resources:
        barrier_allocator.add_resource(resource)
    barrier_allocator.compute_layout()

    tasks = [mma_task, store_task, *load_tasks]
    if padding_task is not None:
        tasks.append(padding_task)
    manager = ts.TaskManager(
        tasks=tasks,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        barrier_allocator=barrier_allocator,
        verbose=True,
        exhaustive_deadlock_race_check=config == GemmConfig(),
        # The two independent load schedules use the bounded representative
        # domain to avoid exponential host validation cost.
        exhaustive_representative_domain=config == GemmConfig(),
    )
    device_manager = manager.to_device()
    kernel = make_gemm_kernel(device_manager, config)
    return GemmProgram(
        manager,
        device_manager,
        kernel,
        smem_allocator,
        tmem_allocator,
        barrier_allocator,
        config,
        config.block_threads,
    )


def make_gemm_kernel(
    device_manager,
    config=GemmConfig(),
):
    """Specialize the kernel for one frozen task manager."""

    # Publish the specialized task partition so ptxas can honor setmaxnreg.
    @cl.kernel(max_threads_per_block=(config.block_threads,))
    def gemm_kernel(
        a: torch.Tensor,
        b: torch.Tensor,
        sfa: torch.Tensor,
        sfb: torch.Tensor,
        c: torch.Tensor,
        problem_m: ct.ScalarInt64,
        problem_n: ct.ScalarInt64,
        num_k_tiles: ct.ScalarInt64,
    ) -> None:
        a_map = cl.tensor_map_tiled(
            a,
            (PACKED_BLOCK_K, CTA_M, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        b_map = cl.tensor_map_tiled(
            b,
            (PACKED_BLOCK_K, CTA_N, 1),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        sfa_map = cl.tensor_map_tiled(sfa, (256, 4, 1, 1))
        sfb_map = cl.tensor_map_tiled(sfb, (256, 4, 1, 1))

        # Materialize the unified SMEM and barrier allocators before reading
        # their named allocations.
        device_allocators = device_manager.setup_resources_and_tasks()
        warp_index = device_allocators.warp_index
        tmem_dealloc = device_allocators.barrier_allocator.get_ptr(
            "tmem_dealloc"
        )
        tmem_storage = device_allocators.smem_allocator.get(
            "tmem_ptr",
            cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        )
        if warp_index == 0:
            cl.tcgen05_allocate(
                tmem_storage.get_base_pointer(),
                TMEM_COLUMNS,
                cta_group=cl.CTAGroup.CTA_2,
            )
            cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_2)
        needs_tmem_sync = warp_index < STORE_WARPS or warp_index == MMA_TASK_WARP_IDX
        if needs_tmem_sync:
            cl.barrier_sync_block(
                number_of_threads=(STORE_WARPS + MMA_WARPS) * WARP_SIZE,
                barrier_id=TMEM_SYNC_BARRIER,
            )
        tmem = tmem_storage[0]

        device_manager.run(
            KernelInputs(
                a_map,
                b_map,
                sfa_map,
                sfb_map,
                tmem,
                c,
                problem_m,
                problem_n,
                cl.int32(num_k_tiles),
            ),
            device_allocators,
        )

        cl.barrier_sync_block()
        if warp_index == 0:
            peer_rank = cl.block_in_cluster_index(0) ^ 1
            peer_dealloc = cl.map_shared_to_cluster(
                tmem_dealloc,
                peer_rank,
            )
            cl.mbarrier_arrive(peer_dealloc, scope=cl.MbarrierScope.BLOCK)
            cl.mbarrier_wait_parity(
                tmem_dealloc,
                0,
                time_hint=10_000_000,
            )
            cl.tcgen05_deallocate(
                tmem,
                TMEM_COLUMNS,
                cta_group=cl.CTAGroup.CTA_2,
            )

    return gemm_kernel


_PROGRAMS = {}


def get_gemm_program(use_two_tma_warps=False):
    """Build the host model once so repeated launches reuse its specialization."""
    key = use_two_tma_warps
    if key not in _PROGRAMS:
        _PROGRAMS[key] = make_gemm_program(use_two_tma_warps)
    return _PROGRAMS[key]


# -----------------------------------------------------------------------------
# Tensor preparation, launch, and validation
# -----------------------------------------------------------------------------


def _parse_mnkl(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected four comma-separated integers"
        ) from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected exactly four MNKL values")
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


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """Convert natural (MN, K/16) E4M3 scales to the tcgen05 layout."""
    rows, columns = scale.shape
    if rows % 128 or columns % 4:
        raise ValueError("scale rows and columns must be multiples of 128 and 4")
    row_blocks = rows // 128
    column_blocks = columns // 4
    blocks = scale.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def _make_scale_tensors(
    batch: int,
    rows: int,
    sf_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = (
        torch.randint(1, 3, (batch, rows, sf_k), dtype=torch.int8)
        .to(torch.float8_e4m3fn)
        .permute(1, 2, 0)
    )
    blocked = torch.stack(
        [to_blocked(reference[:, :, index]) for index in range(batch)]
    ).contiguous()
    tma = (
        blocked.view(torch.uint16)
        .reshape(batch, rows // 128, sf_k // 4, 256)
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
        0,
        2,
        (batch, m, k // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    b_storage = torch.randint(
        0,
        2,
        (batch, n, k // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    a = a_storage.permute(2, 1, 0)
    b = b_storage.permute(2, 1, 0)
    a_ref = a_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
    b_ref = b_storage.view(torch.float4_e2m1fn_x2).permute(1, 2, 0)

    sf_k = cl.cdiv(k, SF_VECTOR_SIZE)
    sfa_ref, sfa = _make_scale_tensors(batch, m, sf_k)
    sfb_ref, sfb = _make_scale_tensors(batch, n, sf_k)
    c = torch.empty(
        (batch, m, n),
        dtype=torch.float16,
        device="cuda",
    ).permute(1, 2, 0)
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


def run(
    tensors: dict[str, torch.Tensor],
    stream=None,
    *,
    verbose=False,
    use_two_tma_warps=False,
) -> None:
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

    program = get_gemm_program(use_two_tma_warps)
    if verbose:
        program.manager.print_verbose_report()
    cl.launch(
        torch.cuda.current_stream() if stream is None else stream,
        (m // CTA_M, n // BLOCK_N, batch),
        (program.block_threads, 1, 1),
        program.kernel,
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
    tensors: dict[str, torch.Tensor],
    tolerance: float = _DEFAULT_TOLERANCE,
    **_,
) -> None:
    a_ref, b_ref, c = tensors["a_ref"], tensors["b_ref"], tensors["c"]
    sfa_ref, sfb_ref = tensors["sfa_ref"], tensors["sfb_ref"]
    _, _, batch = c.shape
    reference = torch.empty_like(c)
    for batch_idx in range(batch):
        a_values = _unpack_e2m1_bytes_to_float(a_ref[:, :, batch_idx].view(torch.uint8))
        b_values = _unpack_e2m1_bytes_to_float(b_ref[:, :, batch_idx].view(torch.uint8))
        scale_a = (
            sfa_ref[:, :, batch_idx]
            .to(
                device=a_values.device,
                dtype=torch.float32,
            )
            .repeat_interleave(SF_VECTOR_SIZE, dim=1)
        )
        scale_b = (
            sfb_ref[:, :, batch_idx]
            .to(
                device=b_values.device,
                dtype=torch.float32,
            )
            .repeat_interleave(SF_VECTOR_SIZE, dim=1)
        )
        reference[:, :, batch_idx] = (a_values * scale_a) @ (
            b_values * scale_b
        ).T.contiguous()
    torch.testing.assert_close(c, reference, atol=tolerance, rtol=1.0e-2)


def verify(
    mnkl: tuple[int, int, int, int] = _DEFAULT_MNKL,
    tolerance: float = _DEFAULT_TOLERANCE,
    *,
    use_two_tma_warps=False,
) -> None:
    _validate_mnkl(mnkl)
    print("===================================================================")
    print("Running Blackwell NVFP4 GEMM task-scheduling tutorial with:")
    print(f"  mnkl:       {mnkl}")
    print("  scheduler:  direct CTA_2 tile mapping")
    print(f"  TMA warps:  {2 if use_two_tma_warps else 1}")
    print("  SF copies:  fused with MMA")
    print(f"  tolerance:  {tolerance}")
    print("===================================================================")
    print()
    tensors = prepare_tensors(*mnkl)
    run(
        tensors,
        verbose=True,
        use_two_tma_warps=use_two_tma_warps,
    )
    torch.cuda.synchronize()
    print(f"Run kernel (mnkl={mnkl}) OK", flush=True)
    verify_output(tensors, tolerance=tolerance)
    print(f"verify (mnkl={mnkl}): PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mnkl",
        type=_parse_mnkl,
        default=_DEFAULT_MNKL,
        help="M,N,K,L dimensions",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help="numerical validation tolerance",
    )
    parser.add_argument("--use-two-tma-warps", action="store_true")
    arguments = parser.parse_args()
    verify(
        arguments.mnkl,
        arguments.tolerance,
        use_two_tma_warps=arguments.use_two_tma_warps,
    )
