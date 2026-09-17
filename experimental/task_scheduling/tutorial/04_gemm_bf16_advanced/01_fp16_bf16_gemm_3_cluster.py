# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Task-scheduled persistent two-CTA FP16/BF16 Blackwell GEMM.

Each two-CTA cluster computes a 256x256 output tile. Warps 4 and 5 issue
independent A and B multicast TMA loads, warp 6 issues CTA_2 tcgen05 MMA,
and warps 0-3 drain FP32 TMEM accumulators. Warp 7 either completes the
register-allocation warp group for static scheduling or drives the CLC work
pipeline for dynamic-persistent scheduling.
"""

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch
import torch.nn.functional as F

import task_scheduling as ts


warp_size = 32
num_mma_ctas = 2
cluster_shape_mnk = (2, 1, 1)
cluster_m = cluster_shape_mnk[0]
cluster_n = cluster_shape_mnk[1]
cluster_size = cluster_m * cluster_n
num_pairs = cluster_size // num_mma_ctas
num_pair_rows = cluster_m // num_mma_ctas
num_pair_cols = cluster_n
_a_mcast_template = sum(1 << (num_mma_ctas * c) for c in range(num_pair_cols))
_b_mcast_template = sum(
    1 << (num_pair_cols * num_mma_ctas * r) for r in range(num_pair_rows)
)
mma_inst_shape_mnk = (256, 256, 16)
mma_tiler_mnk = (256, 256, 64)
mma_tiler_mnk_per_cta = (128, 128, 64)
super_tile_m = mma_tiler_mnk[0]
super_tile_n = mma_tiler_mnk[1]

threads_in_epilogue = 128
ab_stages = 3
acc_stages = 2
num_scheduler_stages = 2
block_size = 8 * warp_size

store_task_warp_idx = 0
load_a_task_warp_idx = 4
load_b_task_warp_idx = 5
mma_task_warp_idx = 6
tmem_sync_barrier = 2
tmem_dealloc_barrier = 3
vec_bytes = 32
vec_alignment_bytes = 16
io_element_bytes = 2
sA_stage_elems = mma_tiler_mnk_per_cta[0] * mma_tiler_mnk[2]
sB_stage_elems = mma_tiler_mnk_per_cta[1] * mma_tiler_mnk[2]
sA_stage_bytes = sA_stage_elems * io_element_bytes
sB_stage_bytes = sB_stage_elems * io_element_bytes
num_tmem_cols = mma_tiler_mnk[1] * acc_stages
clc_response_bytes = cl.cluster_launch_control_token.bitwidth // 8
# The pipeline owns the SMEM arena: A starts at its base and B follows A.
a_smem_offset_bytes = 0
b_smem_offset_bytes = ab_stages * sA_stage_bytes
_DEFAULT_MNK = (512, 512, 512)
_DEFAULT_TOLERANCE = 1.0e-1
_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}
debug_print = False


# -----------------------------------------------------------------------------
# Host resource model and schedule tree
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class GmemAbResource(ts.MemoryResource):
    """Produce A/B coordinates for the current persistent work tile."""

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=4)
    @staticmethod
    def init_tile_coords(stage_info):
        bx, by, bz = stage_info.work_tile.tile_idx
        cta_rank_in_cluster = (by % cluster_n) * cluster_m + (bx % cluster_m)
        return bx, by, bz, cta_rank_in_cluster

    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info, bx, by, bz, cta_rank_in_cluster):
        mma_tile_coord_mnl = (
            bx // cluster_m,
            by // cluster_n,
            bz,
        )
        pair_id = cta_rank_in_cluster // num_mma_ctas
        rank_in_pair = cta_rank_in_cluster % num_mma_ctas
        pair_row = pair_id // num_pair_cols
        pair_col = pair_id % num_pair_cols
        return (
            stage_info.loop_offset * mma_tiler_mnk[2],
            mma_tile_coord_mnl[0] * super_tile_m
            + pair_row * mma_tiler_mnk[0]
            + rank_in_pair * mma_tiler_mnk_per_cta[0],
            mma_tile_coord_mnl[1] * super_tile_n
            + pair_col * mma_tiler_mnk[1]
            + rank_in_pair * mma_tiler_mnk_per_cta[1],
        )


@dataclass(kw_only=True, eq=False)
class SmemAbResource(ts.MemoryResource):
    """Shared-memory A or B operand staged by multicast TMA."""

    @staticmethod
    def _init_multicast_state(stage_info, operand_is_a):
        tasks_inputs = stage_info.context.tasks_inputs
        cta_rank_in_cluster = cl.int32(
            cl._nvvm.read_ptx_sreg_cluster_ctarank()
        )
        pair_id = cta_rank_in_cluster // num_mma_ctas
        rank_in_pair = cta_rank_in_cluster % num_mma_ctas
        pair_row = pair_id // tasks_inputs.act_num_pair_cols
        pair_col = pair_id % tasks_inputs.act_num_pair_cols
        if operand_is_a:
            base = (
                pair_row
                * tasks_inputs.act_num_pair_cols
                * num_mma_ctas
                + rank_in_pair
            )
            tma_mcast_mask = tasks_inputs.act_a_mcast_template << base
            is_leader = pair_col == 0
        else:
            base = pair_col * num_mma_ctas + rank_in_pair
            tma_mcast_mask = tasks_inputs.act_b_mcast_template << base
            is_leader = pair_row == 0
        return rank_in_pair, tma_mcast_mask, is_leader

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=4)
    @staticmethod
    def init_load_state(stage_info, smem_offset, operand_is_a):
        rank_in_pair, tma_mcast_mask, is_leader = (
            SmemAbResource._init_multicast_state(stage_info, operand_is_a)
        )
        shared_smem = (
            stage_info.context.cluster_smem_base.pointer()
            + cl.uint32(smem_offset)
        )
        return shared_smem, rank_in_pair, tma_mcast_mask, is_leader

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=2)
    @staticmethod
    def init_descriptors(
        stage_info,
        smem_offset,
        stage_elems,
        operand_is_a,
    ):
        rank_in_pair, _, _ = SmemAbResource._init_multicast_state(
            stage_info, operand_is_a
        )
        base = stage_info.context.smem_base.pointer()
        pointer = cl.bitcast(
            base + smem_offset,
            cl.pointer_dtype(cl.uint16, base.memory_space),
        )
        shared_smem = cl.Array.from_parts(pointer, (ab_stages, stage_elems))
        return shared_smem, rank_in_pair

    @ts.producer_work
    @staticmethod
    def tma_load_a(
        stage_info,
        cluster_smem_address,
        tma_mcast_mask,
        is_leader,
        coord_k,
        coord_m,
    ):
        destination_address = cluster_smem_address + cl.uint32(
            stage_info.stage_idx * sA_stage_bytes
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            if is_leader:
                cl.copy_async_bulk_tensor_global_to_shared(
                    stage_info.context.tasks_inputs.tma_a_desc,
                    (coord_k, coord_m),
                    destination,
                    stage_info.barrier,
                    multicast_mask=cl.int16(tma_mcast_mask),
                    cta_group=cl.CTAGroup.CTA_2,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_desc_a(stage_info, shared_smem, rank_in_pair):
        descriptor = cl.int64(0)
        if rank_in_pair == 0:
            descriptor = _build_smem_descriptor(
                shared_smem, stage_info.stage_idx
            )
        return descriptor

    @ts.producer_work
    @staticmethod
    def tma_load_b(
        stage_info,
        cluster_smem_address,
        tma_mcast_mask,
        is_leader,
        coord_k,
        coord_n,
    ):
        destination_address = cluster_smem_address + cl.uint32(
            stage_info.stage_idx * sB_stage_bytes
        )
        destination = cl.bitcast(
            destination_address,
            cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED_CLUSTER),
        )
        if cl.elect_sync():
            if is_leader:
                cl.copy_async_bulk_tensor_global_to_shared(
                    stage_info.context.tasks_inputs.tma_b_desc,
                    (coord_k, coord_n),
                    destination,
                    stage_info.barrier,
                    multicast_mask=cl.int16(tma_mcast_mask),
                    cta_group=cl.CTAGroup.CTA_2,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def build_desc_b(stage_info, shared_smem, rank_in_pair):
        descriptor = cl.int64(0)
        if rank_in_pair == 0:
            descriptor = _build_smem_descriptor(
                shared_smem, stage_info.stage_idx
            )
        return descriptor


def _build_smem_descriptor(shared_smem, stage_idx):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=shared_smem.pointer((stage_idx, 0)),
            leading_dimension_byte_offset=16,
            stride_dimension_byte_offset=8 * 128,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )


@dataclass(kw_only=True, eq=False)
class TmemCResource(ts.MemoryResource):
    @staticmethod
    def _init_tmem_state(stage_info):
        """Read the TMEM base and initialize the MMA instruction descriptor."""
        tasks_inputs = stage_info.context.tasks_inputs
        tmem_raw_addr = tasks_inputs.tmem_ptr_i32[0]
        # The tcgen05 input formats use F16=0 and BF16=1. Normalize the
        # aggregate field to uint32 for runtime format selection.
        input_type = cl.uint32(tasks_inputs.is_bf16)
        idesc = cl.Tcgen05InstructionDescriptor(
            d_type=cl.float32,
            a_type=input_type,
            b_type=input_type,
            n=mma_inst_shape_mnk[1],
            m=mma_inst_shape_mnk[0],
        ).encode()
        cta_rank_in_cluster = cl.int32(
            cl._nvvm.read_ptx_sreg_cluster_ctarank()
        )
        return tmem_raw_addr, idesc, cta_rank_in_cluster

    @ts.producer_work(outputs=1)
    @staticmethod
    def mma(
        stage_info,
        desc_a_base,
        desc_b_base,
        tmem_raw_addr,
        idesc,
        cta_rank_in_cluster,
        scale_d,
    ):
        if cta_rank_in_cluster % num_mma_ctas == 0:
            tmem_ptr_for_mma = cl.tcgen05_tmem_offset(
                tmem_raw_addr,
                column_offset=stage_info.stage_idx * mma_tiler_mnk[1],
            )

            # Execute one K-block worth of MMA instructions.
            num_k_blocks = mma_tiler_mnk[2] // mma_inst_shape_mnk[2]
            for k_block_idx in cl.static_iter(range(num_k_blocks)):
                # The descriptor address excludes its four least-significant
                # bits, so convert the byte increment to descriptor units.
                inc_bytes_per_iter = mma_inst_shape_mnk[2] * io_element_bytes
                increment = (inc_bytes_per_iter * k_block_idx) >> 4
                desc_a = desc_a_base + increment
                desc_b = desc_b_base + increment

                if cl.elect_sync():
                    cl.tcgen05_mma(
                        cl.Tcgen05MMAKind.F16,
                        tmem_ptr_for_mma,
                        desc_a,
                        desc_b,
                        idesc,
                        accumulate=scale_d,
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                # Switch to accumulate after the first MMA instruction.
                scale_d = cl.bool_(True)
        return scale_d

    @ts.consumer_work(outputs=1)
    @staticmethod
    def load_subtile(stage_info, tmem_raw_addr, subtile_idx):
        tmem_stage = cl.tcgen05_tmem_offset(
            tmem_raw_addr,
            lane_offset=stage_info.context.warp_index * warp_size,
            column_offset=stage_info.stage_idx * mma_tiler_mnk[1],
        )
        tmem_subtile = cl.tcgen05_tmem_offset(
            tmem_stage,
            column_offset=subtile_idx * 32,
        )
        t2r_rmem = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem_subtile,
            element_count=32,
            dtype=cl.float32,
        )
        cl.tcgen05_wait_load()
        return t2r_rmem

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=3,
    )
    @staticmethod
    def init_accumulator_state(stage_info):
        return TmemCResource._init_tmem_state(stage_info)

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_store_state(stage_info):
        tmem_raw_addr, _, _ = TmemCResource._init_tmem_state(stage_info)
        return tmem_raw_addr

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_work_tile_state(stage_info):
        return cl.bool_(False)


@dataclass(kw_only=True, eq=False)
class GmemDResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def store(stage_info, t2r_rmem, subtile_idx):
        tasks_inputs = stage_info.context.tasks_inputs
        bx, by, bz = stage_info.work_tile.tile_idx
        cta_rank_for_coords = (by % cluster_n) * cluster_m + (bx % cluster_m)
        mma_tile_coord_mnl = (
            bx // cluster_m,
            by // cluster_n,
            bz,
        )
        pair_id = cta_rank_for_coords // num_mma_ctas
        rank_in_pair = cta_rank_for_coords % num_mma_ctas
        pair_row = pair_id // num_pair_cols
        pair_col = pair_id % num_pair_cols
        coordc_m = (
            mma_tile_coord_mnl[0] * super_tile_m
            + pair_row * mma_tiler_mnk[0]
            + rank_in_pair * mma_tiler_mnk_per_cta[0]
        )
        coordc_n = (
            mma_tile_coord_mnl[1] * super_tile_n
            + pair_col * mma_tiler_mnk[1]
        )
        row = coordc_m + cl.thread_index(0)
        column = coordc_n + subtile_idx * 32
        vsize = vec_bytes // 2
        if row < tasks_inputs.num_rows:
            output_row = tasks_inputs.mC_mn.pointer() + (
                cl.uint64(row) * cl.uint64(tasks_inputs.num_cols)
            )
            bias_pointer = tasks_inputs.bias.pointer()
            for vector_idx in cl.static_iter(range(32 // vsize)):
                vector_column = column + vector_idx * vsize
                fragment = t2r_rmem[
                    vector_idx * vsize:vector_idx * vsize + vsize
                ]
                if vector_column + vsize <= tasks_inputs.num_cols:
                    if tasks_inputs.has_bias:
                        bias = (bias_pointer + cl.uint64(vector_column)).load(
                            count=vsize,
                            alignment=vec_alignment_bytes,
                        )
                        fragment = fragment + bias.astype(cl.float32)
                    packed = fragment.astype(tasks_inputs.mC_mn.dtype)
                    (output_row + cl.uint64(vector_column)).store(
                        packed, alignment=vec_alignment_bytes
                    )
                else:
                    # Preserve the vectorized common path while allowing an
                    # arbitrary N tail without padding the output or bias.
                    for element_idx in cl.static_iter(range(vsize)):
                        scalar_column = vector_column + element_idx
                        if scalar_column < tasks_inputs.num_cols:
                            value = fragment[element_idx]
                            if tasks_inputs.has_bias:
                                value += (
                                    bias_pointer + cl.uint64(scalar_column)
                                ).load()
                            packed = cl.Vector(
                                value, dtype=cl.float32
                            ).astype(tasks_inputs.mC_mn.dtype)
                            (output_row + cl.uint64(scalar_column)).store(packed[0])


# -----------------------------------------------------------------------------
# Resource construction helpers
# -----------------------------------------------------------------------------


def create_gmem_ab_resource():
    """Create the global-memory A/B coordinate resource."""
    return GmemAbResource(name="GmemAb")


def create_smem_ab_resource(operand, smem_allocation):
    """Create an A or B SMEM resource and its TMA-UMMA pipeline."""
    if operand == "a":
        stage_bytes = sA_stage_bytes
    elif operand == "b":
        stage_bytes = sB_stage_bytes
    else:
        raise ValueError(f"operand must be 'a' or 'b', got {operand}")

    pipeline_config = ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=ab_stages,
        num_bytes=stage_bytes * num_mma_ctas,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(1),
        cta_layout_vmnk=(2, 1, 1, 1),
        consumer_signaling_threads=ts.SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=stage_bytes,
    )
    return SmemAbResource(
        name=f"Smem{operand.upper()}",
        pipeline_config=pipeline_config,
        smem_requirements=[smem_allocation],
    )


def create_tmem_c_resource(tmem_allocation, num_epilogue_warps):
    """Create the TMEM accumulator resource and UMMA pipeline."""
    consumer_group = ts.CooperativeGroup(
        num_epilogue_warps * warp_size * num_mma_ctas
    )
    pipeline_config = ts.PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=acc_stages,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=consumer_group,
        cta_layout_vmnk=(2, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.CtaLeader,
    )
    return TmemCResource(
        name="TmemC",
        pipeline_config=pipeline_config,
        tmem_requirements=[tmem_allocation],
    )


def create_gmem_d_resource():
    """Create the global-memory output resource."""
    return GmemDResource(name="GmemD")


def create_work_queue(
    tile_sched_params,
    use_clc_dynamic_scheduler,
    num_load_warps,
    num_epilogue_warps,
    num_mma_warps,
    num_padding_warps,
    num_scheduler_warps,
    clc_response_allocation=None,
):
    """Create the work queue for static or CLC-dynamic scheduling."""
    if use_clc_dynamic_scheduler:
        cluster_consumer_threads = (
            (
                num_load_warps
                + num_mma_warps
                + num_epilogue_warps
                + num_padding_warps
                + num_scheduler_warps
            )
            * warp_size
            * cluster_m
        )
        pipeline_config = (
            ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
                num_stages=num_scheduler_stages,
                num_bytes=clc_response_bytes,
                producer_group=ts.CooperativeGroup(1),
                consumer_group=ts.CooperativeGroup(cluster_consumer_threads),
                cta_layout_vmnk=(2, 1, 1, 1),
                producer_signaling_threads=ts.SignalingThreads.CtaLeader,
                consumer_signaling_threads=ts.SignalingThreads.All,
            )
        )
        tile_scheduler_config = (
            ts.TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_sched_params,
                clc_response_allocation,
            )
        )
        return ts.WorkQueue(
            name="WorkQueue",
            tile_scheduler_config=tile_scheduler_config,
            pipeline_config=pipeline_config,
            smem_requirements=[clc_response_allocation],
        )

    tile_scheduler_config = (
        ts.TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
            tile_sched_params
        )
    )
    return ts.WorkQueue(
        name="WorkQueue",
        tile_scheduler_config=tile_scheduler_config,
    )


# -----------------------------------------------------------------------------
# Task schedule construction helpers
# -----------------------------------------------------------------------------


def create_load_a_task(
    gmem_ab_resource,
    smem_a_resource,
    pdl_wait,
    pdl_launch,
    work_queue,
    smem_offset,
):
    """Create the A TMA load task."""

    @ts.schedule
    def load_a_schedule(
        stage_info,
        gmem_ab,
        smem_a,
        pdl_wait_resource,
        pdl_launch_resource,
        wq,
    ):
        pdl_wait_resource.wait_griddep()
        shared_smem, _, tma_mcast_mask, is_leader = smem_a.init_load_state(
            smem_offset,
            True,
        )
        with ts.work_tile_loop(wq):
            bx, by, bz, cta_rank_in_cluster = gmem_ab.init_tile_coords()

            def loop_body():
                coord_k, coord_m, _ = gmem_ab.compute_coords(
                    bx, by, bz, cta_rank_in_cluster
                )
                smem_a.try_acquire()
                smem_a.acquire()
                smem_a.tma_load_a(
                    shared_smem,
                    tma_mcast_mask,
                    is_leader,
                    coord_k,
                    coord_m,
                )
                smem_a.commit()

            ts.domain_loop(
                0,
                stage_info.context.tasks_inputs.num_k_tiles,
                1,
                loop_body,
            )
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()
        pdl_launch_resource.launch_griddep()

    result = load_a_schedule(
        gmem_ab_resource,
        smem_a_resource,
        pdl_wait,
        pdl_launch,
        work_queue,
    )
    return ts.Task(
        warp_idx=load_a_task_warp_idx,
        num_warps=1,
        schedule=result,
        num_registers=40,
        name="LoadATask",
        debug_print=debug_print,
    )


def create_load_b_task(
    gmem_ab_resource,
    smem_b_resource,
    work_queue,
    smem_offset,
):
    """Create the B TMA load task."""

    @ts.schedule
    def load_b_schedule(stage_info, gmem_ab, smem_b, wq):
        shared_smem, _, tma_mcast_mask, is_leader = smem_b.init_load_state(
            smem_offset,
            False,
        )
        with ts.work_tile_loop(wq):
            bx, by, bz, cta_rank_in_cluster = gmem_ab.init_tile_coords()

            def loop_body():
                coord_k, _, coord_n = gmem_ab.compute_coords(
                    bx, by, bz, cta_rank_in_cluster
                )
                smem_b.try_acquire()
                smem_b.acquire()
                smem_b.tma_load_b(
                    shared_smem,
                    tma_mcast_mask,
                    is_leader,
                    coord_k,
                    coord_n,
                )
                smem_b.commit()

            ts.domain_loop(
                0,
                stage_info.context.tasks_inputs.num_k_tiles,
                1,
                loop_body,
            )
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = load_b_schedule(gmem_ab_resource, smem_b_resource, work_queue)
    return ts.Task(
        warp_idx=load_b_task_warp_idx,
        num_warps=1,
        schedule=result,
        num_registers=40,
        name="LoadBTask",
        debug_print=debug_print,
    )


def create_mma_task(
    smem_a_resource,
    smem_b_resource,
    tmem_c_resource,
    work_queue,
    a_smem_offset,
    b_smem_offset,
    num_mma_warps,
):
    """Create the MMA compute task."""

    @ts.schedule
    def mma_schedule(stage_info, smem_a, smem_b, tmem_c, wq):
        shared_smem_a, rank_in_pair_a = smem_a.init_descriptors(
            a_smem_offset,
            sA_stage_elems,
            True,
        )
        shared_smem_b, rank_in_pair_b = smem_b.init_descriptors(
            b_smem_offset,
            sB_stage_elems,
            False,
        )
        tmem_raw_addr, idesc, cta_rank_in_cluster = (
            tmem_c.init_accumulator_state()
        )
        with ts.work_tile_loop(wq):
            scale_d = tmem_c.init_work_tile_state()
            tmem_c.try_acquire()
            tmem_c.acquire()

            def loop_body(scale_d):
                smem_a.try_wait()
                smem_b.try_wait()
                smem_a.wait()
                smem_b.wait()
                desc_a_base = smem_a.build_desc_a(
                    shared_smem_a,
                    rank_in_pair_a,
                )
                desc_b_base = smem_b.build_desc_b(
                    shared_smem_b,
                    rank_in_pair_b,
                )
                scale_d = tmem_c.mma(
                    desc_a_base,
                    desc_b_base,
                    tmem_raw_addr,
                    idesc,
                    cta_rank_in_cluster,
                    scale_d,
                )
                smem_a.release()
                smem_b.release()
                return scale_d

            scale_d = ts.domain_loop(
                0,
                stage_info.context.tasks_inputs.num_k_tiles,
                1,
                loop_body,
                scale_d,
            )
            tmem_c.commit()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = mma_schedule(
        smem_a_resource,
        smem_b_resource,
        tmem_c_resource,
        work_queue,
    )
    return ts.Task(
        warp_idx=mma_task_warp_idx,
        num_warps=num_mma_warps,
        schedule=result,
        num_registers=40,
        name="MmaTask",
        debug_print=debug_print,
    )


def create_store_task(
    tmem_c_resource,
    gmem_d_resource,
    work_queue,
    num_epilogue_warps,
):
    """Create the epilogue store task."""
    subtile_cnt = mma_tiler_mnk[1] // 32

    @ts.schedule
    def store_schedule(stage_info, tmem_c, gmem_d, wq):
        tmem_raw_addr = tmem_c.init_store_state()
        with ts.work_tile_loop(wq):
            with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
                pass
            tmem_c.try_wait()
            tmem_c.wait()
            for subtile_idx in range(subtile_cnt):
                t2r_rmem = tmem_c.load_subtile(
                    tmem_raw_addr,
                    subtile_idx=subtile_idx,
                )
                gmem_d.store(t2r_rmem, subtile_idx=subtile_idx)
            tmem_c.release()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = store_schedule(tmem_c_resource, gmem_d_resource, work_queue)
    return ts.Task(
        warp_idx=store_task_warp_idx,
        num_warps=num_epilogue_warps,
        schedule=result,
        num_registers=160,
        name="StoreTask",
        debug_print=debug_print,
    )


def create_padding_task(
    work_queue,
    num_padding_warps,
    total_num_warps_so_far,
):
    """Create the padding task."""

    @ts.schedule
    def padding_schedule(stage_info, wq):
        with ts.work_tile_loop(wq):
            with ts.domain_loop(stage_info.context.tasks_inputs.num_k_tiles):
                pass
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = padding_schedule(work_queue)
    return ts.Task(
        warp_idx=total_num_warps_so_far,
        num_warps=num_padding_warps,
        schedule=result,
        num_registers=40,
        name="PaddingTask",
        debug_print=debug_print,
    )


def create_work_schedule_task(
    work_queue,
    num_scheduler_warps,
    scheduler_warp_idx,
):
    """Create the CLC dynamic-persistent scheduler task."""

    @ts.schedule
    def scheduler_schedule(stage_info, wq):
        with ts.work_tile_loop(wq):
            with ts.domain_loop(0):
                pass
            wq.try_acquire()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    result = scheduler_schedule(work_queue)
    return ts.Task(
        warp_idx=scheduler_warp_idx,
        num_warps=num_scheduler_warps,
        schedule=result,
        num_registers=40,
        name="WorkScheduleTask",
        debug_print=debug_print,
    )


@dataclass(frozen=True)
class TasksInputs:
    tma_a_desc: object
    tma_b_desc: object
    tmem_ptr_i32: object
    mC_mn: object
    bias: object
    num_k_tiles: object
    num_rows: object
    num_cols: object
    act_num_pair_cols: object
    act_a_mcast_template: object
    act_b_mcast_template: object
    is_bf16: object
    has_bias: object


@dataclass(frozen=True)
class GemmPipeline:
    """Host-side task pipeline and its frozen device representation."""

    task_manager: object
    device_task_manager: object
    mma_task: object
    store_task: object
    work_queue: ts.WorkQueue
    allocator: object
    tmem_allocator: object
    barrier_allocator: object
    tmem_ptr_alloc: object
    a_allocation: object
    b_allocation: object
    use_clc_dynamic_scheduler: bool


def _create_gemm_pipeline(m, n, use_clc_dynamic_scheduler=False):
    """Create resources, tasks, allocators, and the task manager."""
    num_epilogue_warps = 4
    num_mma_warps = 1
    num_load_warps = 2
    num_scheduler_warps = 1 if use_clc_dynamic_scheduler else 0
    total_num_warps_so_far = (
        num_epilogue_warps
        + num_mma_warps
        + num_load_warps
        + num_scheduler_warps
    )
    num_padding_warps = (
        (total_num_warps_so_far + 3) // 4 * 4 - total_num_warps_so_far
    )

    # Resource construction.
    a_allocation = ts.SmemAllocation(
        "smem_a",
        ab_stages * sA_stage_bytes,
        alignment=128,
    )
    b_allocation = ts.SmemAllocation(
        "smem_b",
        ab_stages * sB_stage_bytes,
        alignment=128,
    )
    acc_allocation = ts.TmemAllocation("tmem_acc", num_tmem_cols)
    tmem_ptr_alloc = ts.SmemAllocation("tmem_ptr", 4, alignment=4)

    gmem_ab_resource = create_gmem_ab_resource()
    smem_a_resource = create_smem_ab_resource(
        "a",
        a_allocation,
    )
    smem_b_resource = create_smem_ab_resource(
        "b",
        b_allocation,
    )
    tmem_c_resource = create_tmem_c_resource(
        acc_allocation,
        num_epilogue_warps,
    )
    gmem_d_resource = create_gmem_d_resource()
    pdl_wait = ts.PdlWaitBarrier(name="PdlWait")
    pdl_launch = ts.PdlLaunchBarrier(name="PdlLaunch")

    problem_shape_ntile_mnl = (
        (m + mma_tiler_mnk_per_cta[0] - 1)
        // mma_tiler_mnk_per_cta[0],
        (n + mma_tiler_mnk[1] - 1) // mma_tiler_mnk[1],
        1,
    )
    clc_response_allocation = None
    if use_clc_dynamic_scheduler:
        tile_sched_params = ts.ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            cluster_shape_mnk,
        )
        clc_response_allocation = ts.SmemAllocation(
            "clc_response",
            size_bytes=num_scheduler_stages * clc_response_bytes,
            alignment=16,
            count=num_scheduler_stages,
        )
    else:
        tile_sched_params = ts.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            cluster_shape_mnk,
        )
    work_queue = create_work_queue(
        tile_sched_params,
        use_clc_dynamic_scheduler,
        num_load_warps,
        num_epilogue_warps,
        num_mma_warps,
        num_padding_warps,
        num_scheduler_warps,
        clc_response_allocation,
    )

    # Unified SMEM allocator: operand buffers and infrastructure storage.
    allocator = ts.SmemAllocator(default_add_barriers=False)
    allocator.add_resource(smem_a_resource)
    allocator.add_resource(smem_b_resource)
    allocator.add(tmem_ptr_alloc)
    if use_clc_dynamic_scheduler:
        allocator.add_resource(work_queue)
    allocator.compute_layout()

    # Task schedule construction.
    load_a_task = create_load_a_task(
        gmem_ab_resource,
        smem_a_resource,
        pdl_wait,
        pdl_launch,
        work_queue,
        a_allocation.offset,
    )
    load_b_task = create_load_b_task(
        gmem_ab_resource,
        smem_b_resource,
        work_queue,
        b_allocation.offset,
    )
    mma_task = create_mma_task(
        smem_a_resource,
        smem_b_resource,
        tmem_c_resource,
        work_queue,
        a_allocation.offset,
        b_allocation.offset,
        num_mma_warps,
    )
    store_task = create_store_task(
        tmem_c_resource,
        gmem_d_resource,
        work_queue,
        num_epilogue_warps,
    )
    task_list = [load_a_task, load_b_task, mma_task, store_task]

    if num_padding_warps > 0:
        padding_task = create_padding_task(
            work_queue,
            num_padding_warps,
            total_num_warps_so_far,
        )
        task_list.append(padding_task)

    if use_clc_dynamic_scheduler:
        scheduler_warp_idx = total_num_warps_so_far - num_scheduler_warps
        work_schedule_task = create_work_schedule_task(
            work_queue,
            num_scheduler_warps,
            scheduler_warp_idx,
        )
        task_list.append(work_schedule_task)

    resource_dependency_graph = {
        pdl_launch: [],
        smem_a_resource: [gmem_ab_resource, pdl_wait, work_queue],
        smem_b_resource: [gmem_ab_resource, work_queue],
        tmem_c_resource: [
            smem_a_resource,
            smem_b_resource,
            work_queue,
        ],
        gmem_d_resource: [tmem_c_resource, work_queue],
    }
    if use_clc_dynamic_scheduler:
        resource_dependency_graph[work_queue] = [work_queue]

    tmem_allocator = ts.TmemAllocator()
    tmem_allocator.add_resource(tmem_c_resource)
    tmem_allocator.compute_layout()

    barrier_allocator = ts.BarrierAllocator()
    barrier_allocator.add(
        ts.BarrierAllocation("tmem_dealloc", 1, warp_size)
    )
    for resource in (smem_a_resource, smem_b_resource, tmem_c_resource):
        barrier_allocator.add_resource(resource)
    if use_clc_dynamic_scheduler:
        barrier_allocator.add_resource(work_queue)
    barrier_allocator.compute_layout()

    task_manager = ts.TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=allocator,
        tmem_allocator=tmem_allocator,
        barrier_allocator=barrier_allocator,
        verbose=True,
        exhaustive_representative_domain=True,
    )
    device_task_manager = task_manager.to_device()
    return GemmPipeline(
        task_manager,
        device_task_manager,
        mma_task,
        store_task,
        work_queue,
        allocator,
        tmem_allocator,
        barrier_allocator,
        tmem_ptr_alloc,
        a_allocation,
        b_allocation,
        use_clc_dynamic_scheduler,
    )


# -----------------------------------------------------------------------------
# Kernel launch and validation
# -----------------------------------------------------------------------------


def kernel(device_task_manager):
    """Specialize the kernel for one frozen task manager."""

    @cl.kernel(max_threads_per_block=(block_size,))
    def gemm_kernel(
        a,
        b,
        mC_mn,
        bias,
        m: cl.int32,
        n: cl.int32,
        k: cl.int32,
        act_num_pair_cols: cl.int32,
        act_a_mcast_template: cl.int32,
        act_b_mcast_template: cl.int32,
        is_bf16: cl.Constant[bool],
        has_bias: cl.Constant[bool],
    ):
        tma_a_desc = cl.tensor_map_tiled(
            a,
            (mma_tiler_mnk[2], mma_tiler_mnk_per_cta[0]),
            order="F",
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        tma_b_desc = cl.tensor_map_tiled(
            b,
            (mma_tiler_mnk[2], mma_tiler_mnk_per_cta[1]),
            order="F",
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
        )
        warp_index = cl.warp_index()
        # Overlap tensor-map fetch with resource setup and cluster sync.
        if warp_index == load_a_task_warp_idx:
            cl.prefetch_tensor_map(tma_a_desc)
            cl.prefetch_tensor_map(tma_b_desc)

        device_allocators = device_task_manager.setup_resources_and_tasks()
        warp_index = device_allocators.warp_index
        tmem_ptr_i32 = device_allocators.smem_allocator.get(
            "tmem_ptr",
            cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        )
        tmem_dealloc_mbar_ptr = device_allocators.barrier_allocator.get_ptr(
            "tmem_dealloc"
        )

        if warp_index == 0:
            cl.tcgen05_allocate(
                tmem_ptr_i32.pointer(),
                num_tmem_cols,
                cta_group=cl.CTAGroup.CTA_2,
            )
            cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_2)
        if (
            warp_index < threads_in_epilogue // warp_size
            or warp_index == mma_task_warp_idx
        ):
            cl.barrier_sync_block_aligned(
                number_of_threads=threads_in_epilogue + warp_size,
                barrier_id=tmem_sync_barrier,
            )
        device_task_manager.run(
            TasksInputs(
                tma_a_desc,
                tma_b_desc,
                tmem_ptr_i32,
                mC_mn,
                bias,
                (k + mma_tiler_mnk[2] - 1) // mma_tiler_mnk[2],
                m,
                n,
                act_num_pair_cols,
                act_a_mcast_template,
                act_b_mcast_template,
                is_bf16,
                has_bias,
            ),
            device_allocators,
        )

        if warp_index < threads_in_epilogue // warp_size:
            cl.barrier_sync_block_aligned(
                number_of_threads=threads_in_epilogue,
                barrier_id=tmem_dealloc_barrier,
            )
        if warp_index == 0:
            tmem_raw_addr = tmem_ptr_i32[0]
            peer_rank = cl._nvvm.read_ptx_sreg_cluster_ctarank() ^ 1
            peer_dealloc = cl.map_shared_to_cluster(
                tmem_dealloc_mbar_ptr,
                peer_rank,
            )
            cl.mbarrier_arrive(
                peer_dealloc, scope=cl.MbarrierScope.BLOCK
            )
            cl.mbarrier_wait_parity(
                tmem_dealloc_mbar_ptr, 0, time_hint=10_000_000
            )
            cl.tcgen05_deallocate(
                tmem_raw_addr,
                num_tmem_cols,
                cta_group=cl.CTAGroup.CTA_2,
            )

    return gemm_kernel


def _validate_mnk(mnk: tuple[int, int, int]) -> None:
    if len(mnk) != 3:
        raise ValueError("MNK must contain exactly three values")
    m, n, k = mnk
    if m <= 0 or n <= 0:
        raise ValueError("M and N must be positive")
    if k <= 0 or k % mma_tiler_mnk[2]:
        raise ValueError(
            f"K must be a positive multiple of {mma_tiler_mnk[2]}"
        )


def prepare_tensors(
    m: int, n: int, k: int, dtype: str = "fp16", has_bias: bool = False
) -> dict[str, torch.Tensor]:
    _validate_mnk((m, n, k))
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"dtype must be one of {tuple(_DTYPE_MAP)}")
    torch.manual_seed(1111)
    torch_dtype = _DTYPE_MAP[dtype]

    def make(rows, columns):
        return (torch.randn(rows, columns, dtype=torch.float32) * 0.01).to(
            device="cuda:0", dtype=torch_dtype
        )

    tensors = {
        "a": make(m, k),
        "b": make(n, k),
        "c": torch.empty((m, n), device="cuda:0", dtype=torch_dtype),
    }
    if has_bias:
        tensors["bias"] = torch.randn(n, device="cuda:0", dtype=torch_dtype)
    return tensors


def compute_grid(work_queue, use_clc_dynamic_scheduler):
    """Compute the persistent launch grid for the selected scheduler."""
    tile_sched_params = work_queue.tile_scheduler_config.tile_scheduler_params
    if use_clc_dynamic_scheduler:
        return ts.ClcDynamicPersistentTileScheduler.get_grid_shape(
            tile_sched_params
        )

    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    max_active_clusters = max(1, sm_count // cluster_m)
    return ts.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params,
        max_active_clusters,
    )


def host_function(
    tensors: dict[str, torch.Tensor],
    stream=None,
    *,
    verbose=False,
    use_clc_dynamic_scheduler=False,
) -> None:
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    m, k = a.shape
    if b.ndim != 2 or b.shape[1] != k:
        raise ValueError("B must have shape (N, K) with the same K as A")
    n = b.shape[0]
    _validate_mnk((m, n, k))
    if c.shape != (m, n):
        raise ValueError("C must have shape (M, N)")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise ValueError("A, B, and C must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("A, B, and C must be FP16 or BF16")

    bias = tensors.get("bias")
    if bias is not None and (bias.shape != (n,) or bias.dtype != a.dtype):
        raise ValueError("bias must have shape (N,) and match the input dtype")

    pipeline = _create_gemm_pipeline(m, n, use_clc_dynamic_scheduler)
    if verbose:
        pipeline.task_manager.print_verbose_report()
    compiled_kernel = kernel(pipeline.device_task_manager)
    grid = compute_grid(pipeline.work_queue, use_clc_dynamic_scheduler)
    cl.launch(
        torch.cuda.current_stream() if stream is None else stream,
        grid,
        (block_size, 1, 1),
        compiled_kernel,
        (
            a,
            b,
            c,
            c.reshape(-1) if bias is None else bias,
            m,
            n,
            k,
            num_pair_cols,
            _a_mcast_template,
            _b_mcast_template,
            a.dtype == torch.bfloat16,
            bias is not None,
        ),
        block_in_cluster_count=cluster_shape_mnk,
        programmatic_dependent_launch=True,
    )


def verify_output(tensors: dict[str, torch.Tensor], tolerance=_DEFAULT_TOLERANCE):
    a, b, c = tensors["a"], tensors["b"], tensors["c"]
    bias = tensors.get("bias")
    expected = F.linear(
        a.float(),
        b.float(),
        bias.float() if bias is not None else None,
    ).to(c.dtype)
    torch.testing.assert_close(c, expected, atol=tolerance, rtol=1.0e-5)


def verify(
    mnk: tuple[int, int, int] = _DEFAULT_MNK,
    dtype: str = "fp16",
    has_bias: bool = False,
    tolerance: float = _DEFAULT_TOLERANCE,
    use_clc_dynamic_scheduler: bool = False,
):
    print("===================================================================")
    print("Running Blackwell 16-bit GEMM TS tutorial kernel with:")
    print(f"  mnk:       {mnk}")
    print(f"  dtype:     {dtype}")
    print(f"  has_bias:  {has_bias}")
    scheduler_mode = (
        "CLC dynamic persistent"
        if use_clc_dynamic_scheduler
        else "static persistent"
    )
    print(f"  scheduler: {scheduler_mode}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()
    tensors = prepare_tensors(*mnk, dtype=dtype, has_bias=has_bias)
    host_function(
        tensors,
        use_clc_dynamic_scheduler=use_clc_dynamic_scheduler,
    )
    torch.cuda.synchronize()
    verify_output(tensors, tolerance)
    print("PASS")


def _parse_mnk(value):
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected exactly three MNK values")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnk", type=_parse_mnk, default=_DEFAULT_MNK)
    parser.add_argument("--dtype", choices=tuple(_DTYPE_MAP), default="fp16")
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument(
        "--clc-dynamic-scheduler",
        action="store_true",
        help="use CLC dynamic-persistent scheduling",
    )
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    arguments = parser.parse_args()
    verify(
        arguments.mnk,
        arguments.dtype,
        arguments.has_bias,
        arguments.tolerance,
        arguments.clc_dynamic_scheduler,
    )
