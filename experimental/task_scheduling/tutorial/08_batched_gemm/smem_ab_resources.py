# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SMEM operand resources; retain the source K-box and descriptor arithmetic."""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts


def _init_smem_state(stage_info, smem_offset):
    return stage_info.context.smem_base.pointer() + smem_offset


def _tma_load(smem_dst, tma_desc, coords, barrier, cfg):
    if cfg.has_cluster:
        cta_rank = cl.int32(cl._nvvm.read_ptx_sreg_cluster_ctarank())
        smem_dst = cl.map_shared_to_cluster(smem_dst, cta_rank)
        cl.copy_async_bulk_tensor_global_to_shared(
            tma_desc,
            coords,
            smem_dst,
            barrier,
            multicast_mask=cl.int16(cl.int32(1) << cta_rank),
            cta_group=cl.CTAGroup.CTA_2,
        )
    else:
        cl.copy_async_bulk_tensor_global_to_shared(tma_desc, coords, smem_dst, barrier)


def _build_mma_desc(smem_buf, stage_idx, stage_bytes):
    stage_base = smem_buf + stage_bytes * stage_idx
    desc_mma_base = cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=stage_base,
            leading_dimension_byte_offset=stage_bytes,
            stride_dimension_byte_offset=1024,
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )
    return desc_mma_base, stage_base


@dataclass(kw_only=True, eq=False)
class SmemAResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_load_state(stage_info, smem_offset):
        return _init_smem_state(stage_info, smem_offset)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_mma_state(stage_info, smem_offset):
        return _init_smem_state(stage_info, smem_offset)

    @ts.producer_work
    @staticmethod
    def load_a_tile(
        stage_info,
        smem_buf,
        coord_a_k,
        coord_a_mn,
        coord_a_l,
        expert_idx,
        mn_limit,
        cfg,
    ):
        stage_base = smem_buf + cfg.num_bytes_a_per_stage * stage_info.stage_idx
        if cl.elect_sync():
            # One 4-D BF16 K-box TMA descriptor transfers
            # every 64-element K slice in this A stage, even for tile_k > 64.
            _tma_load(
                stage_base,
                stage_info.context.tasks_inputs.tma_a_desc,
                (cl.int32(0), coord_a_mn, coord_a_k // cl.int32(64), coord_a_l),
                stage_info.barrier,
                cfg,
            )

    @ts.consumer_work(outputs=2)
    @staticmethod
    def build_mma_desc_a(stage_info, smem_buf, cfg):
        return _build_mma_desc(
            smem_buf, stage_info.stage_idx, cfg.num_bytes_a_per_stage
        )


@dataclass(kw_only=True, eq=False)
class SmemBResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_load_state(stage_info, smem_offset):
        return _init_smem_state(stage_info, smem_offset)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_mma_state(stage_info, smem_offset):
        return _init_smem_state(stage_info, smem_offset)

    @ts.producer_work
    @staticmethod
    def load_b_tile(
        stage_info,
        smem_buf,
        coord_b_k,
        coord_b_mn,
        coord_b_l,
        mn_limit,
        cfg,
    ):
        if cfg.split_b_across_ctas:
            cta_rank = cl.int32(cl._nvvm.read_ptx_sreg_cluster_ctarank())
            coord_b_mn = coord_b_mn + cta_rank * cl.int32(cfg.tile_n // cfg.cluster_m)
        stage_base = smem_buf + cfg.num_bytes_b_smem_per_stage * stage_info.stage_idx
        if cl.elect_sync():
            b_box_k = 64
            for bi in cl.static_iter(range(max(1, cfg.tile_k // b_box_k))):
                box_bytes = b_box_k * cfg.num_bytes_b_smem_per_stage // cfg.tile_k
                _tma_load(
                    stage_base + bi * box_bytes,
                    stage_info.context.tasks_inputs.tma_b_desc,
                    (coord_b_k + cl.int32(bi * b_box_k), coord_b_mn, coord_b_l),
                    stage_info.barrier,
                    cfg,
                )

    @ts.consumer_work(outputs=2)
    @staticmethod
    def build_mma_desc_b(stage_info, smem_buf, cfg):
        return _build_mma_desc(
            smem_buf, stage_info.stage_idx, cfg.num_bytes_b_smem_per_stage
        )
