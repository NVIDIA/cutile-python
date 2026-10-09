# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TMEM accumulator resource for BatchedGemm C tiles."""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts

from .batched_gemm_config import ActKind


@dataclass(kw_only=True, eq=False)
class TmemCResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=2)
    @staticmethod
    def init_accumulator_state(stage_info, cfg):
        tmem_raw_addr = stage_info.context.tasks_inputs.tmem_ptr_i32[0]
        idesc = cl.Tcgen05InstructionDescriptor(
            d_type=cl.float32,
            a_type=cl.bfloat16,
            b_type=cl.bfloat16,
            n=cfg.mma_n,
            m=cfg.mma_m,
        ).encode()
        return tmem_raw_addr, idesc

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_epilogue_state(stage_info):
        return stage_info.context.tasks_inputs.tmem_ptr_i32[0]

    @ts.producer_work
    @staticmethod
    def mma(
        stage_info,
        tmem_raw_addr,
        idesc,
        desc_a_mma_base,
        smem_a_stage_ptr,
        desc_b_mma_base,
        smem_b_stage_ptr,
        cfg,
    ):
        acc_tmem_ptr = cl.tcgen05_tmem_offset(
            tmem_raw_addr,
            column_offset=stage_info.stage_idx * cfg.tmem_c_cols_per_stage,
        )
        is_first_ktile = stage_info.loop_offset == cl.int32(0)
        for kblock_idx in cl.static_iter(range(cfg.tile_k // cfg.mma_k)):
            if cfg.tile_k > 64:
                k_minor = kblock_idx % 4
                k_major = kblock_idx // 4
                desc_a_increment = k_major * 1024 + k_minor * 2
                b_group_stride = max(
                    64, (64 * cfg.num_bytes_b_smem_per_stage // cfg.tile_k) >> 4
                )
                desc_b_increment = k_major * b_group_stride + k_minor * 2
            else:
                desc_a_increment = 2 * kblock_idx
                desc_b_increment = 2 * kblock_idx
            desc_a = desc_a_mma_base + desc_a_increment
            desc_b = desc_b_mma_base + desc_b_increment
            accumulate = ~is_first_ktile if kblock_idx == 0 else cl.bool_(True)
            if cl.elect_sync():
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    acc_tmem_ptr,
                    desc_a,
                    desc_b,
                    idesc,
                    accumulate=accumulate,
                    cta_group=cfg.cta_group,
                )

    @ts.consumer_work(outputs=3)
    @staticmethod
    def consumer_work(stage_info, tmem_raw_addr, subtile_idx, cfg):
        warp_in_group = stage_info.context.warp_index % cfg.num_epilogue_warps
        if cfg.is_swap_ab:
            # Source swapAB path: each four-warp group reads two 16-row slices
            # from the same 16x256b TMEM column fragment.
            warpgroup_idx = warp_in_group // 4
            swap_t2r_repx = cfg.epi_tile_n // 8
            col_offset = (
                subtile_idx * cfg.epi_tile_n
                + warpgroup_idx * cfg.epi_tile_n
            )
            tmem0 = cl.tcgen05_tmem_offset(
                tmem_raw_addr,
                column_offset=(
                    stage_info.stage_idx * cfg.tmem_c_cols_per_stage + col_offset
                ),
            )
            tmem1 = cl.tcgen05_tmem_offset(tmem0, lane_offset=16)
            t2r_rmem = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_16X256B,
                tmem0,
                element_count=swap_t2r_repx * 4,
                dtype=cl.float32,
            )
            t2r_rmem_1 = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_16X256B,
                tmem1,
                element_count=swap_t2r_repx * 4,
                dtype=cl.float32,
            )
            cl.tcgen05_wait_load()
            cl.tcgen05_wait_load()
            return t2r_rmem, t2r_rmem_1, cl.int32(subtile_idx)
        epi_t2r_repx = cfg.epi_tile_n // 4
        tmem = cl.tcgen05_tmem_offset(
            tmem_raw_addr,
            lane_offset=warp_in_group * 32,
            column_offset=(
                stage_info.stage_idx * cfg.tmem_c_cols_per_stage
                + subtile_idx * epi_t2r_repx
            ),
        )
        t2r_rmem = cl.tcgen05_load(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem,
            element_count=epi_t2r_repx,
            dtype=cl.float32,
        )
        cl.tcgen05_wait_load()
        # One completion wait is sufficient for the unrolled static FC2 N64
        # specialization; a redundant wait increases B300 latency.
        # Other paths retain two load-completion waits.
        fc2_static_n64 = (
            not cfg.is_persistent
            and not cfg.is_swap_ab
            and cfg.act_kind == int(ActKind.NONE)
            and cfg.tile_n == 64
            and cfg.tile_k == 64
        )
        if not fc2_static_n64:
            cl.tcgen05_wait_load()
        return t2r_rmem, cl.Vector(0.0, dtype=cl.float32), cl.int32(0)
