# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Task factories for batched-GEMM load/MMA/epilogue schedules."""

from contextlib import contextmanager

import task_scheduling as ts

from .batched_gemm_config import ActKind


def _is_fc2(cfg):
    return not cfg.is_swap_ab and cfg.act_kind == int(ActKind.NONE)


def _load_domain_unroll(cfg):
    # Six-way load-domain expansion with a tail improves B300 performance
    # for the CLC FC2 N128 specialization.
    if cfg.is_persistent and _is_fc2(cfg) and cfg.tile_n == 128 and cfg.tile_k == 128:
        return 6
    return 0


def _mma_domain_unroll(cfg):
    # Six-way expansion gives the static FC2 N64 specialization enough
    # independent pipeline work to match the source kernel on B300.
    # The main loop plus guarded tail preserves its runtime K bound.
    if not cfg.is_persistent and _is_fc2(cfg) and cfg.tile_n == 64 and cfg.tile_k == 64:
        return 6
    if cfg.use_unroll_loop_2x_for_mma:
        return 2
    return 0


def _persistent_tail(work_queue):
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


@contextmanager
def _work_tile_schedule_loop(cfg, work_queue):
    if cfg.is_persistent:
        with ts.work_tile_loop(work_queue):
            yield
            _persistent_tail(work_queue)
    else:
        yield


def create_load_a_task(cfg, gmem_a, smem_a, work_queue, num_k_tiles, smem_offset):
    load_domain_unroll = _load_domain_unroll(cfg)

    def load_a_schedule_body(stage_info, gmem, smem, wq=None):
        gmem.init_coords_state()
        smem_buf = smem.init_load_state(smem_offset)
        with _work_tile_schedule_loop(cfg, wq):
            coords = gmem.compute_a_coords_head(cfg=cfg)

            def loop_body():
                coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit = (
                    gmem.compute_a_coords_loop(*coords[1:], cfg=cfg)
                )
                smem.try_acquire()
                smem.acquire()
                smem.load_a_tile(
                    smem_buf,
                    coord_a_k=coord_a_k,
                    coord_a_mn=coord_a_mn,
                    coord_a_l=coord_a_l,
                    expert_idx=expert_idx,
                    mn_limit=mn_limit,
                    cfg=cfg,
                )
                smem.commit()

            if load_domain_unroll:
                ts.domain_loop(
                    0, num_k_tiles, 1, loop_body, unroll=load_domain_unroll
                )
            else:
                ts.domain_loop(0, num_k_tiles, 1, loop_body)

    @ts.schedule
    def direct_load_a_schedule(stage_info, gmem, smem):
        load_a_schedule_body(stage_info, gmem, smem)

    @ts.schedule
    def persistent_load_a_schedule(stage_info, gmem, smem, wq):
        load_a_schedule_body(stage_info, gmem, smem, wq)

    load_a_schedule = (
        persistent_load_a_schedule if cfg.is_persistent else direct_load_a_schedule
    )
    resources = (gmem_a, smem_a, work_queue) if cfg.is_persistent else (gmem_a, smem_a)
    return ts.Task(
        name="LoadATask",
        warp_idx=cfg.load_a_warp_idx,
        num_warps=cfg.num_load_a_warps,
        num_registers=cfg.load_a_task_regs,
        schedule=load_a_schedule(*resources),
    )


def create_load_b_task(cfg, gmem_b, smem_b, work_queue, num_k_tiles, smem_offset):
    load_domain_unroll = _load_domain_unroll(cfg)

    def load_b_schedule_body(stage_info, gmem, smem, wq=None):
        gmem.init_coords_state()
        smem_buf = smem.init_load_state(smem_offset)
        with _work_tile_schedule_loop(cfg, wq):
            coords = gmem.compute_b_coords_head(cfg=cfg)

            def loop_body():
                coord_b_k, coord_b_mn, coord_b_l, mn_limit = gmem.compute_b_coords_loop(
                    *coords[1:], cfg=cfg
                )
                smem.try_acquire()
                smem.acquire()
                smem.load_b_tile(
                    smem_buf,
                    coord_b_k=coord_b_k,
                    coord_b_mn=coord_b_mn,
                    coord_b_l=coord_b_l,
                    mn_limit=mn_limit,
                    cfg=cfg,
                )
                smem.commit()

            if load_domain_unroll:
                ts.domain_loop(
                    0, num_k_tiles, 1, loop_body, unroll=load_domain_unroll
                )
            else:
                ts.domain_loop(0, num_k_tiles, 1, loop_body)

    @ts.schedule
    def direct_load_b_schedule(stage_info, gmem, smem):
        load_b_schedule_body(stage_info, gmem, smem)

    @ts.schedule
    def persistent_load_b_schedule(stage_info, gmem, smem, wq):
        load_b_schedule_body(stage_info, gmem, smem, wq)

    load_b_schedule = (
        persistent_load_b_schedule if cfg.is_persistent else direct_load_b_schedule
    )
    resources = (gmem_b, smem_b, work_queue) if cfg.is_persistent else (gmem_b, smem_b)
    return ts.Task(
        name="LoadBTask",
        warp_idx=cfg.load_b_warp_idx,
        num_warps=cfg.num_load_b_warps,
        num_registers=cfg.load_b_task_regs,
        schedule=load_b_schedule(*resources),
    )


def create_mma_task(
    cfg, smem_a, smem_b, tmem_c, work_queue, num_k_tiles, a_offset, b_offset
):
    mma_domain_unroll = _mma_domain_unroll(cfg)

    def mma_schedule_body(stage_info, smem_a_res, smem_b_res, tmem_c_res, wq=None):
        smem_a_buf = smem_a_res.init_mma_state(a_offset)
        smem_b_buf = smem_b_res.init_mma_state(b_offset)
        tmem_raw_addr, idesc = tmem_c_res.init_accumulator_state(cfg=cfg)
        with _work_tile_schedule_loop(cfg, wq):
            tmem_c_res.try_acquire()
            tmem_c_res.acquire()

            def _mma_loop_body():
                smem_a_res.try_wait()
                smem_b_res.try_wait()
                smem_a_res.wait()
                smem_b_res.wait()
                desc_a_mma_base, smem_a_stage_ptr = smem_a_res.build_mma_desc_a(
                    smem_a_buf, cfg=cfg
                )
                desc_b_mma_base, smem_b_stage_ptr = smem_b_res.build_mma_desc_b(
                    smem_b_buf, cfg=cfg
                )
                tmem_c_res.mma(
                    tmem_raw_addr,
                    idesc,
                    desc_a_mma_base=desc_a_mma_base,
                    smem_a_stage_ptr=smem_a_stage_ptr,
                    desc_b_mma_base=desc_b_mma_base,
                    smem_b_stage_ptr=smem_b_stage_ptr,
                    cfg=cfg,
                )
                smem_a_res.release()
                smem_b_res.release()

            if mma_domain_unroll:
                ts.domain_loop(
                    0, num_k_tiles, 1, _mma_loop_body, unroll=mma_domain_unroll
                )
            else:
                ts.domain_loop(0, num_k_tiles, 1, _mma_loop_body)
            tmem_c_res.commit()

    @ts.schedule
    def direct_mma_schedule(stage_info, smem_a_res, smem_b_res, tmem_c_res):
        mma_schedule_body(stage_info, smem_a_res, smem_b_res, tmem_c_res)

    @ts.schedule
    def persistent_mma_schedule(stage_info, smem_a_res, smem_b_res, tmem_c_res, wq):
        mma_schedule_body(stage_info, smem_a_res, smem_b_res, tmem_c_res, wq)

    mma_schedule = persistent_mma_schedule if cfg.is_persistent else direct_mma_schedule
    resources = (smem_a, smem_b, tmem_c)
    if cfg.is_persistent:
        resources += (work_queue,)
    return ts.Task(
        name="MmaTask0",
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.num_mma_warps,
        num_registers=cfg.mma_regs,
        schedule=mma_schedule(*resources),
        run_only_on_cta_id=0 if cfg.has_cluster else None,
    )


def create_epilogue_task(cfg, tmem_c, gmem_c, work_queue, num_k_tiles):
    if cfg.is_swap_ab:
        epi_warpgroup_count = max(1, cfg.num_epilogue_warps // 4)
        epi_cols_per_call = cfg.epi_tile_n * epi_warpgroup_count
        epi_subtile_cnt = max(1, cfg.tile_n // epi_cols_per_call)
    else:
        epi_t2r_repx = cfg.epi_tile_n // 4
        epi_subtile_cnt = max(1, cfg.tile_n // epi_t2r_repx)

    def epilogue_schedule_body(stage_info, tmem, gmem, wq=None):
        tmem_raw_addr = tmem.init_epilogue_state()
        gmem.init_store_state()
        with _work_tile_schedule_loop(cfg, wq):
            tile_expert_idx, tile_token_limit = gmem.init_epilogue_tile_state(cfg=cfg)
            with ts.domain_loop(0, num_k_tiles, 1):
                pass
            tmem.try_wait()
            tmem.wait()
            for subtile_idx in range(epi_subtile_cnt):
                t2r_rmem, t2r_rmem_1, t2r_output_call_idx = tmem.consumer_work(
                    tmem_raw_addr, subtile_idx=subtile_idx, cfg=cfg
                )
                gmem.store_epilogue(
                    tile_expert_idx,
                    tile_token_limit,
                    t2r_rmem=t2r_rmem,
                    t2r_rmem_1=t2r_rmem_1,
                    t2r_output_call_idx=t2r_output_call_idx,
                    subtile_idx=subtile_idx,
                    cfg=cfg,
                )
            tmem.release()

    @ts.schedule
    def direct_epilogue_schedule(stage_info, tmem, gmem):
        epilogue_schedule_body(stage_info, tmem, gmem)

    @ts.schedule
    def persistent_epilogue_schedule(stage_info, tmem, gmem, wq):
        epilogue_schedule_body(stage_info, tmem, gmem, wq)

    epilogue_schedule = (
        persistent_epilogue_schedule if cfg.is_persistent else direct_epilogue_schedule
    )
    resources = (tmem_c, gmem_c, work_queue) if cfg.is_persistent else (tmem_c, gmem_c)
    return ts.Task(
        name="EpilogueTask0",
        warp_idx=cfg.epilogue_warp_idx,
        num_warps=cfg.num_epilogue_warps,
        num_registers=cfg.epilogue_regs,
        schedule=epilogue_schedule(*resources),
    )


def create_workid_task(cfg, work_queue):
    @ts.schedule
    def workid_schedule(stage_info, wq):
        with ts.work_tile_loop(wq):
            wq.try_acquire()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            _persistent_tail(wq)

    return ts.Task(
        name="WorkScheduleTask",
        warp_idx=cfg.workid_warp_idx,
        num_warps=cfg.num_workid_warps,
        num_registers=cfg.workid_regs,
        schedule=workid_schedule(work_queue),
    )


def create_padding_task(cfg, work_queue, num_k_tiles):
    def padding_schedule_body(stage_info, wq=None):
        # Keep the source's work-tile/domain nesting explicit during capture.
        with _work_tile_schedule_loop(cfg, wq):  # noqa: SIM117
            with ts.domain_loop(0, num_k_tiles, 1):
                pass

    @ts.schedule
    def direct_padding_schedule(stage_info):
        padding_schedule_body(stage_info)

    @ts.schedule
    def persistent_padding_schedule(stage_info, wq):
        padding_schedule_body(stage_info, wq)

    padding_schedule = (
        persistent_padding_schedule if cfg.is_persistent else direct_padding_schedule
    )
    resources = (work_queue,) if cfg.is_persistent else ()
    return ts.Task(
        name="PaddingTask",
        warp_idx=cfg.padding_warp_idx,
        num_warps=cfg.num_padding_warps,
        num_registers=cfg.padding_regs,
        schedule=padding_schedule(*resources),
    )
