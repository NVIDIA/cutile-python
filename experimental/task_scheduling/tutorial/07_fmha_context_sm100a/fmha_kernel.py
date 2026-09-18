# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""SM100 query/head-paired D128 FMHA kernel and task-manager builder."""

from dataclasses import dataclass, replace

import cuda.lang as cl
import torch

import task_scheduling as ts

try:
    from .fmha_resources import (
        CLC_RESPONSE_BYTES,
        RAGGED_TMA_DIM_MAX,
        RAGGED_XLARGE_N,
        SCHEDULER_STAGES,
        SMEM_ALIGNMENT,
        SUPPORTED_PAGE_SIZES,
        FmhaConfig,
        GmemOResource,
        GmemQKVResource,
        S0S1SequenceResource,
        SmemKVResource,
        SmemOResource,
        SmemQResource,
        TmemStatsResource,
        TmemOResource,
        TmemSPResource,
        bottom_right_window_max_tiles,
        bottom_right_window_tile_start,
    )
    from .fmha_tasks import (
        PackedContextWorkQueue,
        create_correction_task,
        create_epilogue_task,
        create_load_task,
        create_mma_task,
        create_padding_task,
        create_scheduler_task,
        create_softmax_task,
    )
except ImportError:
    from fmha_resources import (
        CLC_RESPONSE_BYTES,
        RAGGED_TMA_DIM_MAX,
        RAGGED_XLARGE_N,
        SCHEDULER_STAGES,
        SMEM_ALIGNMENT,
        SUPPORTED_PAGE_SIZES,
        FmhaConfig,
        GmemOResource,
        GmemQKVResource,
        S0S1SequenceResource,
        SmemKVResource,
        SmemOResource,
        SmemQResource,
        TmemStatsResource,
        TmemOResource,
        TmemSPResource,
        bottom_right_window_max_tiles,
        bottom_right_window_tile_start,
    )
    from fmha_tasks import (
        PackedContextWorkQueue,
        create_correction_task,
        create_epilogue_task,
        create_load_task,
        create_mma_task,
        create_padding_task,
        create_scheduler_task,
        create_softmax_task,
    )


@dataclass(frozen=True)
class TasksInputs:
    tma_q_desc: object
    tma_k_desc: object
    tma_v_desc: object
    tma_o_desc: object
    scale_softmax_log2: object
    output_scale: object
    tmem_base: object
    num_kv_tiles: object
    q_offset: object


@dataclass(frozen=True)
class RaggedTasksInputs(TasksInputs):
    cum_seqlen_q: object
    cum_seqlen_k: object


@dataclass(frozen=True)
class PagedTasksInputs(RaggedTasksInputs):
    page_idx_kv: object


@dataclass(frozen=True)
class _ResolvedCausalDomainTask:
    num_kv_tiles: object
    q_offset: object
    cum_seqlen_q: object
    cum_seqlen_k: object
    cta_m: int
    kv_n: int
    seq_idx: int
    batch_idx: int
    has_varlen: bool
    reverse_seq_tiles: int
    offset: int
    window_size_left: int
    packed_window: bool
    runtime_kv_tile_multiple: int


@dataclass(frozen=True)
class _DeviceCausalDomainTask:
    cta_m: int
    kv_n: int
    seq_idx: int
    batch_idx: int
    has_varlen: bool
    reverse_seq_tiles: int
    offset: int
    window_size_left: int
    packed_window: bool
    runtime_kv_tile_multiple: int

    def bind_inputs(self, tasks_inputs):
        return _ResolvedCausalDomainTask(
            tasks_inputs.num_kv_tiles,
            tasks_inputs.q_offset,
            (tasks_inputs.cum_seqlen_q if self.has_varlen else None),
            (tasks_inputs.cum_seqlen_k if self.has_varlen else None),
            self.cta_m,
            self.kv_n,
            self.seq_idx,
            self.batch_idx,
            self.has_varlen,
            self.reverse_seq_tiles,
            self.offset,
            self.window_size_left,
            self.packed_window,
            self.runtime_kv_tile_multiple,
        )


class CausalDomainTask(ts.Task):
    """Task with a per-work-tile causal/window K/V loop domain."""

    def __init__(
        self,
        *args,
        num_kv_tiles,
        q_offset,
        cta_m,
        kv_n,
        seq_idx,
        batch_idx,
        has_varlen,
        reverse_seq_tiles=0,
        offset=0,
        window_size_left=0,
        packed_window=False,
        runtime_kv_tile_multiple=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_kv_tiles = num_kv_tiles
        self.q_offset = q_offset
        self.cum_seqlen_q = None
        self.cum_seqlen_k = None
        self.cta_m = cta_m
        self.kv_n = kv_n
        self.seq_idx = seq_idx
        self.batch_idx = batch_idx
        self.has_varlen = has_varlen
        self.reverse_seq_tiles = reverse_seq_tiles
        self.offset = offset
        self.window_size_left = window_size_left
        self.packed_window = packed_window
        self.runtime_kv_tile_multiple = runtime_kv_tile_multiple

    def _freeze_domain_task(self):
        return _DeviceCausalDomainTask(
            self.cta_m,
            self.kv_n,
            self.seq_idx,
            self.batch_idx,
            self.has_varlen,
            self.reverse_seq_tiles,
            self.offset,
            self.window_size_left,
            self.packed_window,
            self.runtime_kv_tile_multiple,
        )

    def get_domain(self, tile_coord):
        seq_coord = tile_coord[self.seq_idx]
        if self.reverse_seq_tiles:
            seq_coord = self.reverse_seq_tiles - seq_coord - 1
        num_kv_tiles = self.num_kv_tiles
        q_offset = self.q_offset
        q_tile_active = None
        if self.has_varlen and self.cum_seqlen_q is not None:
            batch_coord = tile_coord[self.batch_idx]
            q_begin = self.cum_seqlen_q[batch_coord]
            k_begin = self.cum_seqlen_k[batch_coord]
            seqlen_q = self.cum_seqlen_q[batch_coord + 1] - q_begin
            seqlen_k = self.cum_seqlen_k[batch_coord + 1] - k_begin
            q_tile_active = seq_coord * self.cta_m < seqlen_q
            q_offset = seqlen_k - seqlen_q
            num_kv_tiles = (seqlen_k + self.kv_n - 1) // self.kv_n
            if self.runtime_kv_tile_multiple > 1:
                num_kv_tiles = (
                    (num_kv_tiles + self.runtime_kv_tile_multiple - 1)
                    // self.runtime_kv_tile_multiple
                    * self.runtime_kv_tile_multiple
                )
        if self.packed_window:
            result = cl.minimum(
                num_kv_tiles,
                bottom_right_window_max_tiles(
                    self.cta_m,
                    self.kv_n,
                    self.window_size_left,
                ),
            )
        else:
            max_q_row = q_offset + seq_coord * self.cta_m + self.cta_m - 1
            causal_n = max_q_row // self.kv_n + 1
            result = cl.maximum(
                self.cta_m // self.kv_n,
                cl.minimum(num_kv_tiles, causal_n),
            )
        if self.window_size_left > 0 and not self.packed_window:
            result -= bottom_right_window_tile_start(
                seq_coord,
                self.cta_m,
                self.kv_n,
                q_offset,
                self.window_size_left,
            )
        result -= self.offset
        if q_tile_active is not None:
            minimum_domain = max(self.cta_m // self.kv_n - self.offset, 0)
            result = q_tile_active * result + (1 - q_tile_active) * minimum_domain
        return result


class CausalSoftmaxDomainTask(CausalDomainTask):
    """Marker subclass for causal softmax domains."""


@dataclass(frozen=True)
class FmhaDomainPolicy:
    domain_n_kwargs: dict
    domain_n_minus_1_kwargs: dict
    softmax0_domain_kwargs: dict
    softmax1_domain_kwargs: dict


@dataclass(frozen=True)
class FmhaPipeline:
    """Host schedule plus the allocations needed by the launch wrapper."""

    cfg: FmhaConfig
    task_manager: object
    device_task_manager: object
    work_queue: object | None
    tasks: tuple[object, ...]
    resources: tuple[object, ...]
    q_allocation: object
    kv_allocation: object
    tmem_vec0_allocation: object
    tmem_vec1_allocation: object
    o0_allocation: object
    o1_allocation: object
    tmem_ptr_allocation: object
    clc_response_allocation: object | None
    num_kv_tiles: int
    q_offset: int
    problem_shape: tuple[int, int, int]
    num_heads_q: int


def _make_tma_pipeline(num_stages, num_bytes):
    return ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=num_stages,
        num_bytes=num_bytes,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(1),
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
        consumer_signaling_threads=ts.SignalingThreads.CtaLeader,
        advance_on_wait=True,
        tcgen05_fence_after_wait=False,
    )


def _make_umma_async_pipeline(num_stages, consumers):
    return ts.PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=num_stages,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(consumers),
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.CtaLeader,
        consumer_signaling_threads=ts.SignalingThreads.All,
        tcgen05_fence_after_wait=False,
    )


def _make_async_pipeline(num_stages, producers, consumers):
    return ts.PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=num_stages,
        producer_group=ts.CooperativeGroup(producers),
        consumer_group=ts.CooperativeGroup(consumers),
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.All,
        consumer_signaling_threads=ts.SignalingThreads.All,
    )


def _validate_geometry(batch, seq_q, seq_k, heads, fmha_config):
    values = {"batch": batch, "seq_q": seq_q, "seq_k": seq_k, "heads": heads}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError(f"FMHA geometry must contain positive integers, got {values}")
    if heads % fmha_config.h_r:
        raise ValueError(
            f"Q heads ({heads}) must be divisible by h_r ({fmha_config.h_r})"
        )
    if heads % fmha_config.work_tile_q_heads:
        raise ValueError(
            f"Q heads ({heads}) must be divisible by the head-paired tile width "
            f"({fmha_config.work_tile_q_heads})"
        )
    if fmha_config.is_causal and not fmha_config.has_varlen and seq_q > seq_k:
        raise ValueError("bottom-right causal attention requires seq_q <= seq_k")
    if fmha_config.use_paged_kv:
        if not fmha_config.has_varlen:
            raise ValueError("paged K/V attention requires has_varlen=True")
        page_size = fmha_config.num_tokens_per_page
        if page_size not in SUPPORTED_PAGE_SIZES:
            raise ValueError(
                f"paged K/V page size must be one of {SUPPORTED_PAGE_SIZES}, "
                f"got {page_size}"
            )
        pages_per_tile = fmha_config.kv_tile_n // page_size
        max_pages = fmha_config.max_num_pages_per_seq_kv
        if type(max_pages) is not int or max_pages <= 0:
            raise ValueError("max_num_pages_per_seq_kv must be a positive integer")
        if max_pages % pages_per_tile:
            raise ValueError(
                "max_num_pages_per_seq_kv must be a multiple of pages per K/V "
                f"tile ({pages_per_tile})"
            )


def _causal_domain_kwargs(
    cfg,
    *,
    num_kv_tiles,
    q_offset,
    offset,
    cta_m=None,
    runtime_kv_tile_multiple=1,
):
    return {
        "task_class": CausalDomainTask,
        "num_kv_tiles": num_kv_tiles,
        "q_offset": q_offset,
        "cta_m": cfg.cta_tiler[0] if cta_m is None else cta_m,
        "kv_n": cfg.kv_tile_n,
        "seq_idx": cfg.work_tile_coord_indices[0],
        "batch_idx": cfg.work_tile_coord_indices[2],
        "has_varlen": cfg.has_varlen,
        "reverse_seq_tiles": (
            cfg.num_seq_tiles
            if cfg.uses_causal_reversed_head_batch_seq_tile_order
            else 0
        ),
        "offset": offset,
        "window_size_left": cfg.window_size_left,
        "packed_window": cfg.head_paired
        and cfg.has_varlen
        and cfg.window_size_left > 0,
        "runtime_kv_tile_multiple": runtime_kv_tile_multiple,
    }


def _select_fmha_domain_policy(cfg, num_kv_tiles, q_offset):
    if not cfg.is_causal:
        domain_n = {"domain": num_kv_tiles}
        domain_n_minus_1 = {"domain": num_kv_tiles - 1}
        return FmhaDomainPolicy(
            domain_n,
            domain_n_minus_1,
            domain_n,
            domain_n,
        )

    if cfg.head_paired:
        domain_n = _causal_domain_kwargs(
            cfg,
            num_kv_tiles=num_kv_tiles,
            q_offset=q_offset,
            offset=0,
            cta_m=cfg.q_tile_m,
        )
        domain_n_minus_1 = _causal_domain_kwargs(
            cfg,
            num_kv_tiles=num_kv_tiles,
            q_offset=q_offset,
            offset=1,
            cta_m=cfg.q_tile_m,
        )
        softmax = {
            **domain_n_minus_1,
            "task_class": CausalSoftmaxDomainTask,
        }
        return FmhaDomainPolicy(
            domain_n,
            domain_n_minus_1,
            softmax,
            softmax,
        )

    if cfg.causal_single_kv_tile:
        domain_n = {"domain": 1}
        domain_n_minus_1 = {"domain": 0}
        return FmhaDomainPolicy(
            domain_n,
            domain_n_minus_1,
            domain_n_minus_1,
            domain_n_minus_1,
        )

    runtime_multiple = (
        (cfg.cta_tiler[0] + cfg.kv_tile_n - 1) // cfg.kv_tile_n
        if cfg.skip_causal_invalid_peer0
        else 1
    )
    domain_n = _causal_domain_kwargs(
        cfg,
        num_kv_tiles=num_kv_tiles,
        q_offset=q_offset,
        offset=0,
        runtime_kv_tile_multiple=runtime_multiple,
    )
    domain_n_minus_1 = _causal_domain_kwargs(
        cfg,
        num_kv_tiles=num_kv_tiles,
        q_offset=q_offset,
        offset=1,
        runtime_kv_tile_multiple=runtime_multiple,
    )
    domain_n_minus_2 = _causal_domain_kwargs(
        cfg,
        num_kv_tiles=num_kv_tiles,
        q_offset=q_offset,
        offset=2,
        runtime_kv_tile_multiple=runtime_multiple,
    )
    softmax0_base = (
        domain_n_minus_2 if cfg.skip_causal_invalid_peer0 else domain_n_minus_1
    )
    return FmhaDomainPolicy(
        domain_n,
        domain_n_minus_1,
        {**softmax0_base, "task_class": CausalSoftmaxDomainTask},
        {**domain_n_minus_1, "task_class": CausalSoftmaxDomainTask},
    )


def build_fmha_task_manager(
    batch,
    seq_q,
    seq_k,
    heads,
    cfg=FmhaConfig(),
    *,
    verbose=False,
    exhaustive_deadlock_race_check=False,
):
    """Build and freeze the requested FMHA task graph."""
    _validate_geometry(
        batch,
        seq_q,
        seq_k,
        heads,
        cfg,
    )
    num_q_tiles = (seq_q + cfg.cta_tiler[0] - 1) // cfg.cta_tiler[0]
    num_kv_tiles = (seq_k + cfg.kv_tile_n - 1) // cfg.kv_tile_n
    # Packed causal plans only carry a live per-request Q/K offset when the
    # caller opts into it. Equal Q/K packed plans can then retain the
    # query-paired invalid-tail fast path.
    # Fixed inputs still expose their static bottom-right offset automatically.
    has_q_offset = cfg.is_causal and (
        cfg.has_q_offset or (not cfg.has_varlen and seq_q != seq_k)
    )
    cfg = replace(
        cfg,
        num_seq_tiles=num_q_tiles,
        balance_causal_workload=(
            cfg.balance_causal_workload or (cfg.is_causal and cfg.is_clc_dynamic)
        ),
        has_q_offset=has_q_offset,
        causal_single_kv_tile=(
            cfg.is_causal
            and not cfg.head_paired
            and not cfg.has_varlen
            and num_kv_tiles == 1
        ),
        fixed_dense_k_tail=(
            seq_k % cfg.kv_tile_n if not cfg.is_causal and not cfg.has_varlen else 0
        ),
    )
    q_offset = seq_k - seq_q if cfg.is_causal and not cfg.has_varlen else 0
    domain_policy = _select_fmha_domain_policy(cfg, num_kv_tiles, q_offset)

    # -----------------------------------------------------------------------
    # SMEM / TMEM allocations
    # -----------------------------------------------------------------------
    q_allocation = ts.SmemAllocation(
        "smem_q",
        cfg.q_stage * cfg.tma_copy_q_bytes,
        alignment=cfg.buffer_align_bytes,
    )
    kv_allocation = ts.SmemAllocation(
        "smem_kv",
        cfg.kv_stage * cfg.tma_copy_kv_bytes,
        alignment=cfg.buffer_align_bytes,
    )
    stats_bytes = cfg.q_tile_m * 2 * 4
    tmem_vec0_allocation = ts.SmemAllocation(
        "smem_vec_0",
        stats_bytes,
        alignment=16,
    )
    tmem_vec1_allocation = ts.SmemAllocation(
        "smem_vec_128",
        stats_bytes,
        alignment=16,
    )
    o_stage_bytes = cfg.sO_stage_elements * 2
    o0_allocation = ts.SmemAllocation(
        "smem_o_0",
        o_stage_bytes,
        alignment=cfg.buffer_align_bytes,
    )
    o1_allocation = ts.SmemAllocation(
        "smem_o_1",
        o_stage_bytes,
        alignment=cfg.buffer_align_bytes,
    )
    tmem_ptr_allocation = ts.SmemAllocation("tmem_ptr", 4, alignment=4)
    clc_response_allocation = None
    if cfg.is_clc_dynamic:
        clc_response_allocation = ts.SmemAllocation(
            "clc_response",
            SCHEDULER_STAGES * CLC_RESPONSE_BYTES,
            alignment=16,
            count=SCHEDULER_STAGES,
        )

    sp_tmem_columns = cfg.qk_mma_tiler[1] * cfg.mma_softmax_stage
    sp0_tmem = ts.TmemAllocation("tmem_sp_0", sp_tmem_columns)
    sp1_tmem = ts.TmemAllocation("tmem_sp_1", sp_tmem_columns)
    o_tmem = ts.TmemAllocation(
        "tmem_o",
        cfg.num_qkv_instances * cfg.epi_tile[1],
    )

    # -----------------------------------------------------------------------
    # Cooperative groups and pipeline configs
    # -----------------------------------------------------------------------
    num_softmax_threads = len(cfg.softmax0_warp_ids) * 32
    num_correction_threads = len(cfg.correction_warp_ids) * 32
    smem_q_pipeline_cfg = _make_tma_pipeline(
        cfg.q_stage,
        cfg.tma_copy_q_bytes,
    )
    smem_kv_pipeline_cfg = _make_tma_pipeline(
        cfg.kv_stage,
        cfg.tma_copy_kv_bytes,
    )
    tmem_sp0_pipeline_cfg = _make_umma_async_pipeline(
        cfg.mma_softmax_stage,
        num_softmax_threads,
    )
    tmem_sp1_pipeline_cfg = _make_umma_async_pipeline(
        cfg.mma_softmax_stage,
        num_softmax_threads,
    )
    tmem_vec0_pipeline_cfg = _make_async_pipeline(
        cfg.softmax_corr_stage,
        num_softmax_threads,
        num_correction_threads,
    )
    tmem_vec1_pipeline_cfg = _make_async_pipeline(
        cfg.softmax_corr_stage,
        num_softmax_threads,
        num_correction_threads,
    )
    tmem_o_pipeline_cfg = _make_umma_async_pipeline(
        cfg.mma_corr_stage,
        num_correction_threads,
    )
    smem_o_0_pipeline_cfg = _make_async_pipeline(
        1,
        num_correction_threads,
        32,
    )
    smem_o_1_pipeline_cfg = _make_async_pipeline(
        1,
        num_correction_threads,
        32,
    )
    s0s1_seq_pipeline_cfg = _make_async_pipeline(
        1,
        num_softmax_threads,
        num_softmax_threads,
    )

    # -----------------------------------------------------------------------
    # Create resource instances
    # -----------------------------------------------------------------------
    gmem_qkv = GmemQKVResource(name="gmem_qkv")
    smem_q = SmemQResource(
        name="smem_q",
        pipeline_config=smem_q_pipeline_cfg,
        smem_requirements=[q_allocation],
    )
    smem_kv = SmemKVResource(
        name="smem_kv",
        pipeline_config=smem_kv_pipeline_cfg,
        smem_requirements=[kv_allocation],
    )
    tmem_sp0 = TmemSPResource(
        name="tmem_sp0",
        pipeline_config=tmem_sp0_pipeline_cfg,
        tmem_requirements=[sp0_tmem],
    )
    tmem_sp1 = TmemSPResource(
        name="tmem_sp1",
        pipeline_config=tmem_sp1_pipeline_cfg,
        tmem_requirements=[sp1_tmem],
    )
    tmem_vec0 = TmemStatsResource(
        name="tmem_vec0",
        pipeline_config=tmem_vec0_pipeline_cfg,
        smem_requirements=[tmem_vec0_allocation],
    )
    tmem_vec1 = TmemStatsResource(
        name="tmem_vec1",
        pipeline_config=tmem_vec1_pipeline_cfg,
        smem_requirements=[tmem_vec1_allocation],
    )
    tmem_o = TmemOResource(
        name="tmem_o",
        pipeline_config=tmem_o_pipeline_cfg,
        tmem_requirements=[o_tmem],
    )
    smem_o0 = SmemOResource(
        name="smem_o_0",
        pipeline_config=smem_o_0_pipeline_cfg,
        smem_requirements=[o0_allocation],
    )
    smem_o1 = SmemOResource(
        name="smem_o_1",
        pipeline_config=smem_o_1_pipeline_cfg,
        smem_requirements=[o1_allocation],
    )
    gmem_o0 = GmemOResource(name="gmem_o_0")
    gmem_o1 = GmemOResource(name="gmem_o_1")
    s0s1_seq = S0S1SequenceResource(
        name="s0s1_seq",
        is_barrier=True,
        pipeline_config=s0s1_seq_pipeline_cfg,
    )

    num_head_tiles = heads // cfg.work_tile_q_heads
    problem_shape = (
        (num_head_tiles, batch, num_q_tiles)
        if cfg.uses_head_batch_seq_tile_order
        else (num_q_tiles, num_head_tiles, batch)
    )
    work_queue = None
    if cfg.is_clc_dynamic:
        scheduler_params = ts.ClcDynamicPersistentTileSchedulerParams(
            problem_shape,
            cfg.cluster_shape,
        )
        work_queue_type = (
            PackedContextWorkQueue
            if cfg.is_causal and cfg.has_varlen and not cfg.has_uniform_varlen
            else ts.WorkQueue
        )
        work_queue_kwargs = dict(
            name="work_queue",
            pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
                num_stages=SCHEDULER_STAGES,
                num_bytes=CLC_RESPONSE_BYTES,
                producer_group=ts.CooperativeGroup(1),
                consumer_group=ts.CooperativeGroup(cfg.block_threads),
                cta_layout_vmnk=(1, 1, 1, 1),
                producer_signaling_threads=ts.SignalingThreads.CtaLeader,
                consumer_signaling_threads=ts.SignalingThreads.All,
            ),
            tile_scheduler_config=(
                ts.TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    scheduler_params,
                    clc_response_allocation,
                )
            ),
            smem_requirements=[clc_response_allocation],
        )
        work_queue = (
            work_queue_type(cfg=cfg, **work_queue_kwargs)
            if work_queue_type is PackedContextWorkQueue
            else work_queue_type(**work_queue_kwargs)
        )
    elif cfg.is_persistent:
        scheduler_params = ts.PersistentTileSchedulerParams(
            problem_shape,
            cfg.cluster_shape,
        )
        work_queue_type = (
            PackedContextWorkQueue
            if cfg.is_causal and cfg.has_varlen and not cfg.has_uniform_varlen
            else ts.WorkQueue
        )
        work_queue_kwargs = dict(
            name="work_queue",
            tile_scheduler_config=(
                ts.TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                    scheduler_params
                )
            ),
        )
        work_queue = (
            work_queue_type(cfg=cfg, **work_queue_kwargs)
            if work_queue_type is PackedContextWorkQueue
            else work_queue_type(**work_queue_kwargs)
        )

    smem_allocator = ts.SmemAllocator(default_add_barriers=False)
    for resource in (
        smem_q,
        smem_kv,
        tmem_vec0,
        tmem_vec1,
        smem_o0,
        smem_o1,
    ):
        smem_allocator.add_resource(resource)
    if work_queue is not None and work_queue.pipeline_config is not None:
        smem_allocator.add_resource(work_queue)
    smem_allocator.add(tmem_ptr_allocation)
    smem_allocator.compute_layout()

    tmem_allocator = ts.TmemAllocator()
    for resource in (tmem_sp0, tmem_sp1, tmem_o):
        tmem_allocator.add_resource(resource)
    tmem_allocator.compute_layout()
    actual_offsets = (sp0_tmem.offset, sp1_tmem.offset, o_tmem.offset)
    expected_offsets = (
        cfg.tmem_s0_offset,
        cfg.tmem_s1_offset,
        cfg.tmem_o0_offset,
    )
    if actual_offsets != expected_offsets:
        raise ValueError(
            "unexpected TMEM map; expected "
            f"S0/S1/O={expected_offsets}, got {actual_offsets}"
        )

    load_task = create_load_task(
        gmem_qkv,
        smem_q,
        smem_kv,
        work_queue,
        fmha_config=cfg,
        q_smem_offset=q_allocation.offset,
        kv_smem_offset=kv_allocation.offset,
        **domain_policy.domain_n_kwargs,
    )
    mma_task = create_mma_task(
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_o,
        work_queue,
        fmha_config=cfg,
        q_smem_offset=q_allocation.offset,
        kv_smem_offset=kv_allocation.offset,
        **domain_policy.domain_n_minus_1_kwargs,
    )
    softmax0_task = create_softmax_task(
        0,
        tmem_sp0,
        tmem_vec0,
        s0s1_seq,
        work_queue,
        fmha_config=cfg,
        tmem_vec_smem_offset=tmem_vec0_allocation.offset,
        **domain_policy.softmax0_domain_kwargs,
    )
    softmax1_task = create_softmax_task(
        1,
        tmem_sp1,
        tmem_vec1,
        s0s1_seq,
        work_queue,
        fmha_config=cfg,
        tmem_vec_smem_offset=tmem_vec1_allocation.offset,
        **domain_policy.softmax1_domain_kwargs,
    )
    correction_task = create_correction_task(
        tmem_vec0,
        tmem_vec1,
        tmem_o,
        smem_o0,
        smem_o1,
        work_queue,
        fmha_config=cfg,
        tmem_vec0_smem_offset=tmem_vec0_allocation.offset,
        tmem_vec1_smem_offset=tmem_vec1_allocation.offset,
        o0_smem_offset=o0_allocation.offset,
        o1_smem_offset=o1_allocation.offset,
        **domain_policy.domain_n_minus_1_kwargs,
    )
    epilogue_task = create_epilogue_task(
        smem_o0,
        smem_o1,
        gmem_o0,
        gmem_o1,
        work_queue,
        fmha_config=cfg,
        o0_smem_offset=o0_allocation.offset,
        o1_smem_offset=o1_allocation.offset,
        **domain_policy.domain_n_kwargs,
    )
    auxiliary_task = (
        create_scheduler_task(work_queue, fmha_config=cfg)
        if cfg.is_clc_dynamic
        else create_padding_task(
            work_queue,
            fmha_config=cfg,
            **domain_policy.domain_n_kwargs,
        )
    )
    tasks = (
        softmax0_task,
        softmax1_task,
        correction_task,
        mma_task,
        load_task,
        epilogue_task,
        auxiliary_task,
    )

    def scheduler_deps(*resources):
        deps = list(resources)
        if work_queue is not None:
            deps.append(work_queue)
        return deps

    resource_dependency_graph = {
        smem_q: scheduler_deps(gmem_qkv),
        smem_kv: scheduler_deps(gmem_qkv),
        tmem_sp0: scheduler_deps(tmem_sp0, smem_q, smem_kv),
        tmem_sp1: scheduler_deps(tmem_sp1, smem_q, smem_kv),
        tmem_vec0: scheduler_deps(tmem_sp0),
        tmem_vec1: scheduler_deps(tmem_sp1),
        tmem_o: scheduler_deps(tmem_sp0, tmem_sp1),
        smem_o0: scheduler_deps(tmem_vec0, tmem_o),
        smem_o1: scheduler_deps(tmem_vec1, tmem_o),
        gmem_o0: scheduler_deps(smem_o0),
        gmem_o1: scheduler_deps(smem_o1),
        s0s1_seq: [tmem_sp0],
    }
    if work_queue is not None:
        resource_dependency_graph[work_queue] = (
            [work_queue] if cfg.is_clc_dynamic else []
        )

    barrier_allocator = ts.BarrierAllocator()
    for resource in (
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_vec0,
        tmem_vec1,
        tmem_o,
        smem_o0,
        smem_o1,
        s0s1_seq,
    ):
        barrier_allocator.add_resource(resource)
    if work_queue is not None and work_queue.pipeline_config is not None:
        barrier_allocator.add_resource(work_queue)
    barrier_allocator.compute_layout()

    task_manager = ts.TaskManager(
        tasks=list(tasks),
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        barrier_allocator=barrier_allocator,
        verbose=verbose,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
        exhaustive_representative_domain=True,
        cta_warps=cfg.block_warps,
    )
    device_task_manager = task_manager.to_device()
    # Data and pipeline barriers use adjacent dynamic-SMEM arrays. Carrying the
    # buffers' 1024-byte alignment onto the first array would add 896 bytes of
    # trailing padding before the barrier array and exceed SM100's launch
    # limit.  The individual allocation offsets remain 1024-byte aligned; use
    # the ISA-required 128-byte arena alignment to retain the packed layout.
    device_task_manager = replace(
        device_task_manager,
        smem_alignment=SMEM_ALIGNMENT,
    )
    resources = (
        gmem_qkv,
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_vec0,
        tmem_vec1,
        tmem_o,
        smem_o0,
        smem_o1,
        gmem_o0,
        gmem_o1,
        s0s1_seq,
    )
    if work_queue is not None:
        resources += (work_queue,)
    return FmhaPipeline(
        cfg,
        task_manager,
        device_task_manager,
        work_queue,
        tasks,
        resources,
        q_allocation,
        kv_allocation,
        tmem_vec0_allocation,
        tmem_vec1_allocation,
        o0_allocation,
        o1_allocation,
        tmem_ptr_allocation,
        clc_response_allocation,
        num_kv_tiles,
        q_offset,
        problem_shape,
        heads,
    )


def build_fmha_pipeline(*args, **kwargs):
    """Compatibility alias for the original CUDA Lang tutorial API."""
    return build_fmha_task_manager(*args, **kwargs)


def _make_ragged_tma_array(tensor, heads, fmha_config):
    """Build the synthetic rank-5 array used by the Q/O maps."""
    head_dim = fmha_config.qk_mma_tiler[2]
    packed_row_stride = heads * head_dim
    return cl.Array.from_parts(
        tensor.pointer(),
        (
            head_dim,
            heads,
            fmha_config.q_tile_m,
            RAGGED_TMA_DIM_MAX,
            RAGGED_TMA_DIM_MAX,
        ),
        (
            1,
            head_dim,
            packed_row_stride,
            RAGGED_XLARGE_N - packed_row_stride,
            packed_row_stride,
        ),
    )


def make_fmha_kernel(device_task_manager, num_kv_tiles, q_offset, cfg, heads):
    """Lower one frozen fixed or packed-ragged task manager."""
    fmha_config = cfg

    def kernel_body(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        scale_softmax_log2,
        output_scale,
        cum_seqlen_q=None,
        cum_seqlen_k=None,
        page_idx_kv=None,
    ):
        warp_idx, _ = cl.shuffle_sync(cl.ShuffleKind.INDEX, cl.thread_index(0) // 32, 0)
        if warp_idx == fmha_config.load_warp_id and cl.elect_sync():
            cl.prefetch_tensor_map(tma_q_desc)
            cl.prefetch_tensor_map(tma_k_desc)
            cl.prefetch_tensor_map(tma_v_desc)
            cl.prefetch_tensor_map(tma_o_desc)

        allocators = device_task_manager.setup_resources_and_tasks()
        tmem_ptr = allocators.smem_allocator.get(
            "tmem_ptr",
            cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        )
        if warp_idx == fmha_config.mma_warp_id:
            cl.tcgen05_allocate(
                tmem_ptr.pointer(),
                fmha_config.tmem_alloc_cols,
                cta_group=cl.CTAGroup.CTA_1,
            )
            cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)
        cl.barrier_sync_block_aligned()
        tmem_base = tmem_ptr[0]

        tasks_inputs = TasksInputs(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            tma_o_desc=tma_o_desc,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            tmem_base=tmem_base,
            num_kv_tiles=num_kv_tiles,
            q_offset=q_offset,
        )
        if cl.ensure_constant(fmha_config.use_paged_kv):
            tasks_inputs = PagedTasksInputs(
                tma_q_desc=tma_q_desc,
                tma_k_desc=tma_k_desc,
                tma_v_desc=tma_v_desc,
                tma_o_desc=tma_o_desc,
                scale_softmax_log2=scale_softmax_log2,
                output_scale=output_scale,
                tmem_base=tmem_base,
                num_kv_tiles=num_kv_tiles,
                q_offset=q_offset,
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                page_idx_kv=page_idx_kv,
            )
        elif cl.ensure_constant(fmha_config.has_varlen):
            tasks_inputs = RaggedTasksInputs(
                tma_q_desc=tma_q_desc,
                tma_k_desc=tma_k_desc,
                tma_v_desc=tma_v_desc,
                tma_o_desc=tma_o_desc,
                scale_softmax_log2=scale_softmax_log2,
                output_scale=output_scale,
                tmem_base=tmem_base,
                num_kv_tiles=num_kv_tiles,
                q_offset=q_offset,
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
            )
        device_task_manager.run(tasks_inputs, allocators)

        cl.barrier_sync_block_aligned()
        if warp_idx == fmha_config.mma_warp_id:
            cl.tcgen05_deallocate(
                tmem_base,
                fmha_config.tmem_alloc_cols,
                cta_group=cl.CTAGroup.CTA_1,
            )

    if cfg.use_paged_kv:

        @cl.kernel(
            max_threads_per_block=(cfg.block_threads,),
            min_blocks_per_sm=1,
        )
        def fmha_kernel(
            q,
            k,
            v,
            o,
            scale_softmax_log2,
            output_scale,
            cum_seqlen_q,
            cum_seqlen_k,
            page_idx_kv,
        ):
            ragged_q = _make_ragged_tma_array(q, heads, fmha_config)
            ragged_o = _make_ragged_tma_array(o, heads, fmha_config)
            tma_q_desc = cl.tensor_map_tiled(
                ragged_q,
                (
                    fmha_config.tma_copy_q_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                    1,
                ),
                order=(0, 1, 2, 3, 4),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            paged_kv_box = (
                fmha_config.tma_copy_kv_granu_inner,
                fmha_config.num_tokens_per_page,
                1,
                1,
            )
            tma_k_desc = cl.tensor_map_tiled(
                k,
                paged_kv_box,
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.L2_128B,
            )
            tma_v_desc = cl.tensor_map_tiled(
                v,
                paged_kv_box,
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.L2_128B,
            )
            tma_o_desc = cl.tensor_map_tiled(
                ragged_o,
                (
                    fmha_config.tma_copy_o_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                    1,
                ),
                order=(0, 1, 2, 3, 4),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            kernel_body(
                tma_q_desc,
                tma_k_desc,
                tma_v_desc,
                tma_o_desc,
                scale_softmax_log2,
                output_scale,
                cum_seqlen_q,
                cum_seqlen_k,
                page_idx_kv,
            )

    elif cfg.has_varlen:

        @cl.kernel(
            max_threads_per_block=(cfg.block_threads,),
            min_blocks_per_sm=1,
        )
        def fmha_kernel(
            q,
            k,
            v,
            o,
            scale_softmax_log2,
            output_scale,
            cum_seqlen_q,
            cum_seqlen_k,
        ):
            ragged_q = _make_ragged_tma_array(q, heads, fmha_config)
            ragged_o = _make_ragged_tma_array(o, heads, fmha_config)
            tma_q_desc = cl.tensor_map_tiled(
                ragged_q,
                (
                    fmha_config.tma_copy_q_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                    1,
                ),
                order=(0, 1, 2, 3, 4),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            tma_k_desc = cl.tensor_map_tiled(
                k,
                (
                    fmha_config.tma_copy_kv_granu_inner,
                    1,
                    fmha_config.kv_tile_n,
                ),
                order=(0, 1, 2),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            tma_v_desc = cl.tensor_map_tiled(
                v,
                (
                    fmha_config.tma_copy_kv_granu_inner,
                    1,
                    fmha_config.kv_tile_n,
                ),
                order=(0, 1, 2),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            tma_o_desc = cl.tensor_map_tiled(
                ragged_o,
                (
                    fmha_config.tma_copy_o_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                    1,
                ),
                order=(0, 1, 2, 3, 4),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            kernel_body(
                tma_q_desc,
                tma_k_desc,
                tma_v_desc,
                tma_o_desc,
                scale_softmax_log2,
                output_scale,
                cum_seqlen_q,
                cum_seqlen_k,
            )

    else:

        @cl.kernel(
            max_threads_per_block=(cfg.block_threads,),
            min_blocks_per_sm=1,
        )
        def fmha_kernel(
            q,
            k,
            v,
            o,
            scale_softmax_log2,
            output_scale,
        ):
            tma_q_desc = cl.tensor_map_tiled(
                q,
                (
                    fmha_config.tma_copy_q_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                ),
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=(
                    cl.TensorMapL2Promotion.NONE
                    if fmha_config.head_paired
                    else cl.TensorMapL2Promotion.L2_128B
                ),
            )
            tma_k_desc = cl.tensor_map_tiled(
                k,
                (
                    fmha_config.tma_copy_kv_granu_inner,
                    1,
                    fmha_config.kv_tile_n,
                    1,
                ),
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.L2_128B,
            )
            tma_v_desc = cl.tensor_map_tiled(
                v,
                (
                    fmha_config.tma_copy_kv_granu_inner,
                    1,
                    fmha_config.kv_tile_n,
                    1,
                ),
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.L2_128B,
            )
            tma_o_desc = cl.tensor_map_tiled(
                o,
                (
                    fmha_config.tma_copy_o_granu_inner,
                    1,
                    fmha_config.q_tile_m,
                    1,
                ),
                order=(0, 1, 2, 3),
                swizzle=cl.SwizzleMode.SWIZZLE_128B,
                l2_promotion=cl.TensorMapL2Promotion.NONE,
            )
            kernel_body(
                tma_q_desc,
                tma_k_desc,
                tma_v_desc,
                tma_o_desc,
                scale_softmax_log2,
                output_scale,
            )

    return fmha_kernel


def make_tma_view(tensor):
    """Present contiguous BSHD or packed THD storage in TMA coordinates."""
    if tensor.ndim == 3:
        total_sequence, heads, depth = tensor.shape
        return torch.as_strided(
            tensor,
            size=(depth, heads, total_sequence),
            stride=(1, depth, heads * depth),
        )
    if tensor.ndim != 4:
        raise ValueError("TMA view requires contiguous BSHD or packed THD storage")
    batch, sequence, heads, depth = tensor.shape
    return torch.as_strided(
        tensor,
        size=(depth, heads, sequence, batch),
        stride=(1, depth, heads * depth, sequence * heads * depth),
    )


def make_paged_kv_tma_view(tensor):
    """Present contiguous [pages, Hkv, page, D] cache storage to TMA."""
    if tensor.ndim != 4:
        raise ValueError("paged K/V TMA view requires [pages, Hkv, page, D]")
    pages, heads, page_size, depth = tensor.shape
    return torch.as_strided(
        tensor,
        size=(depth, page_size, heads, pages),
        stride=(1, depth, page_size * depth, heads * page_size * depth),
    )


def compute_grid(pipeline):
    if pipeline.cfg.scheduler == "direct":
        return pipeline.problem_shape
    params = pipeline.work_queue.tile_scheduler_config.tile_scheduler_params
    if pipeline.cfg.is_clc_dynamic:
        return ts.ClcDynamicPersistentTileScheduler.get_grid_shape(params)
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return ts.StaticPersistentTileScheduler.get_grid_shape(params, sm_count)


_PIPELINE_CACHE = {}
_KERNEL_CACHE = {}


def get_pipeline(batch, seq_q, seq_k, heads, cfg=FmhaConfig(), *, verbose=False):
    key = (batch, seq_q, seq_k, heads, cfg)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = build_fmha_task_manager(
            batch,
            seq_q,
            seq_k,
            heads,
            cfg,
            verbose=verbose,
        )
    elif verbose:
        _PIPELINE_CACHE[key].task_manager.print_verbose_report()
    return _PIPELINE_CACHE[key]


def get_kernel(pipeline):
    # The frozen task manager embeds the problem geometry. Kernels with equal
    # N but different B/H/Sq must therefore not share a closure.
    key = id(pipeline.device_task_manager)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = make_fmha_kernel(
            pipeline.device_task_manager,
            pipeline.num_kv_tiles,
            pipeline.q_offset,
            pipeline.cfg,
            pipeline.num_heads_q,
        )
    return _KERNEL_CACHE[key]
