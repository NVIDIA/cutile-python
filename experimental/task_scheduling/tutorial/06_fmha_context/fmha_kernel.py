# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""SM120 FMHA context kernel and immutable task-manager construction."""

from dataclasses import dataclass, replace

import cuda.lang as cl

import task_scheduling as ts

try:
    from .fmha_resources import (
        BUFFER_ALIGNMENT,
        SM120_SMEM_CAPACITY_BYTES,
        FmhaConfig,
        GmemOResource,
        GmemQKVResource,
        SmemKResource,
        SmemVResource,
    )
    from .fmha_tasks import (
        create_compute_task,
        create_load_task,
        create_padding_task,
    )
except ImportError:
    from fmha_resources import (
        BUFFER_ALIGNMENT,
        SM120_SMEM_CAPACITY_BYTES,
        FmhaConfig,
        GmemOResource,
        GmemQKVResource,
        SmemKResource,
        SmemVResource,
    )
    from fmha_tasks import (
        create_compute_task,
        create_load_task,
        create_padding_task,
    )


@dataclass(frozen=True)
class TasksInputs:
    q: object
    tma_k_desc: object
    tma_v_desc: object
    o: object
    seqlen_q: object
    seqlen_k: object
    num_heads_q: object
    num_kv_tiles: object
    softmax_scale_log2: object


@dataclass(frozen=True)
class _ResolvedCausalDomainTask:
    _num_kv_tiles: object
    _seqlen_q: object
    _num_heads_q: object
    _q_tile: int
    _kv_tile: int
    _use_head_fast_grid: bool


@dataclass(frozen=True)
class _DeviceCausalDomainTask:
    _q_tile: int
    _kv_tile: int
    _use_head_fast_grid: bool

    def bind_inputs(self, tasks_inputs):
        return _ResolvedCausalDomainTask(
            tasks_inputs.num_kv_tiles,
            tasks_inputs.seqlen_q,
            tasks_inputs.num_heads_q,
            self._q_tile,
            self._kv_tile,
            self._use_head_fast_grid,
        )


class CausalDomainTask(ts.Task):
    """Task with a per-work-tile dynamic K/V loop count."""

    def __init__(
        self,
        *args,
        q_tile,
        kv_tile,
        use_head_fast_grid,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._q_tile = q_tile
        self._kv_tile = kv_tile
        self._use_head_fast_grid = use_head_fast_grid

    def _freeze_domain_task(self):
        return _DeviceCausalDomainTask(
            self._q_tile,
            self._kv_tile,
            self._use_head_fast_grid,
        )

    def get_domain(self, tile_coord):
        """Return the K/V loop count for the current work tile."""
        tile_q_idx = tile_coord[0]
        if self._use_head_fast_grid:
            tile_q_idx = tile_q_idx // self._num_heads_q
        num_q_tiles = cl.cdiv(self._seqlen_q, self._q_tile)
        seq_q_idx = num_q_tiles - tile_q_idx - 1
        if self._q_tile == self._kv_tile:
            return cl.minimum(self._num_kv_tiles, seq_q_idx + 1)
        max_q_row = seq_q_idx * self._q_tile + self._q_tile - 1
        causal_n = max_q_row // self._kv_tile + 1
        return cl.minimum(self._num_kv_tiles, causal_n)


def _full_kv_domain(tasks_inputs):
    return tasks_inputs.num_kv_tiles


def build_fmha_task_manager(
    cfg: FmhaConfig,
    *,
    verbose=False,
    exhaustive_deadlock_race_check=True,
):
    """Construct and validate an FMHA context ``TaskManager``."""
    task_class = CausalDomainTask if cfg.is_causal else ts.Task
    domain_end = (
        CausalDomainTask.get_domain
        if cfg.is_causal
        else ts.dynamic_domain_bound(_full_kv_domain)
    )
    domain_kwargs = (
        {
            "task_class": task_class,
            "domain": domain_end,
            "q_tile": cfg.q_tile,
            "kv_tile": cfg.kv_tile,
            "use_head_fast_grid": cfg.use_causal_head_fast_grid,
        }
        if cfg.is_causal
        else {"domain": domain_end}
    )
    k_allocation = ts.SmemAllocation(
        "smem_k", cfg.tma_copy_kv_bytes, alignment=BUFFER_ALIGNMENT
    )
    v_allocation = ts.SmemAllocation(
        "smem_v", cfg.tma_copy_kv_bytes, alignment=BUFFER_ALIGNMENT
    )
    tma_producer_group = ts.CooperativeGroup(1)
    compute_consumer_group = ts.CooperativeGroup(cfg.num_compute_warps)
    smem_k_cfg = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        num_stages=cfg.k_stage,
        num_bytes=cfg.tma_copy_kv_bytes,
        producer_group=tma_producer_group,
        consumer_group=compute_consumer_group,
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
        consumer_signaling_threads=ts.SignalingThreads.All,
    )
    smem_v_cfg = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        num_stages=cfg.v_stage,
        num_bytes=cfg.tma_copy_kv_bytes,
        producer_group=tma_producer_group,
        consumer_group=compute_consumer_group,
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.TaskWarpLeader,
        consumer_signaling_threads=ts.SignalingThreads.All,
    )

    gmem_qkv = GmemQKVResource(name="gmem_qkv")
    smem_k = SmemKResource(
        name="smem_k",
        pipeline_config=smem_k_cfg,
        smem_requirements=[k_allocation],
    )
    smem_v = SmemVResource(
        name="smem_v",
        pipeline_config=smem_v_cfg,
        smem_requirements=[v_allocation],
    )
    gmem_o = GmemOResource(name="gmem_o")

    allocator = ts.SmemAllocator()
    allocator.add_resource(smem_k)
    allocator.add_resource(smem_v)
    allocator.compute_layout()
    if k_allocation.offset != 0 or v_allocation.offset != cfg.tma_copy_kv_bytes:
        raise ValueError("unexpected K/V shared-memory layout")
    cfg = replace(
        cfg,
        k_smem_offset=k_allocation.offset,
        v_smem_offset=v_allocation.offset,
    )

    load_task = create_load_task(cfg, gmem_qkv, smem_k, smem_v, **domain_kwargs)
    compute_task = create_compute_task(cfg, smem_k, smem_v, gmem_o, **domain_kwargs)
    padding_task = create_padding_task(cfg, **domain_kwargs)
    tasks = [load_task, compute_task, padding_task]

    resource_dependency_graph = {
        smem_k: [gmem_qkv],
        smem_v: [gmem_qkv],
        gmem_o: [smem_k, smem_v],
    }

    barrier_allocator = ts.BarrierAllocator()
    barrier_allocator.add_producer_consumer(
        "smem_k", cfg.k_stage, tma_producer_group, compute_consumer_group
    )
    barrier_allocator.add_producer_consumer(
        "smem_v", cfg.v_stage, tma_producer_group, compute_consumer_group
    )
    barrier_allocator.compute_layout()

    return ts.TaskManager(
        tasks,
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=allocator,
        barrier_allocator=barrier_allocator,
        verbose=verbose,
        smem_capacity_bytes=SM120_SMEM_CAPACITY_BYTES,
        exhaustive_representative_domain=exhaustive_deadlock_race_check,
    )


def _swizzle_mode(cfg):
    return {
        128: cl.SwizzleMode.SWIZZLE_128B,
        64: cl.SwizzleMode.SWIZZLE_64B,
        32: cl.SwizzleMode.SWIZZLE_32B,
    }[cfg.swizzle_chunk_bytes]


def make_fmha_kernel(task_manager, cfg: FmhaConfig):
    """Specialize the CUDA Lang kernel around one frozen device schedule."""
    task_manager = task_manager.to_device()

    @cl.kernel(
        max_threads_per_block=(cfg.block_threads, 1, 1),
        min_blocks_per_sm=1,
    )
    def fmha_kernel(
        q,
        tma_k_desc,
        tma_v_desc,
        o,
        seqlen_q,
        seqlen_k,
        num_heads_q: cl.Constant[int],
        softmax_scale_log2,
        head_dim: cl.Constant[int],
        q_tile: cl.Constant[int],
        kv_tile: cl.Constant[int],
        num_compute_warps: cl.Constant[int],
        tma_swizzle_chunk_elems: cl.Constant[int],
        is_bf16: cl.Constant[bool],
        is_causal: cl.Constant[bool],
        use_causal_head_fast_grid: cl.Constant[bool],
    ):
        if cl.lane_index() == 0:
            cl.prefetch_tensor_map(tma_k_desc)
            cl.prefetch_tensor_map(tma_v_desc)

        num_kv_tiles = cl.cdiv(seqlen_k, kv_tile)

        allocators = task_manager.setup_resources_and_tasks()
        task_manager.run(
            TasksInputs(
                q.pointer(),
                tma_k_desc,
                tma_v_desc,
                o.pointer(),
                seqlen_q,
                seqlen_k,
                num_heads_q,
                num_kv_tiles,
                softmax_scale_log2,
            ),
            allocators,
        )

    return fmha_kernel


def make_kv_tensor_maps(k_view, v_view, cfg: FmhaConfig):
    tile_shape = (
        cfg.tma_swizzle_chunk_elems,
        cfg.kv_tile,
        cfg.tma_swizzle_chunks,
        1,
        1,
    )
    swizzle = _swizzle_mode(cfg)
    return (
        cl.tensor_map_tiled(k_view, tile_shape, swizzle=swizzle),
        cl.tensor_map_tiled(v_view, tile_shape, swizzle=swizzle),
    )


_TASK_MANAGER_CACHE = {}
_KERNEL_CACHE = {}


def _get_fmha_task_manager(cfg: FmhaConfig, *, verbose=False):
    if cfg not in _TASK_MANAGER_CACHE:
        _TASK_MANAGER_CACHE[cfg] = build_fmha_task_manager(cfg, verbose=verbose)
    elif verbose:
        _TASK_MANAGER_CACHE[cfg].print_verbose_report()
    return _TASK_MANAGER_CACHE[cfg]


def get_fmha_kernel(cfg: FmhaConfig, *, verbose=False):
    task_manager = _get_fmha_task_manager(cfg, verbose=verbose)
    if cfg not in _KERNEL_CACHE:
        _KERNEL_CACHE[cfg] = make_fmha_kernel(task_manager, cfg)
    return _KERNEL_CACHE[cfg]
