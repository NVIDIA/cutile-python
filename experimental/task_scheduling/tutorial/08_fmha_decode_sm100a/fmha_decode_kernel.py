# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""D128 paged decode with fixed SQ1 or packed queries."""

from dataclasses import dataclass, replace
from functools import lru_cache

import cuda.lang as cl
import task_scheduling as ts

from .fmha_decode_config import FmhaDecodeConfig, validate_config
from .fmha_decode_resources.helpers_common import q_group_token_base
from .fmha_decode_resources.smem_resources import (
    SmemQResource,
    SmemKvResource,
    SmemPageOffsetsKvResource,
)
from .fmha_decode_resources.tmem_resources import (
    TmemSResource,
    SmemPResource,
    TmemSoftmaxStatsResource,
    TmemOResource,
)
from .fmha_decode_tasks import (
    create_load_task,
    create_page_offsets_task,
    create_mma_task,
    create_softmax_task,
    create_correction_task,
    create_scheduler_task,
    ScheduleTokenThrottleResource,
    PackedDecodeWorkQueue,
)


@dataclass(frozen=True)
class TasksInputs:
    tma_q_desc: object
    tma_k_desc: object
    tma_v_desc: object
    o: object
    seq_len: object
    page_begin: object
    page_count: object
    paged_kv_indices: object
    head_ratio: object
    tmem_base: object
    seq_lens: object
    paged_kv_indptr: object
    qo_indptr: object
    q_token_offset: object
    seq_len_q: object


def _pipeline(kind, stages, producers, consumers, **kwargs):
    method = getattr(ts.PipelineConfig, "create_" + kind + "_pipeline_cfg")
    return method(
        num_stages=stages,
        producer_group=ts.CooperativeGroup(producers),
        consumer_group=ts.CooperativeGroup(consumers),
        cta_layout_vmnk=(1, 1, 1, 1),
        producer_signaling_threads=(
            ts.SignalingThreads.TaskWarpLeader
            if kind == "tma_umma"
            else ts.SignalingThreads.CtaLeader if kind == "umma_async" else ts.SignalingThreads.All
        ),
        consumer_signaling_threads=(
            ts.SignalingThreads.CtaLeader if kind.endswith("umma") else ts.SignalingThreads.All
        ),
        advance_on_wait=True,
        tcgen05_fence_after_wait=False,
        **kwargs,
    )


def build_fmha_decode_task_manager(
    cfg=FmhaDecodeConfig(), *, problem_shape=(1, 1, 1), verbose=False
):
    validate_config(cfg)
    smem = ts.SmemAllocator(default_add_barriers=False)
    allocations = {}
    for name, size, alignment in (
        ("q", 4096, 1024),
        ("kv", 131072, 1024),
        ("p0", 2048, 1024),
        ("p1", 2048, 1024),
        ("output", 2048, 1024),
        ("pages", 768, 128),
        ("max0", 32, 16),
        ("max1", 32, 16),
        ("sum", 128, 16),
        ("tmem_ptr", 4, 4),
    ):
        allocations[name] = ts.SmemAllocation(name, size, alignment=alignment)
        smem.add(allocations[name])
    work_queue = None
    throttle = None
    if cfg.use_persistent_scheduler:
        response = ts.SmemAllocation("clc_response", 32, alignment=16, count=2)
        smem.add(response)
        queue_cls = PackedDecodeWorkQueue if cfg.use_variable_seqlens_q else ts.WorkQueue
        work_queue = queue_cls(
            **({"cfg": cfg} if cfg.use_variable_seqlens_q else {}),
            name="work_queue",
            pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
                num_stages=2,
                num_bytes=16,
                producer_group=ts.CooperativeGroup(1),
                consumer_group=ts.CooperativeGroup(512),
                cta_layout_vmnk=(1, 1, 1, 1),
                producer_signaling_threads=ts.SignalingThreads.CtaLeader,
                consumer_signaling_threads=ts.SignalingThreads.All,
            ),
            tile_scheduler_config=(
                ts.TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    ts.ClcDynamicPersistentTileSchedulerParams(problem_shape, (1, 1, 1)),
                    response,
                )
            ),
        )
        throttle = ScheduleTokenThrottleResource(
            name="schedule_token_throttle",
            pipeline_config=_pipeline("async_async", 2, 32, 32),
        )
    smem.compute_layout()
    offsets = {name: allocation.offset for name, allocation in allocations.items()}
    sq = SmemQResource(
        name="smem_q", pipeline_config=_pipeline("tma_umma", 2, 1, 1, num_bytes=2048)
    )
    skv = SmemKvResource(
        name="smem_kv", pipeline_config=_pipeline("tma_umma", 4, 1, 1, num_bytes=32768)
    )
    pages = SmemPageOffsetsKvResource(
        name="smem_page_offsets_kv", pipeline_config=_pipeline("async_async", 6, 32, 32)
    )
    s0 = TmemSResource(name="tmem_s0", pipeline_config=_pipeline("umma_async", 1, 1, 128))
    s1 = TmemSResource(name="tmem_s1", pipeline_config=_pipeline("umma_async", 1, 1, 128))
    p0 = SmemPResource(name="smem_p0", pipeline_config=_pipeline("async_umma", 1, 128, 1))
    p1 = SmemPResource(name="smem_p1", pipeline_config=_pipeline("async_umma", 1, 128, 1))
    stats0 = TmemSoftmaxStatsResource(
        name="tmem_stats0", pipeline_config=_pipeline("async_async", 1, 128, 128)
    )
    stats1 = TmemSoftmaxStatsResource(
        name="tmem_stats1", pipeline_config=_pipeline("async_async", 1, 128, 128)
    )
    o = TmemOResource(name="tmem_o", pipeline_config=_pipeline("umma_async", 2, 1, 128))
    resources = (sq, skv, pages, s0, s1, p0, p1, stats0, stats1, o)
    if work_queue is not None:
        resources += (work_queue, throttle)
    barriers = ts.BarrierAllocator()
    for resource in resources:
        barriers.add_resource(resource)
    barriers.compute_layout()
    tmem = ts.TmemAllocator()
    for name, columns in (("s0", 8), ("s1", 8), ("stats0", 32), ("stats1", 32), ("o", 16)):
        tmem.add(ts.TmemAllocation(name, columns))
    tmem.compute_layout()
    tasks = [
        create_softmax_task(s0, p0, stats0, 0, offsets, cfg, work_queue),
        create_softmax_task(s1, p1, stats1, 1, offsets, cfg, work_queue),
        create_correction_task(stats0, stats1, o, offsets, cfg, work_queue),
        create_mma_task(sq, skv, s0, s1, p0, p1, o, offsets, cfg, work_queue),
        create_load_task(sq, skv, pages, offsets, cfg, work_queue, throttle),
        create_page_offsets_task(pages, offsets["pages"], cfg, work_queue),
    ]
    dependencies = {
        sq: [],
        pages: [],
        skv: [pages],
        s0: [sq, skv],
        s1: [sq, skv],
        p0: [s0],
        p1: [s1],
        stats0: [s0],
        stats1: [s1],
        o: [skv, p0, p1],
    }
    if work_queue is not None:
        tasks.append(create_scheduler_task(work_queue, throttle, cfg))
        for deps in dependencies.values():
            deps.append(work_queue)
        dependencies[work_queue] = [work_queue, throttle]
        dependencies[throttle] = [work_queue]
    manager = ts.TaskManager(
        tasks=tasks,
        resource_dependency_graph=dependencies,
        smem_allocator=smem,
        tmem_allocator=tmem,
        barrier_allocator=barriers,
        cta_warps=16,
        verbose=verbose,
        exhaustive_deadlock_race_check=False,
    )
    return manager


@lru_cache(maxsize=None)
def make_fmha_decode_kernel(head_ratio, heads_kv, cfg=FmhaDecodeConfig(), *, batch=1):
    if cfg.heads_q_per_kv not in (0, head_ratio):
        raise ValueError("heads_q_per_kv does not match the input tensors")
    if cfg.groups_tokens_heads_q and head_ratio not in (1, 2, 4, 8):
        raise ValueError("grouped Q8 supports Hq/Hkv = 1, 2, 4, or 8")
    cfg = replace(cfg, heads_q_per_kv=head_ratio)
    device_manager = build_fmha_decode_task_manager(
        cfg, problem_shape=(cfg.num_q_ctas(head_ratio), heads_kv, batch)
    ).to_device()

    def fmha_decode_impl(tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr):
        if cfg.use_variable_seqlens_q:
            seq_lens = cl.Array.from_parts(seq_lens.pointer(), seq_lens.shape, (1,))
            paged_kv_indptr = cl.Array.from_parts(
                paged_kv_indptr.pointer(), paged_kv_indptr.shape, (1,)
            )
            paged_kv_indices = cl.Array.from_parts(
                paged_kv_indices.pointer(), paged_kv_indices.shape, (1,)
            )
            qo_indptr = cl.Array.from_parts(qo_indptr.pointer(), qo_indptr.shape, (1,))
            row_stride = head_ratio * heads_kv * 128
            o = cl.Array.from_parts(
                o.pointer(), (o.shape[0], head_ratio * heads_kv, 128), (row_stride, 128, 1)
            )
        q_token_offset, seq_len_q = cl.int32(0), cl.int32(1)
        q_tile_is_active = True
        if cfg.use_variable_seqlens_q and not cfg.use_persistent_scheduler:
            request = cl.block_index(2)
            q_token_offset = qo_indptr[request]
            seq_len_q = qo_indptr[request + 1] - q_token_offset
            q_tile_is_active = q_group_token_base(cl.block_index(0), cfg) < seq_len_q
        allocators = device_manager.setup_resources_and_tasks()
        if q_tile_is_active:
            ptr = allocators.smem_allocator.get(
                "tmem_ptr", cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR)
            )
            warp, _ = cl.shuffle_sync(cl.ShuffleKind.INDEX, cl.thread_index(0) // 32, 0)
            if warp == 12:
                cl.tcgen05_allocate(ptr.pointer(), 128, cta_group=cl.CTAGroup.CTA_1)
                cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)
            cl.barrier_sync_block_aligned()
            base = ptr[0]
            if cfg.use_persistent_scheduler:
                # Each worker binds these fields at the head of its work-tile loop.
                seq_len = cl.int32(0)
                page_begin = cl.int32(0)
                page_count = cl.int32(0)
            else:
                request = cl.block_index(2)
                seq_len = seq_lens[request]
                page_begin = paged_kv_indptr[request]
                page_count = paged_kv_indptr[request + 1] - page_begin
            inputs = TasksInputs(
                tq, tk, tv, o, seq_len, page_begin, page_count, paged_kv_indices, head_ratio, base,
                seq_lens, paged_kv_indptr,
                qo_indptr, q_token_offset, seq_len_q,
            )
            device_manager.run(inputs, allocators)
            cl.barrier_sync_block_aligned()
            if warp == 12:
                cl.tcgen05_deallocate(base, 128, cta_group=cl.CTAGroup.CTA_1)

    if cfg.use_variable_seqlens_q:
        @cl.kernel(max_threads_per_block=(512,), min_blocks_per_sm=1)
        def fmha_decode(tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr):
            fmha_decode_impl(tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr)
    else:
        @cl.kernel(max_threads_per_block=(512,), min_blocks_per_sm=1)
        def fmha_decode(tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices):
            fmha_decode_impl(tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices, None)

    return fmha_decode


@lru_cache(maxsize=None)
def make_fmha_decode_launcher(head_ratio, heads_kv, cfg=FmhaDecodeConfig(), *, batch=1):
    """Encode the Q/K/V tensor maps on the host before launching decode."""
    kernel = make_fmha_decode_kernel(head_ratio, heads_kv, cfg, batch=batch)
    cfg = replace(cfg, heads_q_per_kv=head_ratio)
    grid = (cfg.num_q_ctas(head_ratio), heads_kv, batch)
    q_box = (64, 8, 1, 1, 1)
    if cfg.groups_tokens_heads_q:
        q_box = (64, head_ratio, 1, 8 // head_ratio, 1)
    if cfg.use_variable_seqlens_q:
        q_box = (64, head_ratio if cfg.groups_tokens_heads_q else 8, cfg.q_tokens_per_cta, 1, 1)

    def launch_impl(stream, q, k, v, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr):
        row_stride = head_ratio * heads_kv * 128
        if cfg.use_variable_seqlens_q:
            q_view = cl.Array.from_parts(
                q.pointer(),
                (128, head_ratio * heads_kv, cfg.q_tokens_per_cta, 1 << 31, 1 << 31),
                (1, 128, row_stride, (1 << 35) - row_stride, row_stride),
            )
        else:
            q_view = cl.Array.from_parts(
                q.pointer(),
                (128, head_ratio, heads_kv, 1, q.shape[0]),
                (1, 128, head_ratio * 128, row_stride, row_stride),
            )
        kv_stride = cfg.num_tokens_per_page * 128
        k_view = cl.Array.from_parts(
            k.pointer(),
            (128, cfg.num_tokens_per_page, heads_kv, k.shape[0]),
            (1, 128, kv_stride, heads_kv * kv_stride),
        )
        v_view = cl.Array.from_parts(
            v.pointer(),
            (128, cfg.num_tokens_per_page, heads_kv, v.shape[0]),
            (1, 128, kv_stride, heads_kv * kv_stride),
        )
        tq = cl.tensor_map_tiled(
            q_view, q_box, order=(0, 1, 2, 3, 4),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
            l2_promotion=cl.TensorMapL2Promotion.NONE,
        )
        tk = cl.tensor_map_tiled(
            k_view, (64, cfg.num_tokens_per_page, 1, 1), order=(0, 1, 2, 3),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
            l2_promotion=cl.TensorMapL2Promotion.NONE,
        )
        tv = cl.tensor_map_tiled(
            v_view, (64, cfg.num_tokens_per_page, 1, 1), order=(0, 1, 2, 3),
            swizzle=cl.SwizzleMode.SWIZZLE_128B,
            l2_promotion=cl.TensorMapL2Promotion.NONE,
        )
        args = (tq, tk, tv, o, seq_lens, paged_kv_indptr, paged_kv_indices)
        if cfg.use_variable_seqlens_q:
            args += (qo_indptr,)
        cl.launch(stream, grid, (512,), kernel, args, programmatic_dependent_launch=False)

    if cfg.use_variable_seqlens_q:
        @cl.host_entry
        def launcher(stream, q, k, v, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr):
            launch_impl(stream, q, k, v, o, seq_lens, paged_kv_indptr, paged_kv_indices, qo_indptr)
    else:
        @cl.host_entry
        def launcher(stream, q, k, v, o, seq_lens, paged_kv_indptr, paged_kv_indices):
            launch_impl(stream, q, k, v, o, seq_lens, paged_kv_indptr, paged_kv_indices, None)

    return launcher
