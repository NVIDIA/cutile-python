# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""BatchedGemm task-manager assembly and CUDA Lang kernel launch."""

from dataclasses import dataclass, replace

import cuda.lang as cl
import task_scheduling as ts

from .batched_gemm_config import (
    ActKind,
    DType,
    compute_warp_layout,
    freeze_config,
    make_config,
    validate_config,
)
from .batched_gemm_resources import (
    GmemAResource,
    GmemBResource,
    GmemCResource,
    SmemAResource,
    SmemBResource,
    TmemCResource,
)
from .batched_gemm_tasks import (
    create_epilogue_task,
    create_load_a_task,
    create_load_b_task,
    create_mma_task,
    create_padding_task,
    create_workid_task,
)


def _make_pipeline_configs(cfg):
    one_thread = ts.CooperativeGroup(1)
    cluster_vmnk = (cfg.cluster_m, 1, 1, 1)
    result = {}
    for operand, num_bytes, num_stages, mcast_mode in (
        ("smem_a", cfg.num_bytes_a_tma_per_stage, cfg.num_stages_a, (1, 0)),
        ("smem_b", cfg.num_bytes_b_tma_per_stage, cfg.num_stages_b, (0, 1)),
    ):
        result[operand] = ts.PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=num_stages,
            num_bytes=num_bytes * cfg.cluster_m,
            producer_group=one_thread,
            consumer_group=one_thread,
            cta_layout_vmnk=cluster_vmnk,
            consumer_signaling_threads=ts.SignalingThreads.CtaLeader,
            mcast_mode_mn=mcast_mode,
            # Match the source TMA/UMMA pipeline: completion is ordered by
            # its mbarrier, without an additional tcgen05 wait fence.
            tcgen05_fence_after_wait=False,
        )
    result["tmem_c"] = ts.PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.num_stages_tmem_acc,
        producer_group=one_thread,
        consumer_group=ts.CooperativeGroup(cfg.num_epilogue_warps * 32 * cfg.cluster_m),
        cta_layout_vmnk=cluster_vmnk,
        producer_signaling_threads=ts.SignalingThreads.CtaLeader,
        tcgen05_fence_after_wait=False,
    )
    return result


@dataclass(frozen=True)
class BatchedGemmPipeline:
    cfg: object
    task_manager: object
    device_task_manager: object
    work_queue: object
    grid: tuple


def _num_k_tiles(tasks_inputs):
    return tasks_inputs.num_k_tiles


def build_batched_gemm_task_manager(cfg, problem_mnk, *, verbose=False):
    cfg = replace(cfg)
    validate_config(cfg, problem_mnk)
    compute_warp_layout(cfg)
    cfg = freeze_config(cfg)
    # Retain the source problem names; device work resolves K through TasksInputs.
    problem_m, problem_n, problem_k = problem_mnk  # noqa: RUF059
    # The source derives K-loop bounds from the live launch arguments. Keep
    # that runtime bound instead of specializing the schedule to this shape.
    num_k_tiles = ts.dynamic_domain_bound(_num_k_tiles)
    pcfgs = _make_pipeline_configs(cfg)
    _alloc_a = ts.SmemAllocation(
        "SmemA_a", cfg.num_bytes_a_per_stage * cfg.num_stages_a, alignment=1024
    )
    _alloc_b = ts.SmemAllocation(
        "SmemB_b", cfg.num_bytes_b_smem_per_stage * cfg.num_stages_b, alignment=1024
    )
    _alloc_c = ts.TmemAllocation(
        "TmemC_tmem_c", cfg.tmem_c_cols_per_stage * cfg.num_stages_tmem_acc
    )
    _alloc_sc = ts.SmemAllocation(
        "GmemC_sc", cfg.num_bytes_c_smem_scratch, alignment=1024
    )
    tmem_ptr_alloc = ts.SmemAllocation("tmem_ptr", 4, alignment=4)
    gmem_a = GmemAResource(name="GmemA")
    gmem_b = GmemBResource(name="GmemB")
    gmem_c = GmemCResource(name="GmemC", smem_requirements=[_alloc_sc])
    smem_a = SmemAResource(
        name="SmemA", pipeline_config=pcfgs["smem_a"], smem_requirements=[_alloc_a]
    )
    smem_b = SmemBResource(
        name="SmemB", pipeline_config=pcfgs["smem_b"], smem_requirements=[_alloc_b]
    )
    tmem_c = TmemCResource(
        name="TmemC", pipeline_config=pcfgs["tmem_c"], tmem_requirements=[_alloc_c]
    )
    grid = (problem_m // cfg.tile_m, problem_n // cfg.tile_n, 1)
    work_queue = None
    if cfg.is_persistent:
        tile_sched_params = ts.ClcDynamicPersistentTileSchedulerParams(
            grid, (cfg.cluster_m, 1, 1)
        )
        clc_response_alloc = ts.SmemAllocation(
            "clc_response",
            cfg.num_stages_workid * 16,
            alignment=16,
            count=cfg.num_stages_workid,
        )
        work_queue = ts.WorkQueue(
            name="WorkQueue",
            tile_scheduler_config=(
                ts.TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    tile_sched_params, clc_response_alloc
                )
            ),
            pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
                num_stages=cfg.num_stages_workid,
                num_bytes=16,
                producer_group=ts.CooperativeGroup(1),
                consumer_group=ts.CooperativeGroup(cfg.threads_per_cta),
            ),
            smem_requirements=[clc_response_alloc],
        )
        grid = ts.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

    smem_alloc = ts.SmemAllocator(default_add_barriers=False)
    for resource in (smem_a, smem_b):
        smem_alloc.add_resource(resource)
    if work_queue is not None:
        smem_alloc.add_resource(work_queue)
    # Reserve C scratch even for direct global stores, retaining its allocation
    # order and the configured shared-memory footprint.
    smem_alloc.add_resource(gmem_c)
    smem_alloc.add(tmem_ptr_alloc)
    smem_alloc.compute_layout()
    tmem_alloc = ts.TmemAllocator()
    tmem_alloc.add_resource(tmem_c)
    tmem_alloc.compute_layout()
    barrier_alloc = ts.BarrierAllocator()
    for resource in (smem_a, smem_b, tmem_c):
        barrier_alloc.add_resource(resource)
    if work_queue is not None:
        barrier_alloc.add_resource(work_queue)
    barrier_alloc.compute_layout()
    task_list = [
        create_load_a_task(
            cfg, gmem_a, smem_a, work_queue, num_k_tiles, _alloc_a.offset
        ),
        create_load_b_task(
            cfg, gmem_b, smem_b, work_queue, num_k_tiles, _alloc_b.offset
        ),
        create_mma_task(
            cfg,
            smem_a,
            smem_b,
            tmem_c,
            work_queue,
            num_k_tiles,
            _alloc_a.offset,
            _alloc_b.offset,
        ),
        create_epilogue_task(cfg, tmem_c, gmem_c, work_queue, num_k_tiles),
    ]
    if cfg.is_persistent:
        task_list.append(create_workid_task(cfg, work_queue))
    if cfg.num_padding_warps:
        task_list.append(create_padding_task(cfg, work_queue, num_k_tiles))
    wq_deps = [work_queue] if work_queue is not None else []
    dep_graph = {
        smem_a: [gmem_a] + wq_deps,
        smem_b: [gmem_b] + wq_deps,
        tmem_c: [smem_a, smem_b] + wq_deps,
        gmem_c: [tmem_c] + wq_deps,
    }
    task_manager = ts.TaskManager(
        tasks=task_list,
        resource_dependency_graph=dep_graph,
        smem_allocator=smem_alloc,
        tmem_allocator=tmem_alloc,
        barrier_allocator=barrier_alloc,
        verbose=verbose,
        exhaustive_representative_domain=True,
    )
    return BatchedGemmPipeline(
        cfg, task_manager, task_manager.to_device(), work_queue, grid
    )


@dataclass(frozen=True)
class TasksInputs:
    tma_a_desc: object
    tma_b_desc: object
    tmem_ptr_i32: object
    gC: object
    tile_idx_view: object
    mn_limit_view: object
    problem_m: object
    problem_n: object
    num_k_tiles: object


def make_batched_gemm_kernel(pipeline):
    cfg = pipeline.cfg
    device_task_manager = pipeline.device_task_manager
    num_tmem_cols = max(32, 1 << (cfg.tmem_required_cols - 1).bit_length())
    # These unrolled specializations need the final O3 cleanup after domain-loop
    # expansion. Other variants retain the established O2 tradeoff.
    fc2_static_n64 = (
        not cfg.is_persistent
        and not cfg.is_swap_ab
        and cfg.act_kind == int(ActKind.NONE)
        and cfg.tile_n == 64
        and cfg.tile_k == 64
    )
    kernel_opt_level = (
        3
        if fc2_static_n64
        or (
            cfg.is_swap_ab
            and cfg.tile_n == 128
            and cfg.tile_k == 128
            and cfg.epi_tile_n == 32
            and cfg.use_unroll_loop_2x_for_mma
        )
        else 2
    )

    @cl.kernel(max_threads_per_block=(cfg.threads_per_cta,), opt_level=kernel_opt_level)
    def batched_gemm_kernel_bf16(
        tma_a_desc,
        tma_b_desc,
        gC,
        tile_idx_view,
        mn_limit_view,
        problem_m: cl.int32,
        problem_n: cl.int32,
        problem_k: cl.int32,
    ):
        warp_idx = cl.warp_index()
        cl.prefetch_tensor_map(tma_a_desc)
        cl.prefetch_tensor_map(tma_b_desc)
        device_allocators = device_task_manager.setup_resources_and_tasks()
        tmem_ptr_i32 = device_allocators.smem_allocator.get(
            "tmem_ptr", cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR)
        )
        if warp_idx == cfg.epilogue_warp_idx:
            cl.tcgen05_allocate(
                tmem_ptr_i32.pointer(), num_tmem_cols, cta_group=cl.CTAGroup.CTA_1
            )
            cl.tcgen05_relinquish_allocation_permit(cta_group=cl.CTAGroup.CTA_1)
        cl.barrier_sync_block_aligned()
        device_task_manager.run(
            TasksInputs(
                tma_a_desc,
                tma_b_desc,
                tmem_ptr_i32,
                gC,
                tile_idx_view,
                mn_limit_view,
                problem_m,
                problem_n,
                problem_k // cfg.tile_k,
            ),
            device_allocators,
        )
        cl.tcgen05_fence_before_thread_sync()
        if warp_idx < cfg.num_epilogue_warps:
            cl.barrier_sync_block_aligned(
                number_of_threads=cfg.num_epilogue_warps * 32, barrier_id=10
            )
            if warp_idx == cfg.epilogue_warp_idx:
                cl.tcgen05_deallocate(
                    tmem_ptr_i32[0], num_tmem_cols, cta_group=cl.CTAGroup.CTA_1
                )

    return batched_gemm_kernel_bf16


def make_tensor_maps(a, b, cfg):
    """Encode the source K-box layouts on the host for each live launch."""
    problem_k = a.shape[2]
    a_kbox = a.reshape(a.shape[0], a.shape[1], problem_k // 64, 64).transpose(1, 2)
    tma_a_desc = cl.tensor_map_tiled(
        a_kbox,
        (64, cfg.tile_m, cfg.tile_k // 64, 1),
        order="F",
        swizzle=cl.SwizzleMode.SWIZZLE_128B,
    )
    tma_b_desc = cl.tensor_map_tiled(
        b, (64, cfg.tile_n, 1), order="F", swizzle=cl.SwizzleMode.SWIZZLE_128B
    )
    return tma_a_desc, tma_b_desc


def gemm(a, b, c, tile_idx, mn_limit, *, cfg=None, stream=None):
    """Launch expert GEMM, including either SwiGLU operand orientation."""
    import torch

    if cfg is None:
        cfg = make_config()
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Expected rank-3 A and B operands")
    if cfg.is_swap_ab:
        num_experts, problem_m, problem_k = a.shape
        if b.shape[0] != 1:
            raise ValueError("swap-AB expects A[E,M,K] and B[1,N,K]")
        _, problem_n, b_k = b.shape
        output_shape = (
            problem_m // 2 if cfg.act_kind == int(ActKind.SWIGLU) else problem_m,
            problem_n,
        )
    else:
        if a.shape[0] != 1:
            raise ValueError("non-swap expects A[1,M,K] and B[E,N,K]")
        _, problem_m, problem_k = a.shape
        num_experts, problem_n, b_k = b.shape
        output_shape = (
            problem_m,
            problem_n // 2 if cfg.act_kind == int(ActKind.SWIGLU) else problem_n,
        )
    if b_k != problem_k or c.shape != output_shape:
        raise ValueError("A, B, C problem shapes disagree")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("A and B must be BF16")
    expected_dtype = torch.bfloat16 if cfg.dtype_c == int(DType.BF16) else torch.float16
    if c.dtype != expected_dtype:
        raise ValueError("C dtype disagrees with cfg.dtype_c")
    if (
        any(
            t.device != a.device or not t.is_contiguous()
            for t in (a, b, tile_idx, mn_limit)
        )
        or not a.is_cuda
    ):
        raise ValueError("All tensors must be contiguous on the same CUDA device")
    valid_c_layout = (
        c.transpose(0, 1).is_contiguous() if cfg.is_swap_ab else c.is_contiguous()
    )
    if c.device != a.device or not valid_c_layout:
        raise ValueError("C must use the source row-major or swap-AB M-major layout")
    if tile_idx.dtype != torch.int32 or mn_limit.dtype != torch.int32:
        raise ValueError("tile_idx and mn_limit must be int32")
    num_token_tiles = (
        problem_n // cfg.tile_n if cfg.is_swap_ab else problem_m // cfg.tile_m
    )
    if tile_idx.shape != (num_token_tiles,) or mn_limit.shape != tile_idx.shape:
        raise ValueError("Expected one expert index and token limit per token tile")
    if num_experts <= 0:
        raise ValueError("At least one expert is required")
    pipeline = build_batched_gemm_task_manager(cfg, (problem_m, problem_n, problem_k))
    kernel = make_batched_gemm_kernel(pipeline)
    tma_a_desc, tma_b_desc = make_tensor_maps(a, b, pipeline.cfg)
    args = (tma_a_desc, tma_b_desc, c, tile_idx, mn_limit, problem_m, problem_n, problem_k)
    cl.launch(
        torch.cuda.current_stream(a.device) if stream is None else stream,
        pipeline.grid,
        (pipeline.cfg.threads_per_cta, 1, 1),
        kernel,
        args,
    )
    return kernel, pipeline, args
