# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from task_scheduling_test_requirements import task_scheduling as ts

from task_scheduling.task import (
    DeviceDomainLoop,
    DeviceGuard,
    DevicePipelineStep,
)


def pipeline_config(stages=2):
    return ts.PipelineConfig.create_tma_async_pipeline_cfg(
        stages, 256, ts.CooperativeGroup(1), ts.CooperativeGroup(128)
    )


def test_pipeline_config_interleave_validation_and_offsets():
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        4,
        128,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(32),
        interleave_stride=2,
    )
    assert config.interleave_strides == (2, 2, 2, 2)
    assert config.resolved_storage_offset_empty == 4
    with pytest.raises(ValueError, match="evenly divide"):
        ts.PipelineConfig.create_tma_async_pipeline_cfg(
            3, 128, ts.CooperativeGroup(1), ts.CooperativeGroup(32), interleave_stride=2
        )


def test_smem_alignment_alias_phases_and_capacity():
    a = ts.SmemAllocation("a", 33, 32)
    b = ts.SmemAllocation("b", 16, 16)
    c = ts.SmemAllocation("c", 64, 64)
    allocator = ts.SmemAllocator(capacity=128)
    allocator.add_alias_group([[a, b], [c]])
    allocator.compute_layout()
    assert a.offset == c.offset == 0
    assert b.offset == 48
    assert allocator.total_smem_bytes == 64
    assert allocator.data_alignment == 64
    assert allocator.allocation_offsets == (("a", 0), ("c", 0), ("b", 48))

    too_small = ts.SmemAllocator(capacity=32)
    too_small.add(ts.SmemAllocation("large", 64, 1))
    with pytest.raises(ValueError, match="budget exceeded"):
        too_small.compute_layout()


def test_smem_usage_report_separates_data_and_barriers(capsys):
    data = ts.SmemAllocation("smem_data", 256, 128)
    resource = ts.MemoryResource(
        name="smem",
        pipeline_config=pipeline_config(stages=1),
        smem_requirements=[data],
    )
    allocator = ts.SmemAllocator()
    allocator.add_resource(resource)
    allocator.compute_layout()

    allocator.print_usage_report()

    report = capsys.readouterr().out
    assert "[smem-layout] SMEM Usage Report" in report
    assert "smem_data" in report
    assert "smem.barriers" not in report
    assert "Data SMEM:             256 B" in report
    assert "Barrier SMEM:           16 B" in report
    assert "Total SMEM:            272 B  (data + barriers)" in report
    assert "Total:               256 B" in report
    assert allocator.data_alignment == 128


def test_tmem_layout_and_duplicate_alias_rejected(capsys):
    a = ts.TmemAllocation("a", 32)
    b = ts.TmemAllocation("b", 64)
    allocator = ts.TmemAllocator(capacity=64)
    with pytest.raises(ValueError, match="only once"):
        allocator.add_alias_group([[a], [a]])
    allocator.add_alias_group([[a], [b]])
    allocator.compute_layout()
    assert allocator.total_tmem_columns == 64

    allocator.print_usage_report()

    report = capsys.readouterr().out
    assert "[tmem-layout] TMEM Usage Report" in report
    assert "Alias group 1:" in report
    assert "Phase 1: a (32 cols)" in report
    assert "Total:                64 cols" in report
    assert "Alias savings:        32 cols" in report


def test_barrier_layout_is_padded_and_names_are_unique(capsys):
    allocator = ts.BarrierAllocator()
    allocator.add_producer_consumer(
        "pipe", 3, ts.CooperativeGroup(1), ts.CooperativeGroup(128)
    )
    with pytest.raises(ValueError, match="duplicate"):
        allocator.add(ts.BarrierAllocation("pipe.full", 1, 1))
    allocator.compute_layout()
    assert allocator.offset_of("pipe.full") == 0
    assert allocator.padded_size_bytes == 32 * 8
    assert allocator.allocation_offsets == (
        ("pipe.full", 0),
        ("pipe.empty", 3),
    )

    allocator.print_usage_report()

    report = capsys.readouterr().out
    assert "[barrier-layout] Barrier Usage Report" in report
    assert "pipe.full" in report
    assert "pipe.empty" in report
    assert "Region V0: offsets 0..32 (32 padded slots)" in report
    assert "[  0,  3): arrive=1" in report
    assert "[  3,  6): arrive=128" in report
    assert "Used barriers:   6" in report
    assert "Padded total:   32" in report


def test_barrier_layout_coalesces_equal_arrival_counts():
    allocator = ts.BarrierAllocator()
    allocator.add(ts.BarrierAllocation("warp_a", 1, 32))
    allocator.add(ts.BarrierAllocation("single", 1, 1))
    allocator.add(ts.BarrierAllocation("warp_b", 2, 32))

    allocator.compute_layout()

    assert allocator.offset_of("single") == 0
    assert allocator.offset_of("warp_a") == 1
    assert allocator.offset_of("warp_b") == 2
    assert allocator.initialization_runs == ((0, 1, 1), (1, 4, 32))


def test_pipeline_group_merge_and_device_pipeline_boundary():
    left = ts.MemoryResource(name="left", pipeline_config=pipeline_config())
    right = ts.MemoryResource(name="right", pipeline_config=pipeline_config())
    group = ts.PipelineGroup(
        name="group", members=[left, right], mode=ts.PipelineGroupMode.Merge
    )
    assert group.left.member is left
    assert group.num_barriers_per_stage == 3

    tma_umma = ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        1, 128, ts.CooperativeGroup(1), ts.CooperativeGroup(128)
    )
    resource = ts.MemoryResource(name="umma", pipeline_config=tma_umma)
    assert resource.create_pipeline() is tma_umma
    binding = ts.DevicePipelineBinding.from_config(tma_umma)
    assert binding.kind == ts.DevicePipelineBinding.TMA_UMMA
    assert binding.num_bytes == 128
    assert binding.consumer_arrivals == 128

    unsupported = ts.PipelineConfig(
        1,
        0,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(1),
        ts.PipelineType.UmmaUmma,
    )
    unsupported_resource = ts.MemoryResource(
        name="umma_umma", pipeline_config=unsupported
    )
    with pytest.raises(NotImplementedError, match="UmmaUmma"):
        unsupported_resource.create_pipeline()
    with pytest.raises(NotImplementedError, match="UmmaUmma"):
        ts.DevicePipelineBinding.from_config(unsupported)


def test_pipeline_stages_lower_without_callbacks():
    config = pipeline_config(stages=3)
    resource = ts.MemoryResource(name="staged", pipeline_config=config)

    @ts.schedule
    def producer_schedule(item):
        with ts.domain_loop(0, 2, 1):
            item.try_acquire()
            item.acquire()
            item.commit()

    task = ts.Task(
        [],
        [resource],
        0,
        1,
        schedule=producer_schedule(resource),
    )
    binding = ts.DevicePipelineBinding.from_config(config)
    device_task = task.to_device({}, pipeline_bindings={resource: binding})

    assert device_task.initial_pipeline_states == (ts.PipelineState(phase=1),)
    assert isinstance(device_task.body[0], DeviceDomainLoop)
    assert [type(node) for node in device_task.body[0].body] == [
        DevicePipelineStep,
        DevicePipelineStep,
        DevicePipelineStep,
    ]
    assert [node.action for node in device_task.body[0].body] == [
        DevicePipelineStep.TRY_ACQUIRE,
        DevicePipelineStep.ACQUIRE,
        DevicePipelineStep.COMMIT,
    ]


def test_generic_pipeline_rejects_unmaterialized_metadata_options():
    config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        2,
        256,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(128),
        interleave_stride=2,
    )
    with pytest.raises(NotImplementedError, match="interleave strides"):
        ts.DevicePipelineBinding.from_config(config)


def test_device_pipeline_binding_supports_two_cta_nvfp4_pipelines():
    tma_umma = ts.PipelineConfig.create_tma_umma_pipeline_cfg(
        5,
        32768,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(1),
        cta_layout_vmnk=(2, 1, 1, 1),
        consumer_signaling_threads=ts.SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=16384,
    )
    tma_binding = ts.DevicePipelineBinding.from_config(tma_umma)
    assert tma_binding.cta_group_size == 2
    assert tma_binding.uses_cluster
    assert not tma_binding.producer_cta_leader
    assert tma_binding.consumer_cta_leader

    umma_async = ts.PipelineConfig.create_umma_async_pipeline_cfg(
        1,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(256),
        cta_layout_vmnk=(2, 1, 1, 1),
        producer_signaling_threads=ts.SignalingThreads.CtaLeader,
    )
    accumulator_binding = ts.DevicePipelineBinding.from_config(umma_async)
    assert accumulator_binding.kind == ts.DevicePipelineBinding.UMMA_ASYNC
    assert accumulator_binding.cta_group_size == 2
    assert accumulator_binding.producer_cta_leader
    assert not accumulator_binding.consumer_cta_leader

    unsupported = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        1,
        128,
        ts.CooperativeGroup(1),
        ts.CooperativeGroup(1),
        cta_layout_vmnk=(2, 1, 1, 1),
    )
    with pytest.raises(NotImplementedError, match="TmaUmma/UmmaAsync"):
        ts.DevicePipelineBinding.from_config(unsupported)


def test_device_guards_use_zero_based_iteration_count_for_strided_loops():
    context = ts.ExecutionContext(
        loop_offset=7,
        iteration_index=0,
        num_iterations=3,
        loop_start=7,
        loop_end=1,
        loop_step=-2,
    )
    assert DeviceGuard("first", 1, 0, -1, False).active(context)
    assert not DeviceGuard("last", 1, 0, -1, False).active(context)
    assert DeviceGuard("every", 2, 0, -1, False).active(context)


def test_package_has_no_cutlass_dependency():
    import inspect
    import task_scheduling.resources as resources

    assert "import cutlass" not in inspect.getsource(resources)
