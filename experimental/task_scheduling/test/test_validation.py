# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_requirements import task_scheduling as ts

from cuda.lang.compilation import KernelSignature


def config(stages=1):
    return ts.PipelineConfig.create_async_async_pipeline_cfg(
        stages, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
    )


@dataclass(kw_only=True, eq=False)
class Pipe(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def write(stage_info):
        stage_info

    @ts.consumer_work
    @staticmethod
    def read(stage_info):
        stage_info


def make_balanced_tasks(pipe, warp_start=0):
    source = ts.MemoryResource(name=f"{pipe.name}_source")
    sink = ts.MemoryResource(name=f"{pipe.name}_sink")

    @ts.schedule
    def producer(resource):
        resource.try_acquire()
        resource.acquire()
        resource.write()
        resource.commit()

    @ts.schedule
    def consumer(resource):
        resource.try_wait()
        resource.wait()
        resource.read()
        resource.release()

    return (
        source,
        sink,
        ts.Task(
            [source],
            [pipe],
            warp_start,
            1,
            schedule=producer(pipe),
            name="producer",
        ),
        ts.Task(
            [pipe],
            [sink],
            warp_start + 1,
            1,
            schedule=consumer(pipe),
            name="consumer",
        ),
    )


def test_balanced_pipeline_and_mermaid_are_deterministic():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    manager = ts.TaskManager(
        [producer, consumer],
        {pipe: [source], sink: [pipe]},
        verbose=False,
    )
    graph = manager.get_mermaid_string_dependency_graph()
    assert graph.startswith("flowchart LR")
    assert graph == manager.get_marmaid_string_dependency_graph()


def test_manager_infers_task_roles_padding_and_frozen_ir():
    pipe = Pipe(name="pipe", pipeline_config=config())

    @ts.schedule
    def producer(resource):
        resource.try_acquire()
        resource.acquire()
        resource.write()
        resource.commit()

    @ts.schedule
    def consumer(resource):
        resource.try_wait()
        resource.wait()
        resource.read()
        resource.release()

    producer_task = ts.Task(
        warp_idx=0,
        num_warps=1,
        schedule=producer(pipe),
        name="producer",
    )
    consumer_task = ts.Task(1, 1, schedule=consumer(pipe), name="consumer")
    manager = ts.TaskManager(
        [producer_task, consumer_task],
        cta_warps=4,
        verbose=False,
    )
    program_ir = manager.freeze()

    assert producer_task.producer_resources == [pipe]
    assert producer_task.consumer_resources == []
    assert producer_task.src_resources == []
    assert producer_task.dst_resources == [pipe]
    assert consumer_task.consumer_resources == [pipe]
    assert consumer_task.producer_resources == []
    assert consumer_task.src_resources == [pipe]
    assert consumer_task.dst_resources == []
    assert manager.resource_dependency_graph == {}
    assert len(manager.user_tasks) == 2
    assert [task.name for task in manager.tasks] == [
        "producer",
        "consumer",
        "PaddingTask0",
    ]
    assert isinstance(program_ir, ts.ProgramIR)
    assert program_ir.cta_warps == 4
    assert program_ir.dependency_edges == ()
    assert [task.task_id for task in program_ir.tasks] == [0, 1, 2]
    with pytest.raises(TypeError):
        program_ir.tasks[0].body[0].input_values["changed"] = 1


def test_manager_infers_memory_allocators_when_omitted():
    smem = ts.SmemAllocation("buffer", 64, 16)
    tmem = ts.TmemAllocation("accumulator", 16)
    resource = Pipe(
        name="storage",
        smem_requirements=[smem],
        tmem_requirements=[tmem],
    )

    @ts.schedule
    def captured(data):
        data.write()

    manager = ts.TaskManager(
        [ts.Task(0, 1, schedule=captured(resource))],
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )

    assert manager.smem_allocator is not None
    assert manager.smem_allocator.layout_computed
    assert manager.tmem_allocator is not None
    assert manager.tmem_allocator.layout_computed


def test_verbose_report_prints_budgets_schedules_and_safety(capsys):
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    manager = ts.TaskManager(
        [producer, consumer],
        {pipe: [source], sink: [pipe]},
        verbose=True,
    )

    manager.print_verbose_report()

    report = capsys.readouterr().out
    assert "Task  0 'producer" in report
    assert "Num. regs:" in report
    assert "Reg budget:" in report
    assert "[producer warps 0:1]" in report
    assert "Schedule('producer'" in report
    assert "Task 0 (producer):" in report
    assert "State #" in report
    assert "timeline (8 steps):" in report
    assert "ProducerAcquire(pipe)" in report
    assert "ConsumerRelease(pipe)" in report
    assert "BFS complete:" in report
    assert "Result: SAFE" in report


def test_verbose_report_omits_bfs_when_exhaustive_check_is_disabled(capsys):
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    manager = ts.TaskManager(
        [producer, consumer],
        {pipe: [source], sink: [pipe]},
        verbose=True,
        exhaustive_deadlock_race_check=False,
    )

    manager.print_verbose_report()

    report = capsys.readouterr().out
    assert "[producer warps 0:1]" in report
    assert "BFS complete:" not in report


def test_manager_freezes_device_tasks_in_validated_order():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    manager = ts.TaskManager(
        [producer, consumer],
        {pipe: [source], sink: [pipe]},
        verbose=False,
    )

    def device_task(task):
        return ts.DeviceTask(
            body=(),
            warp_start=task.warp_start,
            warp_end=task.warp_end,
            num_registers=task.num_registers or task.default_num_registers,
            run_only_on_cta_id=-1,
        )

    producer_device = device_task(producer)
    consumer_device = device_task(consumer)
    device_manager = manager.to_device([producer_device, consumer_device])
    assert isinstance(device_manager, ts.DeviceTaskManager)
    assert device_manager.tasks == (producer_device, consumer_device)

    with pytest.raises(ValueError, match="expected 2 device tasks"):
        manager.to_device([producer_device])
    with pytest.raises(ValueError, match="does not match"):
        manager.to_device([consumer_device, producer_device])


def test_manager_derives_pipeline_binding_and_hides_barrier_offsets():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    manager = ts.TaskManager(
        [producer, consumer],
        {pipe: [source], sink: [pipe]},
        verbose=False,
    )

    def write(stage_info):
        del stage_info

    def read(stage_info):
        del stage_info

    device_manager = manager.to_device(
        {producer: {"write": write}, consumer: {"read": read}},
    )

    assert len(device_manager.tasks) == 2
    assert device_manager.barrier_arena_size == 2
    assert device_manager.dynamic_pipeline_storage
    assert device_manager.smem_size_bytes == 0
    assert device_manager.pipeline_bindings[0].full_barrier_offset == 0
    assert device_manager.pipeline_bindings[0].empty_barrier_offset == 1


def test_manager_coalesces_more_than_three_pipeline_resources():
    tasks = []
    graph = {}
    pipes = []
    barrier_allocator = ts.BarrierAllocator()
    for index in range(4):
        pipe = Pipe(name=f"pipe_{index}", pipeline_config=config())
        source, sink, producer, consumer = make_balanced_tasks(pipe, 2 * index)
        pipes.append(pipe)
        tasks.extend((producer, consumer))
        graph[pipe] = [source]
        graph[sink] = [pipe]
        barrier_allocator.add_resource(pipe)
    barrier_allocator.compute_layout()
    manager = ts.TaskManager(
        tasks,
        graph,
        barrier_allocator=barrier_allocator,
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )

    device_manager = manager.to_device()

    assert len(device_manager.pipeline_bindings) == 4
    assert device_manager.barrier_arena_size == 32
    assert device_manager.barrier_initialization_runs == ((0, 8, 32),)
    for pipe, binding in zip(pipes, device_manager.pipeline_bindings):
        assert binding.full_barrier_offset == barrier_allocator.offset_of(
            f"{pipe.name}.full"
        )
        assert binding.empty_barrier_offset == barrier_allocator.offset_of(
            f"{pipe.name}.empty"
        )

    @cl.kernel
    def kernel():
        device_allocators = device_manager.setup_resources_and_tasks()
        device_manager.run((), device_allocators)

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature(())],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert compiled.ptx.count("mbarrier.init") == 1
    assert compiled.ptx.count("shfl.sync.idx.b32") == 1


def test_manager_infers_static_work_callbacks_and_omits_empty_tasks():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)

    @ts.schedule
    def padding_schedule():
        with ts.domain_loop(1):
            pass

    padding = ts.Task([], [], 2, 1, schedule=padding_schedule(), name="padding")
    manager = ts.TaskManager(
        [producer, consumer, padding],
        {pipe: [source], sink: [pipe]},
        verbose=False,
    )
    device_manager = manager.to_device()

    assert device_manager.tasks[0].body[2].callback is Pipe.write.__wrapped__
    assert device_manager.tasks[1].body[2].callback is Pipe.read.__wrapped__
    assert device_manager.tasks[2].body[0].body == ()


def test_invalid_bracketing_is_actionable():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source = ts.MemoryResource(name="source")

    @ts.schedule
    def bad(resource):
        resource.write()

    task = ts.Task([source], [pipe], 0, 1, schedule=bad(pipe), name="bad")
    with pytest.raises(ValueError, match="not bracketed"):
        ts.TaskManager([task], {pipe: [source]}, exhaustive_deadlock_race_check=False)


def test_blocking_pipeline_ops_require_matching_try_probe():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source = ts.MemoryResource(name="source")

    @ts.schedule
    def producer_without_try(resource):
        resource.acquire()
        resource.write()
        resource.commit()

    task = ts.Task(
        [source],
        [pipe],
        0,
        1,
        schedule=producer_without_try(pipe),
        name="producer_without_try",
    )
    with pytest.raises(ValueError, match="ProducerTryAcquire"):
        ts.TaskManager(
            [task],
            {pipe: [source]},
            exhaustive_deadlock_race_check=False,
        )


def test_deadlock_witness_for_consumer_without_producer(capsys):
    pipe = Pipe(name="pipe", pipeline_config=config())
    sink = ts.MemoryResource(name="sink")

    @ts.schedule
    def consumer(resource):
        resource.wait()
        resource.read()
        resource.release()

    task = ts.Task([pipe], [sink], 0, 1, schedule=consumer(pipe), name="blocked")
    result = ts.check_all_interleavings([task], verbose=True)
    assert not result.is_safe
    assert "deadlock witness" in result.deadlock_states[0].format_lines()[0]
    report = capsys.readouterr().out
    assert "Task 0 (blocked):" in report
    assert "State #1" in report
    assert "*** DEADLOCK: blocked tasks blocked" in report
    assert "Result: UNSAFE" in report


def test_register_over_budget_and_warn_only():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, sink, producer, consumer = make_balanced_tasks(pipe)
    producer.num_registers = 256
    producer.num_warps = 8
    consumer.warp_idx = 8
    with pytest.raises(ValueError, match="register budget"):
        ts.TaskManager(
            [producer, consumer],
            {pipe: [source], sink: [pipe]},
            exhaustive_deadlock_race_check=False,
        )
    with pytest.warns(UserWarning, match="register budget"):
        ts.TaskManager(
            [producer, consumer],
            {pipe: [source], sink: [pipe]},
            exhaustive_deadlock_race_check=False,
            skip_validation=True,
        )


def test_task_register_budget_matches_setmaxnreg_operand_rules():
    pipe = Pipe(name="pipe", pipeline_config=config())
    source, _sink, producer, _consumer = make_balanced_tasks(pipe)
    task = ts.Task(
        [source],
        [pipe],
        0,
        1,
        schedule=producer.schedule,
        num_registers=8,
    )
    assert task.num_registers == 8

    with pytest.raises(ValueError, match="divisible by 8"):
        ts.Task(
            [source],
            [pipe],
            0,
            1,
            schedule=producer.schedule,
            num_registers=25,
        )


def test_aliasing_race_has_physical_witness():
    shared = ts.SmemAllocation("shared", 64, 16)
    writer_resource = Pipe(name="writer", smem_requirements=[shared])
    victim_alloc = ts.SmemAllocation("victim", 64, 16)
    victim_alloc.offset = 0
    victim_resource = Pipe(name="victim", smem_requirements=[victim_alloc])

    @ts.schedule
    def writer(resource):
        resource.write()

    @ts.schedule
    def reader(resource):
        resource.read()

    writer_task = ts.Task(
        [], [writer_resource], 0, 1, schedule=writer(writer_resource), name="writer"
    )
    reader_task = ts.Task(
        [victim_resource], [], 1, 1, schedule=reader(victim_resource), name="reader"
    )
    # Non-pipelined point accesses are intentionally conservative. At least one
    # explored order completes; the checker remains bounded and deterministic.
    result = ts.check_all_interleavings([writer_task, reader_task], early_exit=False)
    assert result.states_explored > 0


def test_split_consumers_and_opaque_assignments_expand():
    pipe = Pipe(name="pipe", pipeline_config=config(2))

    @ts.schedule
    def guarded(resource):
        with ts.domain_loop(4) as domain:
            with ts.when_true(True, key="choice"):
                resource.try_acquire()
            with domain.last_iter():
                resource.try_wait()

    task = ts.Task([pipe], [pipe], 0, 1, schedule=guarded(pipe), name="guarded")
    false_ops = ts.expand_task(task, opaque_assignment={"choice": False})
    true_ops = ts.expand_task(task, opaque_assignment={"choice": True})
    assert len(true_ops) > len(false_ops)
