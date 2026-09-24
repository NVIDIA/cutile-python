# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immediate loop exits through nested conditions and routed device values."""

import importlib
from dataclasses import dataclass

import pytest
import torch

from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_requirements import task_scheduling as ts


# DeviceTask adjusts register limits with setmaxnreg, which Ampere does not support.
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper or newer for DeviceTask register adjustment",
)
LOOP_CASES = [(0, 0), (5, 0), (5, 2), (5, 5)]
NESTED_LOOP_CASES = [(end, stop, 1) for end, stop in LOOP_CASES] + [(5, 0, 0)]


@dataclass(frozen=True)
class Inputs:
    output: object
    stop: object
    enabled: object
    end: object


@dataclass(kw_only=True, eq=False)
class Resource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    @staticmethod
    def should_stop(stage_info):
        return stage_info.loop_offset >= stage_info.context.tasks_inputs.stop

    @ts.consumer_work(outputs=1)
    @staticmethod
    def disabled(stage_info):
        return stage_info.context.tasks_inputs.enabled == 0

    @ts.consumer_work(outputs=1)
    @staticmethod
    def seed(stage_info):
        return cl.int32(10)

    @ts.consumer_work(outputs=1)
    @staticmethod
    def advance(stage_info, state):
        return state + 1

    @ts.producer_work
    @staticmethod
    def record(stage_info, column, value):
        if cl.thread_index(0) == 0:
            stage_info.context.tasks_inputs.output[stage_info.loop_offset, column] = value

    @ts.producer_work
    @staticmethod
    def finish(stage_info, value):
        if cl.thread_index(0) == 0:
            stage_info.context.tasks_inputs.output[7, 3] = value


@pytest.fixture
def resource():
    return Resource(name="data")


def test_host_expansion_stops_at_break_and_preserves_following_work(resource):
    end, stop = 5, 2

    @ts.schedule
    def captured(data):
        with ts.domain_loop(end) as loop:
            data.record(0, 1)
            with ts.every(1, start=stop):
                with ts.when_true(object(), key="exit"):
                    loop.break_loop()
                data.record(1, 1)
            data.record(2, 1)
        data.finish(1)

    tree = captured(resource)
    assert "BreakLoop()" in str(tree)
    visited = []
    tree.visit(visited.append)
    assert sum(isinstance(node, ts.BreakLoop) for node in visited) == 1
    task = ts.Task(0, 1, schedule=tree)
    expanded = ts.expand_task(task, opaque_assignment={"exit": True})
    steps = {node.unique_id: node for node in visited if isinstance(node, ts.Step)}
    columns = [steps[op.unique_id].constexpr_kwargs.get("column") for op in expanded]
    expected = [item for _ in range(min(end, stop)) for item in (0, 2)]
    if stop < end:
        expected.append(0)
    assert columns == expected + [None]
    manager = ts.TaskManager([task], {})
    frozen_break = manager.freeze().tasks[0].body[0].body[1].body[0].body[0]
    assert isinstance(frozen_break, ts.BreakLoopIR)


def test_break_scope_validation():
    with pytest.raises(ts.ScheduleError, match="active|inside"):
        ts.break_loop()

    @ts.schedule
    def outside():
        ts.break_loop()

    with pytest.raises(ts.ScheduleError, match="inside domain_loop"):
        outside()

    @ts.schedule
    def expired():
        with ts.domain_loop(2) as loop:
            pass
        loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="expired"):
        expired()

    tree = ts.Schedule(name="invalid", body=(ts.BreakLoop(),), resources=())
    with pytest.raises(ts.ScheduleError, match="inside domain_loop"):
        ts.Task(0, 1, schedule=tree)


def test_break_only_exits_domain_inside_work_tile_loop(resource):
    queue = ts.WorkQueue(name="queue")

    @ts.schedule
    def captured(data, queue):
        with ts.work_tile_loop(queue):
            with ts.domain_loop(3):
                data.record(0, 1)
                ts.break_loop()
                data.record(1, 1)
            data.finish(1)
            queue.get_and_advance_work_tile()

    tree = captured(resource, queue)
    task = ts.Task(0, 1, schedule=tree)
    expanded = ts.expand_task(task, num_tiles=2)
    assert len(expanded) == 4
    assert [op.unique_id for op in expanded[:2]] == [op.unique_id for op in expanded[2:]]


@pytest.mark.parametrize("consumer_stop", [2, 3])
def test_checker_detects_mismatched_pipeline_exits(consumer_stop):
    pipe = ts.MemoryResource(
        name="pipe",
        pipeline_config=ts.PipelineConfig.create_async_async_pipeline_cfg(
            1, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
        ),
    )

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(4):
            with ts.every(1, start=2):
                ts.break_loop()
            pipe.acquire()
            pipe.producer_work()
            pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(4):
            with ts.every(1, start=consumer_stop):
                ts.break_loop()
            pipe.wait()
            pipe.consumer_work()
            pipe.release()

    tasks = [ts.Task(0, 1, schedule=producer(pipe)), ts.Task(1, 1, schedule=consumer(pipe))]
    result = ts.check_all_interleavings(tasks)
    assert result.is_safe == (consumer_stop == 2)
    if consumer_stop != 2:
        assert result.deadlock_states


def _launch(schedule, *, end, stop, enabled=1, dtype=torch.int32):
    device = ts.Task(0, 1, schedule=schedule).to_device()

    @cl.kernel
    def kernel(output, stop, enabled, end):
        device(device.make_context(Inputs(output, stop, enabled, end)))

    output = torch.zeros((8, 4), dtype=dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (output, stop, enabled, end))
    return output.cpu()


@requires_hopper
@pytest.mark.parametrize("end, stop, enabled", [(5, 2, 1), (5, 5, 1), (5, 0, 0)])
def test_nested_device_break_skips_all_enclosing_suffixes(end, stop, enabled, resource):
    @ts.schedule
    def captured(stage_info, data):
        with ts.domain_loop(stage_info.context.tasks_inputs.end) as loop:
            value = data.seed()
            data.record(0, value)
            stop = data.should_stop()
            with ts.when_true(stop):
                disabled = data.disabled()
                with ts.when_false(disabled):
                    with ts.when_true(stop):
                        loop.break_loop()
                    data.record(1, 99)
                data.record(1, value)
            # A result produced after a potential break must remain well typed
            # even when this entire continuation is skipped.
            updated = data.advance(value)
            data.record(2, updated)
        data.finish(1)
        with ts.domain_loop(2):
            data.record(3, 1)

    actual = _launch(captured(resource), end=end, stop=stop, enabled=enabled)
    expected = torch.zeros((8, 4), dtype=torch.int32)
    completed = min(end, stop) if enabled else end
    visited = min(end, stop + 1) if enabled else end
    expected[:visited, 0] = 10
    if not enabled:
        expected[stop:end, 1] = 10
    expected[:completed, 2] = 11
    expected[7, 3] = 1
    expected[:2, 3] = 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_hopper
@pytest.mark.parametrize("stop_row", [0, 4, 9])
def test_nested_tma_copy_break_on_device(stop_row):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires an SM100-class GPU")
    mod = importlib.import_module(
        "experimental.task_scheduling.tutorial.01_copy_basics.04_copy_tma_nested_conditional"
    )
    mod.run_tma_copy_nested_conditional_kernel_prim(
        (9, 512), stop_row=stop_row, verbose=False
    )


@requires_hopper
def test_unconditional_break_skips_functional_backedge_after_side_effects(resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(state):
            updated = data.advance(state)
            data.record(0, updated)
            ts.break_loop()
            return updated

        result = ts.domain_loop(0, 5, 1, body, initial)
        data.finish(result)

    actual = _launch(captured(resource), end=5, stop=5)
    expected = torch.zeros((8, 4), dtype=torch.int32)
    expected[0, 0] = 11
    expected[7, 3] = 10
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("exit_count", [1, 3])
def test_break_values_must_match_carried_arity(exit_count, resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(left, right):
            ts.break_loop(*([left] * exit_count))
            return left, right

        ts.domain_loop(0, 2, 1, body, initial, initial)

    with pytest.raises(ts.ScheduleError, match=f"expected 2, got {exit_count}"):
        captured(resource)


def test_break_values_require_carried_loop(resource):
    @ts.schedule
    def captured(data):
        value = data.seed()
        with ts.domain_loop(2) as loop:
            loop.break_loop(value)

    with pytest.raises(ts.ScheduleError, match="expected 0, got 1"):
        captured(resource)


@pytest.mark.parametrize("value", [42, None])
def test_break_values_must_be_routed(value, resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(state):
            ts.break_loop(value)
            return state

        ts.domain_loop(0, 2, 1, body, initial)

    with pytest.raises(ts.ScheduleError, match="must be a routed schedule value"):
        captured(resource)


@pytest.mark.parametrize("escape_to", ["parent", "sibling"])
def test_break_values_must_be_visible_at_break(escape_to, resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(state):
            predicate = data.should_stop()
            with ts.when_true(predicate):
                local = data.advance(state)
            if escape_to == "parent":
                ts.break_loop(local)
            else:
                with ts.when_false(predicate):
                    ts.break_loop(local)
            return state

        ts.domain_loop(0, 2, 1, body, initial)

    with pytest.raises(ts.ScheduleError, match="control-flow scope"):
        captured(resource)


def test_break_values_cannot_come_from_another_schedule(resource):
    @ts.schedule
    def other(data):
        data.seed()

    foreign = other(resource).body[0].output_values[0]

    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(state):
            ts.break_loop(foreign)
            return state

        ts.domain_loop(0, 2, 1, body, initial)

    with pytest.raises(ts.ScheduleError, match="another schedule"):
        captured(resource)


def test_break_values_preserve_pipeline_provenance():
    config = ts.PipelineConfig.create_async_async_pipeline_cfg(
        1, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
    )
    left = Resource(name="left", pipeline_config=config)
    right = Resource(name="right", pipeline_config=config)

    @ts.schedule
    def captured(left, right):
        initial = left.seed()

        def body(state):
            ts.break_loop(right.seed())
            return state

        ts.domain_loop(0, 2, 1, body, initial)

    with pytest.raises(ts.ScheduleError, match="changes pipeline stage provenance"):
        captured(left, right)


@requires_hopper
@pytest.mark.parametrize("end, stop, enabled", NESTED_LOOP_CASES)
@pytest.mark.parametrize("explicit", [False, True], ids=["bare", "explicit"])
def test_nested_break_carried_result(end, stop, enabled, explicit, resource):
    @ts.schedule
    def captured(stage_info, data):
        initial = data.seed()

        def body(state):
            stop = data.should_stop()
            with ts.when_true(stop):
                disabled = data.disabled()
                with ts.when_false(disabled):
                    if explicit:
                        exit_state = data.advance(state)
                        ts.break_loop(exit_state)
                    else:
                        ts.break_loop()
                    data.record(1, 99)
            updated = data.advance(state)
            data.record(0, updated)
            return updated

        result = ts.domain_loop(0, stage_info.context.tasks_inputs.end, 1, body, initial)
        data.finish(result)

    actual = _launch(captured(resource), end=end, stop=stop, enabled=enabled)
    completed = min(end, stop) if enabled else end
    broke = enabled and stop < end
    expected = torch.zeros((8, 4), dtype=torch.int32)
    expected[:completed, 0] = torch.arange(11, 11 + completed, dtype=torch.int32)
    expected[7, 3] = 10 + completed + int(broke and explicit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@dataclass(kw_only=True, eq=False)
class MixedStateResource(Resource):
    @ts.consumer_work(outputs=2)
    @staticmethod
    def seed_pair(stage_info):
        return cl.int32(1), cl.float32(0.5)

    @ts.consumer_work(outputs=2)
    @staticmethod
    def advance_pair(stage_info, count, total):
        return count + 1, total + 2.0

    @ts.consumer_work(outputs=2)
    @staticmethod
    def exit_pair(stage_info, count, total):
        return count + 10, total + 0.25

    @ts.producer_work
    @staticmethod
    def finish_pair(stage_info, count, total):
        if cl.thread_index(0) == 0:
            output = stage_info.context.tasks_inputs.output
            output[7, 0] = cl.float32(count)
            output[7, 1] = total


@requires_hopper
@pytest.mark.parametrize("end, stop", [(5, 2), (5, 5)])
def test_explicit_break_returns_multiple_values_with_distinct_types(end, stop):
    resource = MixedStateResource(name="data")

    @ts.schedule
    def captured(stage_info, data):
        count, total = data.seed_pair()

        def body(count, total):
            stop = data.should_stop()
            with ts.when_true(stop):
                exit_count, exit_total = data.exit_pair(count, total)
                ts.break_loop(exit_count, exit_total)
            return data.advance_pair(count, total)

        count, total = ts.domain_loop(
            0, stage_info.context.tasks_inputs.end, 1, body, count, total
        )
        data.finish_pair(count, total)

    actual = _launch(captured(resource), end=end, stop=stop, dtype=torch.float32)
    completed = min(end, stop)
    broke = stop < end
    expected = torch.zeros((8, 4), dtype=torch.float32)
    expected[7, 0] = 1 + completed + (10 if broke else 0)
    expected[7, 1] = 0.5 + 2 * completed + (0.25 if broke else 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_hopper
@pytest.mark.parametrize("stop, expected_result", [(2, 13), (5, 12)])
def test_explicit_and_bare_break_sites_can_share_a_carried_loop(stop, expected_result, resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()

        def body(state):
            stop = data.should_stop()
            with ts.when_true(stop):
                updated = data.advance(state)
                ts.break_loop(updated)
            with ts.every(1, start=2):
                ts.break_loop()
            return data.advance(state)

        result = ts.domain_loop(0, 5, 1, body, initial)
        data.finish(result)

    actual = _launch(captured(resource), end=5, stop=stop)
    expected = torch.zeros((8, 4), dtype=torch.int32)
    expected[7, 3] = expected_result
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_hopper
@pytest.mark.parametrize("stop, expected_result", [(2, 11), (5, 15)])
def test_outer_value_used_only_by_break_stays_live(stop, expected_result, resource):
    @ts.schedule
    def captured(data):
        initial = data.seed()
        exit_value = data.advance(initial)

        def body(state):
            stop = data.should_stop()
            with ts.when_true(stop):
                ts.break_loop(exit_value)
            return data.advance(state)

        result = ts.domain_loop(0, 5, 1, body, initial)
        data.finish(result)

    actual = _launch(captured(resource), end=5, stop=stop)
    expected = torch.zeros((8, 4), dtype=torch.int32)
    expected[7, 3] = expected_result
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
