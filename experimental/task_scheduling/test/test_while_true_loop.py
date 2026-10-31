# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture, verification, and device lowering of ``while_true_loop()``."""

from dataclasses import dataclass

import pytest
import torch

from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_requirements import task_scheduling as ts
from task_scheduling.exhaustive_checker import format_opaque_assignment
from task_scheduling.schedule_builder import _iter_nodes


# DeviceTask adjusts register limits with setmaxnreg, which Ampere does not support.
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper or newer for DeviceTask register adjustment",
)


@dataclass(frozen=True)
class Inputs:
    output: object
    start: object
    stop: object
    enabled: object


@dataclass(kw_only=True, eq=False)
class Resource(ts.MemoryResource):
    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def cursor_init(stage_info):
        return stage_info.context.tasks_inputs.start

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def cursor_advance(stage_info):
        return stage_info.loop_offset + 1

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def done(stage_info):
        return stage_info.loop_offset >= stage_info.context.tasks_inputs.stop

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def enabled(stage_info):
        return stage_info.context.tasks_inputs.enabled != 0

    @ts.consumer_work(outputs=1)
    @staticmethod
    def seed(stage_info):
        return cl.int32(10)

    @ts.consumer_work(outputs=1)
    @staticmethod
    def advance(stage_info, state):
        return state + 1

    @ts.consumer_work(outputs=2)
    @staticmethod
    def two_values(stage_info):
        return cl.int32(0), cl.int32(1)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def routed_hook(stage_info, state):
        return state

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def producer_hook(stage_info):
        return cl.int32(0)

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


def _pipe(name="pipe", stages=1):
    return Resource(
        name=name,
        pipeline_config=ts.PipelineConfig.create_async_async_pipeline_cfg(
            stages, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
        ),
    )


def _capture(body, resource):
    @ts.schedule
    def captured(data):
        body(data)

    return captured(resource)


def _retry_loop(data, *, verify_num_iters=None, **loop_kwargs):
    return ts.while_true_loop(
        init=data.cursor_init,
        advance=data.cursor_advance,
        verify_num_iters=verify_num_iters,
        **loop_kwargs,
    )


# -----------------------------------------------------------------------------
# Capture and builder validation
# -----------------------------------------------------------------------------


def test_capture_records_hooks_breaks_and_exit_formula(resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=(1, 3)) as loop:
            done = data.done()
            with ts.when_true(done, key="done"):
                data.record(1, 1)
                loop.break_loop()
                data.record(2, 1)
            with ts.when_false(done, key="done", verify_num_fires=2):
                data.record(0, 1)
        data.finish(1)

    tree = _capture(body, resource)
    loop = tree.body[0]
    assert isinstance(loop, ts.WhileTrueLoop)
    assert loop.verify_num_iters == (1, 3)
    assert (loop.init.label, loop.advance.label) == ("cursor_init", "cursor_advance")
    # Hook IDs bracket the body so numbering follows execution order.
    body_ids = [
        node.unique_id for node in _iter_nodes(loop.body) if isinstance(node, ts.Step)
    ]
    assert len(body_ids) == 4
    assert loop.init.unique_id < min(body_ids) < max(body_ids) < loop.advance.unique_id
    text = str(tree)
    assert "WhileTrueLoop(init=data.cursor_init, advance=data.cursor_advance" in text
    assert "exit=when_true(key='done')" in text
    assert "verify_num_fires=2" in text
    assert "init Step(data, ConsumerAuxWork, label='cursor_init')" in text
    assert "BreakLoop()" in text

    task = ts.Task(0, 1, schedule=tree)
    assert task.src_resources == [resource]
    frozen = ts.TaskManager([task], {}).freeze().tasks[0].body[0]
    assert isinstance(frozen, ts.WhileTrueLoopIR)
    assert frozen.init.label == "cursor_init"
    assert frozen.body[2].verify_num_fires == 2


def test_cannot_nest_in_domain_loop_or_condition(resource):
    def in_domain(data):
        with ts.domain_loop(2):
            with _retry_loop(data) as loop:
                loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="cannot be nested inside domain_loop"):
        _capture(in_domain, resource)

    def in_condition(data):
        with ts.when_true(data.done()):
            with _retry_loop(data, verify_num_iters=1) as loop:
                loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="inside a condition block"):
        _capture(in_condition, resource)


def test_requires_break_loop(resource):
    def body(data):
        with _retry_loop(data):
            data.record(0, 1)

    with pytest.raises(ts.ScheduleError, match="requires at least one loop.break_loop"):
        _capture(body, resource)


def test_domain_loop_cannot_nest_inside_while_loop(resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=1) as loop:
            with ts.domain_loop(2):
                pass
            loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="nested inside while_true_loop"):
        _capture(body, resource)


def test_cannot_mix_domain_and_while_loops(resource):
    def body(data):
        with ts.domain_loop(2):
            data.record(0, 1)
        with _retry_loop(data, verify_num_iters=1) as loop:
            loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="cannot both appear in one schedule"):
        _capture(body, resource)


def test_at_most_one_while_loop(resource):
    def body(data):
        for _ in range(2):
            with _retry_loop(data, verify_num_iters=1) as loop:
                loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="at most one while_true_loop"):
        _capture(body, resource)


def test_skippable_required_with_skip_if(resource):
    queue = ts.WorkQueue(name="queue")

    def capture(skippable):
        @ts.schedule
        def captured(data, queue):
            with ts.work_tile_loop(queue, skip_if=lambda *_: False) as wtl:
                if skippable:
                    with wtl.skippable():
                        with _retry_loop(data, verify_num_iters=1) as loop:
                            loop.break_loop()
                else:
                    with _retry_loop(data, verify_num_iters=1) as loop:
                        loop.break_loop()
                queue.get_and_advance_work_tile()

        return captured(resource, queue)

    with pytest.raises(ts.ScheduleError, match="must be inside wtl.skippable"):
        capture(False)
    task = ts.Task(0, 1, schedule=capture(True))
    assert ts.expand_task(task, skipped_tile=True) == []


def test_unconditional_break_requires_single_modeled_round(resource):
    def body(data, verify_num_iters):
        with _retry_loop(data, verify_num_iters=verify_num_iters) as loop:
            data.record(0, 1)
            loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="unconditional"):
        _capture(lambda data: body(data, 2), resource)
    tree = _capture(lambda data: body(data, 1), resource)
    expanded = ts.expand_task(ts.Task(0, 1, schedule=tree))
    assert [op.schedule_stage for op in expanded] == [ts.ScheduleStage.ProducerWork]


@pytest.mark.parametrize(
    "verify_num_iters, expected",
    [(None, (1, 2)), (3, (3,)), ((1, 4), (1, 4))],
)
def test_verify_num_iters_normalization(verify_num_iters, expected, resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=verify_num_iters) as loop:
            with ts.when_true(data.done()):
                loop.break_loop()

    assert _capture(body, resource).body[0].verify_num_iters == expected


@pytest.mark.parametrize(
    "verify_num_iters, message",
    [(0, "positive"), ((), "non-empty"), ((1, -1), "positive"), ([1], "an int")],
)
def test_invalid_verify_num_iters(verify_num_iters, message, resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=verify_num_iters) as loop:
            loop.break_loop()

    with pytest.raises(ts.ScheduleError, match=message):
        _capture(body, resource)


@pytest.mark.parametrize(
    "hook, message",
    [
        (lambda data: lambda: None, "traced resource work method"),
        (lambda data: data.producer_hook, "consumer work method"),
        (lambda data: data.two_values, "outputs=1"),
        (lambda data: data.routed_hook, "cannot take routed inputs"),
    ],
)
def test_hook_validation(hook, message, resource):
    def body(data):
        with ts.while_true_loop(init=hook(data), advance=data.cursor_advance) as loop:
            loop.break_loop()

    with pytest.raises(ts.ScheduleError, match=f"init= .*{message}"):
        _capture(body, resource)


def test_break_handle_lifetime_and_values(resource):
    def expired(data):
        with _retry_loop(data, verify_num_iters=1) as loop:
            loop.break_loop()
        loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="expired"):
        _capture(expired, resource)

    def with_value(data):
        value = data.seed()
        with _retry_loop(data, verify_num_iters=1):
            ts.break_loop(value)

    with pytest.raises(ts.ScheduleError, match="takes no values"):
        _capture(with_value, resource)


def test_verify_num_fires_validation(resource):
    def outside(data):
        with ts.domain_loop(2):
            with ts.when_true(data.done(), verify_num_fires=1):
                pass

    with pytest.raises(ts.ScheduleError, match="only valid inside while_true_loop"):
        _capture(outside, resource)

    def negative(data):
        with _retry_loop(data) as loop:
            with ts.when_true(data.done(), verify_num_fires=-1):
                loop.break_loop()

    with pytest.raises(ts.ScheduleError, match="non-negative int"):
        _capture(negative, resource)


def test_break_clause_captures_nested_guards(resource):
    def body(data):
        with _retry_loop(data) as loop:
            with ts.when_true(data.done(), key="done"):
                with ts.when_false(data.enabled(), key="enabled"):
                    loop.break_loop()
            with ts.when_true(object(), key="error"):
                loop.break_loop()

    text = str(_capture(body, resource))
    assert (
        "exit=(when_true(key='done') and when_false(key='enabled')) or "
        "when_true(key='error')"
    ) in text


# -----------------------------------------------------------------------------
# Exhaustive-checker modeling
# -----------------------------------------------------------------------------


def _round_trip_task(data, *, verify_num_iters, extra_key=False, progress=True):
    def body(data):
        with _retry_loop(data, verify_num_iters=verify_num_iters) as loop:
            done = data.done()
            with ts.when_true(done, key="done"):
                loop.break_loop()
            with ts.when_false(done, key="done"):
                if progress:
                    data.wait()
                    data.consumer_work()
                    data.release()
                else:
                    data.enabled()
                if extra_key:
                    with ts.when_true(object(), key="side"):
                        data.record(0, 1)

    return ts.Task(0, 1, schedule=_capture(body, data))


def _histories(task):
    keys = ts.exhaustive_checker.collect_opaque_keys([task])
    return [
        scenario.for_task(task)
        for scenario in ts.opaque_assignment_sequences([task], keys)
    ]


def test_sequences_split_retry_and_break_rounds(resource):
    task = _round_trip_task(resource, verify_num_iters=(1, 2, 3))
    assert _histories(task) == [
        [{"done": True}],
        [{"done": False}, {"done": True}],
        [{"done": False}, {"done": False}, {"done": True}],
    ]


def test_other_keys_are_enumerated_in_every_round(resource):
    task = _round_trip_task(resource, verify_num_iters=2, extra_key=True)
    histories = _histories(task)
    assert len(histories) == 4
    assert all(
        [round_["done"] for round_ in history] == [False, True] for history in histories
    )
    assert {
        (history[0]["side"], history[1]["side"]) for history in histories
    } == {(False, False), (False, True), (True, False), (True, True)}
    assert " -> " in format_opaque_assignment(histories[0])


def test_sibling_breaks_form_a_disjunction(resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=2) as loop:
            with ts.when_true(object(), key="a"):
                loop.break_loop()
            with ts.when_true(object(), key="b"):
                loop.break_loop()
            data.wait()
            data.consumer_work()
            data.release()

    task = ts.Task(0, 1, schedule=_capture(body, resource))
    histories = _histories(task)
    assert all(history[0] == {"a": False, "b": False} for history in histories)
    assert [history[1] for history in histories] == [
        {"a": False, "b": True},
        {"a": True, "b": False},
        {"a": True, "b": True},
    ]


def test_contradictory_break_is_a_hard_error(resource):
    def body(data):
        with _retry_loop(data) as loop:
            done = data.done()
            with ts.when_true(done):
                with ts.when_false(done):
                    loop.break_loop()

    task = ts.Task(0, 1, schedule=_capture(body, resource))
    with pytest.raises(ValueError, match="no opaque assignment that satisfies"):
        _histories(task)


def test_spinning_retry_round_has_no_progress(resource):
    task = _round_trip_task(resource, verify_num_iters=2, progress=False)
    with pytest.raises(ValueError, match="no blocking or state-changing"):
        ts.check_all_interleavings([task], opaque_assignment=_histories(task)[0])


def test_round_structure_is_checked_against_history(resource):
    task = _round_trip_task(resource, verify_num_iters=2)
    with pytest.raises(ValueError, match="does not reach break_loop"):
        ts.expand_task(task, opaque_assignment=[{"done": False}, {"done": False}])
    with pytest.raises(ValueError, match="retry while_true_loop.* reaches break"):
        ts.expand_task(task, opaque_assignment=[{"done": True}, {"done": True}])


def test_fire_count_assertion_filters_histories(resource):
    def body(data, fires):
        with _retry_loop(data, verify_num_iters=(2, 3)) as loop:
            done = data.done()
            with ts.when_true(done, key="done"):
                loop.break_loop()
            with ts.when_false(done, key="done", verify_num_fires=fires):
                data.wait()
                data.consumer_work()
                data.release()

    task = ts.Task(0, 1, schedule=_capture(lambda data: body(data, 2), resource))
    assert _histories(task) == [[{"done": False}, {"done": False}, {"done": True}]]
    task = ts.Task(0, 1, schedule=_capture(lambda data: body(data, 5), resource))
    with pytest.raises(ValueError, match="exactly 2 of 3 modeled rounds"):
        _histories(task)


def _producer_task(pipe, side):
    @ts.schedule
    def producer(pipe, side):
        with ts.domain_loop(1):
            pipe.acquire()
            pipe.producer_work()
            pipe.commit()
        with ts.when_true(object(), key="done"):
            side.record(0, 1)

    return ts.Task(1, 1, schedule=producer(pipe, side), name="LoadTask")


@pytest.mark.parametrize("verify_num_iters, safe", [(2, True), (3, False)])
def test_partner_observes_exit_round_and_extra_retry_deadlocks(verify_num_iters, safe):
    pipe = _pipe()
    consumer = _round_trip_task(pipe, verify_num_iters=verify_num_iters)
    producer = _producer_task(pipe, Resource(name="side"))
    (scenario,) = ts.opaque_assignment_sequences([producer, consumer], ["done"])
    assert isinstance(scenario, ts.OpaqueAssignmentScenario)
    assert scenario.default_assignment == {"done": True}
    assert scenario.for_task(producer) == {"done": True}
    assert scenario.for_task(consumer)[-1] == {"done": True}
    assert len(scenario.for_task(consumer)) == verify_num_iters
    producer_ops = ts.expand_task(producer, opaque_assignment=scenario)
    assert producer_ops[-1].phase == "I"
    result = ts.check_all_interleavings(
        [producer, consumer], opaque_assignment=scenario
    )
    assert result.is_safe == safe
    if not safe:
        assert result.deadlock_states
        with pytest.raises(ValueError, match="done=False -> done=False -> done=True"):
            ts.TaskManager([producer, consumer], {})


# -----------------------------------------------------------------------------
# Device lowering
# -----------------------------------------------------------------------------


def _launch(schedule, *, start, stop, enabled=1):
    device = ts.Task(0, 1, schedule=schedule).to_device()

    @cl.kernel
    def kernel(output, start, stop, enabled):
        device(device.make_context(Inputs(output, start, stop, enabled)))

    output = torch.zeros((8, 4), dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(), (1,), (32,), kernel, (output, start, stop, enabled)
    )
    return output.cpu()


def test_host_interpreter_rejects_cursor_hooks(resource):
    def body(data):
        with _retry_loop(data, verify_num_iters=1) as loop:
            loop.break_loop()

    tree = _capture(body, resource)
    node = tree.body[0]
    with pytest.raises(NotImplementedError, match="frozen device lowering"):
        ts.Task(0, 1, schedule=tree)._run_node(node, ts.ExecutionContext())


@requires_hopper
@pytest.mark.parametrize(
    "start, stop, enabled", [(0, 0, 1), (3, 1, 1), (1, 5, 1), (2, 6, 0)]
)
def test_device_cursor_and_break_skip_round_suffix(start, stop, enabled, resource):
    @ts.schedule
    def captured(data):
        value = data.seed()
        with _retry_loop(data) as loop:
            data.record(0, value)
            done = data.done()
            with ts.when_true(done):
                on = data.enabled()
                with ts.when_true(on):
                    loop.break_loop()
                data.record(1, value)
            # Produced after a potential break; skipped in the exit round.
            updated = data.advance(value)
            data.record(2, updated)
            # With breaking disabled above, this later site ends the round.
            with ts.when_true(data.done()):
                loop.break_loop()
        data.finish(value)

    actual = _launch(captured(resource), start=start, stop=stop, enabled=enabled)
    # The cursor starts at ``start``; every round records column 0, and the
    # exit round is the first with cursor >= stop.
    last = max(start, stop)
    expected = torch.zeros((8, 4), dtype=torch.int32)
    expected[start:last + 1, 0] = 10
    expected[start:last, 2] = 11
    if not enabled:
        expected[last, 1] = 10
        expected[last, 2] = 11
    expected[7, 3] = 10
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
