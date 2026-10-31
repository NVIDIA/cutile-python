# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture, verification, and device lowering of ``&`` / ``|`` / ``~`` guards."""

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_requirements import task_scheduling as ts
from task_scheduling.enums import guard_fires
from task_scheduling.exhaustive_checker import (
    _all_opaque_assignments,
    collect_opaque_keys,
)


# DeviceTask adjusts register limits with setmaxnreg, which Ampere does not support.
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper or newer for DeviceTask register adjustment",
)


@dataclass(frozen=True)
class Inputs:
    output: object
    count: object
    mask_a: object
    mask_b: object


@dataclass(kw_only=True, eq=False)
class Resource(ts.MemoryResource):
    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def bit_a(stage_info):
        mask = stage_info.context.tasks_inputs.mask_a
        return ((mask >> stage_info.loop_offset) & 1) != 0

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def bit_b(stage_info):
        mask = stage_info.context.tasks_inputs.mask_b
        return ((mask >> stage_info.loop_offset) & 1) != 0

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=2)
    @staticmethod
    def probe(stage_info):
        return cl.int32(1) != 0, cl.int32(0) != 0

    @ts.producer_work
    @staticmethod
    def record(stage_info, column):
        if cl.thread_index(0) == 0:
            stage_info.context.tasks_inputs.output[stage_info.loop_offset, column] = 1


@pytest.fixture
def resource():
    return Resource(name="data")


def _capture(body, resource):
    @ts.schedule
    def captured(data):
        body(data)

    return captured(resource)


def _conditions(tree):
    nodes = []
    tree.visit(nodes.append)
    return [node.condition for node in nodes if isinstance(node, ts.ConditionalBlock)]


# -----------------------------------------------------------------------------
# Capture
# -----------------------------------------------------------------------------


def test_boolean_guard_expression_and_or_not(resource):
    def body(data):
        with ts.domain_loop(4):
            first, second = data.probe()
            with ts.when_true((first & ~second) | second):
                data.record(0)

    (guard,) = _conditions(_capture(body, resource))
    assert isinstance(guard, ts.GuardOr)
    conjunction, right = guard.operands
    assert isinstance(conjunction, ts.GuardAnd)
    left, negated = conjunction.operands
    assert isinstance(left, ts.OpaqueCondition) and not left.negated
    assert isinstance(negated, ts.OpaqueCondition) and negated.negated
    assert negated.route_token is right.route_token


def test_condition_bool_and_python_and_rejected(resource):
    captured = []
    _capture(lambda data: captured.extend(data.probe()), resource)
    first, second = captured
    with pytest.raises(ts.ScheduleError, match="cannot be converted to bool"):
        bool(first)
    with pytest.raises(ts.ScheduleError, match="cannot be converted to bool"):
        first and second
    with pytest.raises(ts.ScheduleError, match="cannot be converted to bool"):
        bool(ts.FIRST_ITER | first)


def test_out_of_scope_value_inside_expression_rejected(resource):
    def body(data):
        with ts.domain_loop(4):
            first, _ = data.probe()
            with ts.when_true(first):
                inner, _ = data.probe()
            with ts.when_true(first & inner):
                data.record(0)

    with pytest.raises(ts.ScheduleError, match="outside the control-flow scope"):
        _capture(body, resource)


def test_first_iter_mixed_with_runtime_value(resource):
    def body(data):
        with ts.domain_loop(4) as d:
            ready = data.bit_a()
            with ts.when_true(d.first & ready):
                data.record(0)

    (guard,) = _conditions(_capture(body, resource))
    assert isinstance(guard, ts.GuardAnd)
    assert guard.operands[0] is ts.FIRST_ITER
    assert isinstance(guard.operands[1], ts.OpaqueCondition)


def test_iteration_leaf_requires_domain_loop(resource):
    def body(data):
        ready = data.bit_a()
        with ts.when_true(ready | ts.LAST_ITER):
            data.record(0)

    with pytest.raises(ts.ScheduleError, match="iteration predicates require"):
        _capture(body, resource)


@pytest.mark.parametrize(
    "combine",
    [
        pytest.param(lambda block, ready: block & ready, id="and"),
        pytest.param(lambda block, ready: ready | block, id="ror"),
        pytest.param(lambda block, ready: ~block, id="invert"),
        pytest.param(lambda block, ready: block and ready, id="python-and"),
        pytest.param(lambda block, ready: block, id="bare"),
    ],
)
@pytest.mark.parametrize("api", ["first_iter", "last_iter", "every"])
def test_context_manager_is_not_condition_operand(api, combine, resource):
    def body(data):
        with ts.domain_loop(4) as d:
            ready = data.bit_a()
            block = d.every(2) if api == "every" else getattr(d, api)()
            with ts.when_true(combine(block, ready)):
                pass

    with pytest.raises(ts.ScheduleError, match="context manager, not a condition"):
        _capture(body, resource)


def test_key_on_composite_correlates_all_opaque_leaves(resource):
    def body(data):
        with ts.domain_loop(4):
            first, second = data.probe()
            with ts.when_true((first & second) | ts.LAST_ITER, key="shared"):
                pass

    (guard,) = _conditions(_capture(body, resource))
    conjunction, last = guard.operands
    assert last is ts.LAST_ITER
    assert all(leaf.key == "shared" for leaf in conjunction.operands)


def test_key_on_iteration_only_composite_rejected(resource):
    def body(data):
        with ts.domain_loop(4):
            with ts.when_true(ts.FIRST_ITER | ts.LAST_ITER, key="unused"):
                data.record(0)

    with pytest.raises(ts.ScheduleError, match="requires at least one opaque leaf"):
        _capture(body, resource)


def test_when_false_mixed_tree_uses_negation_normal_form(resource):
    def body(data):
        with ts.domain_loop(4) as d:
            ready = data.bit_a()
            with ts.when_false(d.first & ready):
                pass

    (guard,) = _conditions(_capture(body, resource))
    assert isinstance(guard, ts.GuardOr)
    first, ready = guard.operands
    assert isinstance(first, ts.ComplementedIterationPredicate)
    assert first.predicate is ts.FIRST_ITER
    assert isinstance(ready, ts.OpaqueCondition) and ready.negated


def test_when_false_of_negated_leaf_restores_polarity(resource):
    def body(data):
        with ts.domain_loop(4):
            ready = data.bit_a()
            with ts.when_false(~ready):
                pass
            with ts.when_false(ready | ~ready):
                pass

    single, conjunction = _conditions(_capture(body, resource))
    assert isinstance(single, ts.OpaqueCondition) and not single.negated
    assert isinstance(conjunction, ts.GuardAnd)
    assert [leaf.negated for leaf in conjunction.operands] == [True, False]


def test_iteration_predicate_inversion_is_boolean_complement():
    complement = ~ts.FIRST_ITER
    assert isinstance(complement, ts.ComplementedIterationPredicate)
    assert complement.predicate is ts.FIRST_ITER
    assert complement is not ts.LAST_ITER
    assert ~complement is ts.FIRST_ITER
    assert [complement.fires(i, 3) for i in range(3)] == [False, True, True]
    assert complement.fires(0, 1) is False


def test_operators_flatten_and_reject_non_guards():
    a, b, c = (ts.OpaqueCondition(key=name) for name in "abc")
    assert (a & b & c).operands == (a, b, c)
    assert ((a | b) | (b | c)).operands == (a, b, b, c)
    with pytest.raises(TypeError):
        a & True
    with pytest.raises(ValueError, match="at least two operands"):
        ts.GuardAnd((a,))


def test_domain_predicate_properties_and_printing(resource):
    def body(data):
        with ts.domain_loop(4) as d:
            ready = data.bit_a()
            with ts.when_true(d.first | d.every_pred(2) | ~ready):
                data.record(0)
            with ts.when_true(~d.last):
                data.record(1)

    text = str(_capture(body, resource))
    assert (
        "when_true((FirstIter or Every(period=2, start=0) or "
        "~(key=('data', 'bit_a', 0))))"
    ) in text
    assert "condition=~LastIter" in text


def test_ir_freezes_composite_guards(resource):
    def body(data):
        with ts.domain_loop(4) as d:
            ready = data.bit_a()
            with ts.when_true(~d.last & (ready | d.every_pred(3, start=1))):
                data.record(0)

    program = ts.TaskManager([ts.Task(0, 1, schedule=_capture(body, resource))], {})
    loop = program.freeze().tasks[0].body[0]
    guard = loop.body[1].guard
    assert guard.kind == "and"
    last, disjunction = guard.operands
    assert (last.kind, last.negated) == ("last", True)
    assert disjunction.kind == "or"
    assert [item.kind for item in disjunction.operands] == ["opaque", "every"]
    assert disjunction.operands[0].value_id == 0
    assert (disjunction.operands[1].period, disjunction.operands[1].start) == (3, 1)


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------


def test_guard_fires_boolean_truth_tables():
    a, b = ts.OpaqueCondition(key="a"), ts.OpaqueCondition(key="b")
    values = list(itertools.product((False, True), repeat=2))

    def table(guard):
        return [
            guard_fires(guard, 0, 1, opaque_assignment={"a": x, "b": y})
            for x, y in values
        ]

    assert table(a & b) == [False, False, False, True]
    assert table(a | b) == [False, True, True, True]
    assert table(~(a & b)) == [True, True, True, False]
    assert table((a & ~b) | (~a & b)) == [False, True, True, False]


def test_guard_fires_mixed_and_complemented_iteration_predicates():
    guard = ts.FIRST_ITER & ts.OpaqueCondition(key="flag")
    assert guard_fires(guard, 0, 3, opaque_assignment={"flag": True})
    assert not guard_fires(guard, 1, 3, opaque_assignment={"flag": True})
    assert not guard_fires(guard, 0, 3, opaque_assignment={"flag": False})
    assert [guard_fires(~ts.FIRST_ITER, i, 3) for i in range(3)] == [
        False,
        True,
        True,
    ]
    assert [guard_fires(~ts.LAST_ITER, i, 3) for i in range(3)] == [
        True,
        True,
        False,
    ]
    assert [guard_fires(~ts.Every(2), i, 4) for i in range(4)] == [
        False,
        True,
        False,
        True,
    ]
    assert guard_fires(~ts.FIRST_ITER, 0, 1) is False


def test_composite_guard_collects_leaf_keys():
    guard = ts.OpaqueCondition(key="a") & ts.OpaqueCondition(key="b")
    schedule = ts.Schedule(
        name="composite",
        resources=[],
        body=[ts.ConditionalBlock(body=[], condition=guard)],
    )
    assert collect_opaque_keys([SimpleNamespace(schedule=schedule)]) == ["a", "b"]


def test_and_only_assignments_collapse_to_two_classes():
    guard = ts.OpaqueCondition(key="a") & ts.OpaqueCondition(key="b")
    assignments = _all_opaque_assignments(["a", "b"], [(guard, 0, 1)])
    assert len(assignments) == 2
    assert {
        guard_fires(guard, 0, 1, opaque_assignment=item) for item in assignments
    } == {False, True}


def test_or_only_assignments_collapse_any_true_class():
    guard = ts.OpaqueCondition(key="a") | ts.OpaqueCondition(key="b")
    assignments = _all_opaque_assignments(["a", "b"], [(guard, 0, 1)])
    assert assignments == [{"a": False, "b": False}, {"a": False, "b": True}]


def test_independently_observed_leaf_keeps_third_class():
    a = ts.OpaqueCondition(key="a")
    guard = a & ts.OpaqueCondition(key="b")
    assert len(_all_opaque_assignments(["a", "b"], [(guard, 0, 1), (a, 0, 1)])) == 3


def test_iteration_leaves_are_observed_at_each_domain_iteration(resource):
    def body(data):
        with ts.domain_loop(3):
            first, second = data.probe()
            with ts.when_true((first & ts.LAST_ITER) | second):
                data.record(0)

    task = ts.Task(0, 1, schedule=_capture(body, resource))
    # Iterations 0 and 1 agree on the LAST_ITER leaf, so one is observed.
    observations = ts.collect_guard_observations([task])
    assert [(index, count) for _, index, count in observations] == [(0, 3), (2, 3)]
    keys = collect_opaque_keys([task])
    first_key, second_key = keys
    # ``second`` alone decides early rows and ``first or second`` the last row,
    # so (True, True) behaves like (False, True) and is not checked again.
    assert ts.opaque_assignment_sequences([task], keys) == [
        {first_key: False, second_key: False},
        {first_key: False, second_key: True},
        {first_key: True, second_key: False},
    ]


def _pipe(name="pipe"):
    return Resource(
        name=name,
        pipeline_config=ts.PipelineConfig.create_async_async_pipeline_cfg(
            1, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
        ),
    )


def _pipeline_tasks(consumer_guard):
    pipe = _pipe()

    @ts.schedule
    def producer(pipe):
        with ts.domain_loop(3):
            flag = pipe.bit_a()
            with ts.when_true(flag, key="f"):
                pipe.acquire()
                pipe.producer_work()
                pipe.commit()

    @ts.schedule
    def consumer(pipe):
        with ts.domain_loop(3) as d:
            flag = pipe.bit_a()
            with consumer_guard(d, flag):
                pipe.wait()
                pipe.consumer_work()
                pipe.release()

    return [
        ts.Task(1, 1, schedule=producer(pipe), name="Producer"),
        ts.Task(0, 1, schedule=consumer(pipe), name="Consumer"),
    ]


@pytest.mark.parametrize(
    "consumer_guard, safe",
    [
        pytest.param(lambda d, f: ts.when_true(f, key="f"), True, id="matching"),
        pytest.param(
            lambda d, f: ts.when_true(f & (d.first | ~d.first), key="f"),
            True,
            id="tautology",
        ),
        pytest.param(
            lambda d, f: ts.when_true(f & d.first, key="f"), False, id="first-only"
        ),
        pytest.param(
            lambda d, f: ts.when_false(~f | ~d.first, key="f"),
            False,
            id="first-only-nnf",
        ),
    ],
)
def test_checker_evaluates_composite_guards(consumer_guard, safe):
    tasks = _pipeline_tasks(consumer_guard)
    keys = collect_opaque_keys(tasks)
    assert keys == ["f"]
    results = {
        assignment["f"]: ts.check_all_interleavings(tasks, opaque_assignment=assignment)
        for assignment in ts.opaque_assignment_sequences(tasks, keys)
    }
    assert results[False].is_safe
    assert results[True].is_safe == safe
    if safe:
        ts.TaskManager(tasks, {})
    else:
        assert results[True].deadlock_states
        with pytest.raises(ValueError, match="opaque assignment f=True"):
            ts.TaskManager(tasks, {})


# -----------------------------------------------------------------------------
# Device lowering
# -----------------------------------------------------------------------------


COLUMNS = 10


@ts.schedule
def _device_schedule(stage_info, data):
    with ts.domain_loop(0, stage_info.context.tasks_inputs.count, 1) as d:
        a = data.bit_a()
        b = data.bit_b()
        with ts.when_true(a & b):
            data.record(0)
        with ts.when_true(a | b):
            data.record(1)
        with ts.when_true((a & ~b) | (~a & b)):
            data.record(2)
        with ts.when_true(d.first & a):
            data.record(3)
        with ts.when_true(~d.last & b):
            data.record(4)
        with ts.when_false(a & b):
            data.record(5)
        with ts.when_false(d.first | b):
            data.record(6)
        with ts.when_true(d.every_pred(3, start=1) | a):
            data.record(7)
        with ts.when_true(a):
            with ts.when_true(~b | d.last):
                data.record(8)
        with ts.when_true(ts.FIRST_ITER | ts.LAST_ITER):
            data.record(9)


def _expected(count, mask_a, mask_b):
    expected = torch.zeros((8, COLUMNS), dtype=torch.int32)
    for i in range(count):
        a, b = bool(mask_a >> i & 1), bool(mask_b >> i & 1)
        first, last = i == 0, i == count - 1
        every = i >= 1 and (i - 1) % 3 == 0
        row = [
            a and b,
            a or b,
            a != b,
            first and a,
            not last and b,
            not (a and b),
            not (first or b),
            every or a,
            a and (not b or last),
            first or last,
        ]
        expected[i] = torch.tensor(row, dtype=torch.int32)
    return expected


@requires_hopper
@pytest.mark.parametrize(
    "count, mask_a, mask_b",
    [(1, 0b1, 0b0), (1, 0b1, 0b1), (8, 0b10110101, 0b01100110), (5, 0b11111, 0b0)],
)
def test_device_composite_guards_match_host_truth_table(
    count, mask_a, mask_b, resource
):
    device = ts.Task(0, 1, schedule=_device_schedule(resource)).to_device()

    @cl.kernel
    def kernel(output, count, mask_a, mask_b):
        device(device.make_context(Inputs(output, count, mask_a, mask_b)))

    output = torch.zeros((8, COLUMNS), dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (32,),
        kernel,
        (output, count, mask_a, mask_b),
    )
    torch.testing.assert_close(
        output.cpu(), _expected(count, mask_a, mask_b), rtol=0, atol=0
    )
