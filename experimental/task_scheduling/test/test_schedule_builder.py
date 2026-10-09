# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

from task_scheduling_test_requirements import cuda_lang as cl
from task_scheduling_test_requirements import task_scheduling as ts

from cuda.lang.compilation import KernelSignature
from task_scheduling_test_utils import make_symbolic_tensor


@dataclass(kw_only=True, eq=False)
class RoutedResource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    def produce(self, stage_info):
        return stage_info.loop_offset

    @ts.producer_work
    def consume(self, stage_info, value):
        del stage_info, value


@dataclass(kw_only=True, eq=False)
class UnifiedStaticResource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    @staticmethod
    def produce(stage_info):
        return stage_info.loop_offset

    @ts.producer_work
    @staticmethod
    def consume(stage_info, value):
        return stage_info.loop_offset, value


@dataclass(kw_only=True, eq=False)
class ManyConstexprArgsResource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    @staticmethod
    def produce(stage_info):
        return cl.int32(1)

    @ts.producer_work
    @staticmethod
    def consume(stage_info, a, value, b, c, d):
        stage_info.context.tasks_inputs[0] = value + a + b + c + d


@dataclass(kw_only=True, eq=False)
class ManyRoutedValuesResource(ts.MemoryResource):
    @ts.consumer_work(outputs=4)
    @staticmethod
    def produce(stage_info):
        return cl.int32(1), cl.float32(2.0), cl.bool_(True), cl.uint64(4)

    @ts.producer_work
    @staticmethod
    def consume(stage_info, integer, floating, predicate, wide):
        stage_info.context.tasks_inputs[0] = integer


@dataclass(kw_only=True, eq=False)
class AnonymousRoutedResource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    @staticmethod
    def produce(stage_info):
        stage_info.context
        return cl.int32(3)

    @ts.producer_work
    @staticmethod
    def consume(stage_info, outer, inner):
        stage_info.context.tasks_inputs[0] = outer + inner

    @ts.consumer_work(outputs=2)
    @staticmethod
    def produce_pair(stage_info):
        stage_info.context
        return cl.int32(2), cl.int32(5)

    @ts.producer_work
    @staticmethod
    def consume_one(stage_info, value, output_index):
        stage_info.context.tasks_inputs[output_index] = value


@dataclass(kw_only=True, eq=False)
class LoopCarriedResource(ts.MemoryResource):
    @ts.consumer_work(outputs=1)
    @staticmethod
    def initialize(stage_info):
        stage_info.context
        return cl.int32(0)

    @ts.consumer_work(outputs=1)
    @staticmethod
    def advance(stage_info, state):
        stage_info.context
        return state + 1

    @ts.consumer_work(outputs=1)
    @staticmethod
    def advance_by_offset(stage_info, state):
        stage_info.context
        return state + stage_info.loop_offset

    @ts.producer_work
    @staticmethod
    def consume(stage_info, state):
        stage_info.context.tasks_inputs[0] = state

    @ts.producer_work
    @staticmethod
    def record_iteration(stage_info, slot):
        if cl.thread_index(0) == 0:
            stage_info.context.tasks_inputs[slot] = stage_info.loop_offset


class WorkTileDomainTask(ts.Task):
    def __init__(self, *args, domain_end=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_end = domain_end

    def _freeze_domain_task(self):
        return self.domain_end

    def get_domain(self, tile_coordinate):
        return self - tile_coordinate[0]


def test_nested_schedule_preorder_and_format():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        value = data.produce()
        with ts.domain_loop(1, 9, 2) as domain:
            with domain.first_iter():
                data.consume(value)
            with domain.every(2, start=1):
                data.consume(value)

    tree = captured(resource)
    seen = []
    tree.visit(lambda node: seen.append(type(node).__name__))
    assert seen == [
        "Schedule",
        "Step",
        "DomainLoop",
        "ConditionalBlock",
        "Step",
        "ConditionalBlock",
        "Step",
    ]
    assert "Every(period=2, start=1)" in str(tree)
    assert "value <- %0" in str(tree)
    assert isinstance(tree.body, tuple)


def test_capture_cleanup_after_exception():
    resource = RoutedResource(name="data")

    @ts.schedule
    def bad(data):
        data.produce()
        raise RuntimeError("trace failed")

    with pytest.raises(RuntimeError, match="trace failed"):
        bad(resource)
    with pytest.raises(ts.ScheduleError, match="inside an @schedule"):
        ts.domain_loop(4).__enter__()


@pytest.mark.parametrize(
    "bounds, expected",
    [((7,), (0, 7, 1)), ((2, 7), (2, 7, 1)), ((2, 9, 3), (2, 9, 3))],
)
def test_domain_loop_range_forms(bounds, expected):
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        with ts.domain_loop(*bounds):
            data.produce()

    loop = captured(resource).body[0]
    assert (loop.start, loop.end, loop.step) == expected


def test_dynamic_domain_bound_is_preserved_for_device_resolution():
    resource = RoutedResource(name="data")

    def runtime_end(values):
        return values.end

    bound = ts.dynamic_domain_bound(runtime_end)

    @ts.schedule
    def captured(data):
        with ts.domain_loop(bound):
            data.produce()

    tree = captured(resource)
    task = ts.Task(
        [resource], [], 0, 1, schedule=tree, name="dynamic_consumer"
    )
    device_task = task.to_device({"produce": lambda stage_info: stage_info.loop_offset})
    device_loop = device_task.body[0]

    assert isinstance(bound, ts.DynamicDomainBound)
    assert str(bound) == "<dynamic Int32>"
    assert "end=<dynamic Int32>" in str(tree)
    assert device_loop.dynamic_end
    assert device_loop.end == 0
    assert device_loop.end_resolver is runtime_end
    assert device_loop.num_iterations == -1


def test_task_domain_bound_receives_current_work_tile():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        with ts.domain_loop(WorkTileDomainTask.get_domain):
            data.produce()

    task = WorkTileDomainTask(
        [resource], [], 0, 1, schedule=captured(resource), name="work_tile_domain"
    )
    device_loop = task.to_device(
        {"produce": lambda stage_info: stage_info.loop_offset}
    ).body[0]

    assert device_loop.dynamic_end
    assert device_loop.end_uses_task
    assert device_loop.end_resolver is WorkTileDomainTask.get_domain
    assert device_loop.end_resolver(device_loop.resolver_task, (2, 0, 0)) == 5


def test_schedule_stage_info_resolves_named_kernel_input_bound():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(stage_info, data):
        with ts.domain_loop(stage_info.context.tasks_inputs.end):
            data.produce()

    tree = captured(resource)
    bound = tree.body[0].end
    task = ts.Task([resource], [], 0, 1, schedule=tree, name="dynamic_consumer")
    device_loop = task.to_device(
        {"produce": lambda stage_info: stage_info.loop_offset}
    ).body[0]

    @dataclass(frozen=True)
    class KernelInputs:
        end: object

    assert isinstance(bound, ts.DynamicDomainBound)
    assert bound.resolve(KernelInputs(7)) == 7
    assert str(bound) == "<dynamic Int32>"
    assert "end=<dynamic Int32>" in str(tree)
    assert device_loop.dynamic_end
    assert device_loop.end_resolver is bound.resolver


def test_schedule_stage_info_must_be_first():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data, stage_info):
        data.produce()

    with pytest.raises(ts.ScheduleError, match="must be the first"):
        captured(resource)


def test_invalid_loop_and_guard_grammar():
    resource = RoutedResource(name="data")

    @ts.schedule
    def zero_step(data):
        with ts.domain_loop(0, 4, 0):
            data.produce()

    with pytest.raises(ts.ScheduleError, match="non-zero"):
        zero_step(resource)

    @ts.schedule
    def nested(data):
        with ts.domain_loop(4):
            with ts.domain_loop(2):
                data.produce()

    with pytest.raises(ts.ScheduleError, match="cannot be nested"):
        nested(resource)


def test_opaque_true_false_correlation():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        condition = data.produce()
        with ts.when_true(condition, key="highlight"):
            data.consume(condition)
        with ts.when_false(condition, key="highlight"):
            data.consume(condition)

    tree = captured(resource)
    guards = [tree.body[1].condition, tree.body[2].condition]
    assert [guard.key for guard in guards] == ["highlight", "highlight"]
    assert [guard.negated for guard in guards] == [False, True]


def test_work_tile_queue_advance_placement_and_skippable():
    queue = ts.WorkQueue(name="queue")
    data = RoutedResource(name="data")

    @ts.schedule
    def captured(wq, resource):
        with ts.work_tile_loop(wq, skip_if=lambda _queue, tile: tile < 0) as loop:
            with loop.skippable():
                with ts.domain_loop(4):
                    resource.produce()
            wq.get_and_advance_work_tile()

    assert isinstance(captured(queue, data).body[0], ts.WorkTileLoop)

    @ts.schedule
    def misplaced(wq):
        wq.get_and_advance_work_tile()

    with pytest.raises(ts.ScheduleError, match="direct child"):
        misplaced(queue)


def test_every_validation():
    assert ts.Every(4, 1).fires(5, 20)
    assert not ts.Every(4, 1).fires(4, 20)
    with pytest.raises(ValueError):
        ts.Every(0)
    with pytest.raises(TypeError):
        ts.Every(True)


def test_work_arguments_infer_static_or_dataflow_bindings():
    @dataclass(kw_only=True, eq=False)
    class StaticResource(ts.MemoryResource):
        @ts.consumer_work(outputs=1)
        def load(self, stage_info, subtile):
            del stage_info
            return subtile

    resource = StaticResource(name="static")

    @ts.schedule
    def captured(item):
        item.load(subtile=3)

    step = captured(resource).body[0]
    assert step.input_values == {}
    assert step.constexpr_kwargs == {"subtile": 3}

    @ts.schedule
    def routed(item):
        token = item.load(subtile=1)
        item.load(subtile=token)

    routed_tree = routed(resource)
    produced_value = routed_tree.body[0].output_values[0]
    routed_step = routed_tree.body[1]
    assert routed_step.input_values == {"subtile": produced_value}
    assert routed_step.constexpr_kwargs == {}


def test_device_work_supports_arbitrary_constexpr_argument_count():
    resource = ManyConstexprArgsResource(name="many_constexpr_args")

    @ts.schedule
    def captured(data):
        value = data.produce()
        data.consume(1, value, b=2, c=3, d=4)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )
    device_task = task.to_device()
    assert device_task.body[1].constexpr_args == (1, 2, 3, 4)
    assert device_task.body[1].argument_order == (1, 0, 2, 3, 4)

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature((make_symbolic_tensor((1,), cl.int32),))],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert "st.global" in compiled.ptx


def test_device_work_supports_arbitrary_routed_value_count():
    resource = ManyRoutedValuesResource(name="many_routed_values")

    @ts.schedule
    def captured(data):
        integer, floating, predicate, wide = data.produce()
        data.consume(integer, floating, predicate, wide)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )
    device_task = task.to_device()

    producer_step, consumer_step = device_task.body
    assert producer_step.append_output_count == 4
    assert consumer_step.input_slots == (0, 1, 2, 3)

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature((make_symbolic_tensor((1,), cl.int32),))],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert "st.global" in compiled.ptx


def test_anonymous_routes_append_and_pop_at_domain_scope():
    resource = AnonymousRoutedResource(name="anonymous")

    @ts.schedule
    def captured(data):
        outer = data.produce()
        with ts.domain_loop(2):
            inner = data.produce()
            data.consume(outer, inner)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )
    device_task = task.to_device()
    outer_step, domain = device_task.body
    inner_step, consumer_step = domain.body

    assert outer_step.append_output_count == 1
    assert outer_step.release_after_append == 0
    assert inner_step.append_output_count == 1
    assert inner_step.release_after_append == 0
    assert consumer_step.input_slots == (0, 1)
    assert consumer_step.release_before_append == 1
    assert domain.release_count == 1

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature((make_symbolic_tensor((1,), cl.int32),))],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert "st.global" in compiled.ptx


def test_anonymous_route_cannot_escape_domain_scope():
    resource = AnonymousRoutedResource(name="anonymous")

    @ts.schedule
    def captured(data):
        with ts.domain_loop(1):
            inner = data.produce()
        data.consume(inner, inner)

    with pytest.raises(ts.ScheduleError, match="control-flow scope"):
        captured(resource)


@pytest.mark.parametrize("num_iterations", [0, 3])
def test_domain_loop_functionally_carries_anonymous_routes(num_iterations):
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body(state):
            state = data.advance(state)
            state = data.advance(state)
            return state

        state = ts.domain_loop(0, num_iterations, 1, loop_body, state)
        data.consume(state)

    tree = captured(resource)
    initialize, loop, consume = tree.body
    first_advance, second_advance = loop.body

    assert loop.initial_values["state"] is initialize.output_values[0]
    assert first_advance.input_values["state"] is loop.iter_values["state"]
    assert (
        second_advance.input_values["state"]
        is first_advance.output_values[0]
    )
    assert loop.yield_values["state"] is second_advance.output_values[0]
    assert consume.input_values["state"] is loop.result_values["state"]
    assert loop.iter_values["state"].scope != loop.result_values["state"].scope

    task = ts.Task(0, 1, schedule=tree)
    device_task = task.to_device()
    device_loop = device_task.body[1]
    device_consume = device_task.body[2]
    assert device_loop.initial_route_positions == (0,)
    assert device_loop.num_iterations == num_iterations
    assert device_loop.yield_route_positions == (2,)
    assert device_loop.result_indices == (0,)
    assert device_loop.body[0].input_slots == (1,)
    assert device_loop.body[1].input_slots == (2,)
    assert device_consume.input_slots == (1,)

    manager = ts.TaskManager(
        [task],
        {},
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )
    loop_ir = manager.freeze().tasks[0].body[1]
    assert isinstance(loop_ir, ts.DomainLoopIR)
    assert loop_ir.carried_names == ("state",)
    assert len(
        {
            *loop_ir.initial_values,
            *loop_ir.iter_values,
            *loop_ir.yield_values,
            *loop_ir.output_values,
        }
    ) == 4


def test_functional_domain_loop_captures_body_steps_and_results():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        left = data.initialize()
        middle = data.initialize()
        right = data.initialize()

        def loop_body(left, middle, right):
            left = data.advance(left)
            middle = data.advance(middle)
            right = data.advance_by_offset(right)
            return left, middle, right

        left, middle, right = ts.domain_loop(
            1, 4, 1, loop_body, left, middle, right
        )
        data.consume(left)
        data.consume(middle)
        data.consume(right)

    tree = captured(resource)
    loop = tree.body[3]
    assert isinstance(loop, ts.DomainLoop)
    assert tuple(loop.initial_values) == ("left", "middle", "right")
    assert [step.label for step in loop.body] == [
        "advance",
        "advance",
        "advance_by_offset",
    ]
    assert tuple(loop.yield_values) == ("left", "middle", "right")
    assert tuple(loop.result_values) == ("left", "middle", "right")

    task = ts.Task(0, 1, schedule=tree)
    device_loop = task.to_device().body[3]
    assert len(device_loop.body) == 3

    manager = ts.TaskManager(
        [task],
        {},
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )
    loop_ir = manager.freeze().tasks[0].body[3]
    assert loop_ir.body[2].argument_order == ("state",)


def test_functional_domain_loop_validates_return_arity():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        first = data.initialize()
        second = data.initialize()

        def loop_body(first, second):
            return data.advance(first)

        ts.domain_loop(0, 2, 1, loop_body, first, second)

    with pytest.raises(ts.ScheduleError, match="expected 2, got 1"):
        captured(resource)


def test_functional_domain_loop_captures_iteration_guards():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body():
            with ts.first_iter():
                data.consume(state)
            with ts.every(2, start=1):
                data.consume(state)
            with ts.last_iter():
                data.consume(state)

        ts.domain_loop(0, 5, 1, loop_body)

    tree = captured(resource)
    loop = tree.body[1]
    assert isinstance(loop, ts.DomainLoop)
    assert [type(node).__name__ for node in loop.body] == [
        "ConditionalBlock",
        "ConditionalBlock",
        "ConditionalBlock",
    ]
    assert "FirstIter" in str(tree)
    assert "Every(period=2, start=1)" in str(tree)
    assert "LastIter" in str(tree)

    with pytest.raises(ts.ScheduleError, match="inside an @schedule"):
        ts.first_iter().__enter__()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="requires a Blackwell CC 10.0 GPU",
)
@pytest.mark.parametrize("unroll", [None, 2, 4])
def test_functional_domain_loop_carried_route_executes_on_device(unroll):
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body(state):
            state = data.advance(state)
            state = data.advance(state)
            return state

        state = ts.domain_loop(0, 3, 1, loop_body, state, unroll=unroll)
        data.consume(state)

    device_task = ts.Task(0, 1, schedule=captured(resource)).to_device()

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    output = torch.zeros((1,), device="cuda:0", dtype=torch.int32)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (32,),
        kernel,
        (output,),
    )
    torch.cuda.synchronize()
    assert output.item() == 6


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell GPU",
)
@pytest.mark.parametrize("unroll", [2, 3, 4, 6])
def test_unrolled_domain_loop_runtime_bounds_and_carried_state(unroll):
    resource = LoopCarriedResource(name="state")

    def runtime_start(values):
        return values[1]

    def runtime_end(values):
        return values[2]

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body(state):
            with ts.first_iter():
                data.record_iteration(3)
            with ts.last_iter():
                data.record_iteration(4)
            with ts.every(2, start=1):
                data.record_iteration(5)
            return data.advance_by_offset(state)

        state = ts.domain_loop(
            ts.dynamic_domain_bound(runtime_start),
            ts.dynamic_domain_bound(runtime_end),
            2,
            loop_body,
            state,
            unroll=unroll,
        )
        data.consume(state)

    device_task = ts.Task(0, 1, schedule=captured(resource)).to_device()
    assert device_task.body[1].unroll == unroll

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    # Reuse one specialization for empty, tail-only, exact-group, and odd-tail
    # loops, with a non-unit step and a non-zero dynamic start.
    for start, end in [(5, 1), (1, 1), (1, 2), (1, 9), (3, 20)]:
        output = torch.tensor(
            [-1, start, end, -1, -1, -1], device="cuda", dtype=torch.int32
        )
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (output,))
        torch.cuda.synchronize()
        offsets = list(range(start, end, 2))
        assert output[0].item() == sum(offsets)
        assert output[3:].tolist() == [
            offsets[0] if offsets else -1,
            offsets[-1] if offsets else -1,
            offsets[1::2][-1] if len(offsets) > 1 else -1,
        ]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="requires a Blackwell GPU",
)
def test_functional_domain_loop_injects_loop_offset_in_stage_info():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body(state):
            return data.advance_by_offset(state)

        state = ts.domain_loop(1, 4, 1, loop_body, state)
        data.consume(state)

    device_task = ts.Task(0, 1, schedule=captured(resource)).to_device()

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    output = torch.zeros((1,), device="cuda:0", dtype=torch.int32)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (32,),
        kernel,
        (output,),
    )
    torch.cuda.synchronize()
    assert output.item() == 6


def test_python_rebinding_does_not_implicitly_carry_value_out_of_loop():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()
        with ts.domain_loop(3):
            state = data.advance(state)
        data.consume(state)

    with pytest.raises(
        ts.ScheduleError, match="reassignment.*does not create loop-carried state"
    ):
        captured(resource)


def test_functional_domain_loop_carried_result_must_be_unconditional():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()

        def loop_body(state):
            with ts.first_iter():
                state = data.advance(state)
            return state

        ts.domain_loop(0, 3, 1, loop_body, state)

    with pytest.raises(ts.ScheduleError, match="control-flow scope"):
        captured(resource)


def test_domain_loop_rejects_removed_carried_keyword():
    resource = LoopCarriedResource(name="state")

    @ts.schedule
    def captured(data):
        state = data.initialize()
        with ts.domain_loop(3, carried={"state": state}):
            data.consume(state)

    with pytest.raises(TypeError, match="unexpected keyword argument 'carried'"):
        captured(resource)


def test_anonymous_routes_support_multiple_outputs_consumers_and_non_lifo_use():
    resource = AnonymousRoutedResource(name="anonymous")

    @ts.schedule
    def captured(data):
        first, second = data.produce_pair()
        data.consume_one(first, 0)
        data.consume_one(first, 1)
        data.consume_one(second, 2)

    tree = captured(resource)
    produced = tree.body[0].output_values
    assert len(produced) == 2
    assert all(isinstance(value, ts.ScheduleValue) for value in produced)
    assert produced[0].value_id != produced[1].value_id
    assert tree.body[1].input_values["value"] is produced[0]
    assert tree.body[3].input_values["value"] is produced[1]

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=tree,
    )
    device_task = task.to_device()
    producer, first_use, last_first_use, second_use = device_task.body

    assert producer.append_output_count == 2
    assert producer.release_after_append == 0
    assert first_use.input_slots == (0,)
    assert first_use.release_before_append == 0
    assert last_first_use.input_slots == (0,)
    assert last_first_use.release_before_append == 0
    assert second_use.input_slots == (1,)
    assert second_use.release_before_append == 2

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature((make_symbolic_tensor((3,), cl.int32),))],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert "st.global" in compiled.ptx


def test_anonymous_route_can_guard_a_conditional_and_is_released_afterward():
    resource = AnonymousRoutedResource(name="anonymous")

    @ts.schedule
    def captured(data):
        predicate = data.produce()
        with ts.when_true(predicate):
            data.consume(predicate, predicate)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )
    device_task = task.to_device()
    producer, conditional = device_task.body

    assert producer.release_after_append == 0
    assert conditional.guard.slot == 0
    assert conditional.release_count == 1

    @cl.kernel
    def kernel(output):
        device_task(device_task.make_context(output))

    compiled = cl.compile_simt(
        kernel,
        [KernelSignature((make_symbolic_tensor((1,), cl.int32),))],
        gpu_name="sm_100a",
        arch="compute_100a",
        keep_ptx=True,
    )
    assert "st.global" in compiled.ptx


def test_anonymous_route_cannot_escape_conditional_scope():
    resource = AnonymousRoutedResource(name="anonymous")

    @ts.schedule
    def captured(data):
        predicate = data.produce()
        with ts.when_true(predicate):
            inner = data.produce()
        data.consume(inner, predicate)

    with pytest.raises(ts.ScheduleError, match="control-flow scope"):
        captured(resource)


def test_work_decorators_reject_removed_named_returns():
    with pytest.raises(TypeError, match="unexpected keyword argument 'returns'"):
        ts.consumer_work(returns="value")


def test_device_routing_uses_anonymous_value_stack():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        value = data.produce()
        data.consume(value)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )

    def produce(context):
        return context.loop_offset

    def consume(context, value):
        del context, value

    device_task = task.to_device({"produce": produce, "consume": consume})
    producer_step, consumer_step = device_task.body

    assert producer_step.automatic_routing
    assert producer_step.input_slots == ()
    assert producer_step.append_output_count == 1
    assert consumer_step.input_slots == (0,)
    assert not producer_step.in_domain_loop


def test_static_work_methods_are_unified_host_and_device_callbacks():
    resource = UnifiedStaticResource(name="unified")

    assert resource.produce(ts.StageInfo(loop_offset=7)) == 7
    assert resource.consume(ts.StageInfo(loop_offset=9), 3) == (9, 3)

    @ts.schedule
    def captured(data):
        value = data.produce()
        data.consume(value)

    task = ts.Task(
        [resource],
        [resource],
        0,
        1,
        schedule=captured(resource),
    )
    device_task = task.to_device()

    assert device_task.body[0].callback is UnifiedStaticResource.produce.__wrapped__
    assert device_task.body[1].callback is UnifiedStaticResource.consume.__wrapped__


def test_work_methods_reject_classmethod_descriptors():
    with pytest.raises(TypeError, match="class methods"):
        class InvalidResource:
            @ts.consumer_work
            @classmethod
            def work(cls, stage_info):
                del cls, stage_info


def test_sink_step_inherits_pipeline_stage_from_routed_input():
    @dataclass(kw_only=True, eq=False)
    class SinkResource(ts.MemoryResource):
        @ts.producer_work
        def store(self, stage_info, value):
            del stage_info, value

    config = ts.PipelineConfig.create_async_async_pipeline_cfg(
        2, ts.CooperativeGroup(32), ts.CooperativeGroup(32)
    )
    source = RoutedResource(name="source", pipeline_config=config)
    sink = SinkResource(name="sink")

    @ts.schedule
    def captured(data, output):
        value = data.produce()
        output.store(value)

    task = ts.Task(
        [source],
        [sink],
        0,
        1,
        schedule=captured(source, sink),
    )
    binding = ts.DevicePipelineBinding.from_config(config)
    device_task = task.to_device(
        {"produce": lambda stage_info: 0, "store": lambda stage_info, value: None},
        pipeline_bindings={source: binding},
    )

    source_step, sink_step = device_task.body
    assert source_step.pipeline_slot == 0
    assert sink_step.pipeline_slot == source_step.pipeline_slot
    assert sink_step.pipeline_binding == source_step.pipeline_binding


def test_frozen_schedule_mappings_reject_mutation():
    resource = RoutedResource(name="data")

    @ts.schedule
    def captured(data):
        value = data.produce()
        data.consume(value)

    tree = captured(resource)
    with pytest.raises(TypeError):
        tree.body[1].input_values["value"] = tree.body[0].output_values[0]
