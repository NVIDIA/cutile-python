# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical immutable snapshot of a validated task-scheduling program."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, eq=False)
class ScheduleValue:
    """One independently routed call-site SSA value.

    ``outputs=N`` creates ``N`` instances.  A work call returns the single
    instance directly or a tuple of instances, so ordinary Python unpacking
    works without exposing the device route stack.

    Values are routed by liveness through the device value stack.
    """

    value_id: int
    producer_step: object
    stage_resource: object | None = None
    scope: tuple[int, ...] = ()

    def __iter__(self):
        raise TypeError(
            "a single schedule value cannot be unpacked; declare outputs=N "
            "to return N independently routed values"
        )

    def __getitem__(self, index: object):
        del index
        raise TypeError("a single schedule value cannot be indexed")


@dataclass(frozen=True)
class ResourceIR:
    resource_id: int
    name: str
    type_name: str
    pipeline_config: object | None
    is_barrier: bool
    ranges: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class GuardIR:
    kind: str
    key: object | None = None
    value_id: int | None = None
    negated: bool = False
    period: int = 1
    start: int = 0


@dataclass(frozen=True)
class StepIR:
    resource_id: int
    schedule_stage: object
    argument_order: tuple[str, ...]
    input_values: Mapping[str, int]
    static_values: Mapping[str, object]
    output_values: tuple[int, ...]
    label: str | None
    unique_id: int
    stage_resource_id: int | None = None


@dataclass(frozen=True)
class ConditionalIR:
    body: tuple["NodeIR", ...]
    guard: GuardIR


@dataclass(frozen=True)
class DomainLoopIR:
    start: object
    end: object
    step: object
    unroll: int | None
    body: tuple["NodeIR", ...]
    carried_names: tuple[str, ...] = ()
    initial_values: tuple[int, ...] = ()
    iter_values: tuple[int, ...] = ()
    yield_values: tuple[int, ...] = ()
    output_values: tuple[int, ...] = ()


@dataclass(frozen=True)
class WorkTileLoopIR:
    body: tuple["NodeIR", ...]
    work_queue_id: int
    skip_if: object | None = None


NodeIR = StepIR | ConditionalIR | DomainLoopIR | WorkTileLoopIR


@dataclass(frozen=True)
class TaskIR:
    task_id: int
    name: str
    body: tuple[NodeIR, ...]
    resource_ids: tuple[int, ...]
    consumer_resource_ids: tuple[int, ...]
    producer_resource_ids: tuple[int, ...]
    warp_start: int
    warp_end: int
    num_registers: int
    run_only_on_cta_id: int | None
    live_value_ids: frozenset[int]


@dataclass(frozen=True)
class DependencyEdgeIR:
    upstream_id: int
    downstream_id: int


@dataclass(frozen=True)
class ProgramIR:
    resources: tuple[ResourceIR, ...]
    tasks: tuple[TaskIR, ...]
    dependency_edges: tuple[DependencyEdgeIR, ...]
    cta_warps: int

    @property
    def resource_by_id(self) -> Mapping[int, ResourceIR]:
        return MappingProxyType(
            {resource.resource_id: resource for resource in self.resources}
        )


def _iter_ir_nodes(nodes: tuple[NodeIR, ...]):
    for node in nodes:
        yield node
        if not isinstance(node, StepIR):
            yield from _iter_ir_nodes(node.body)


def iter_ir_steps(task: TaskIR):
    return (node for node in _iter_ir_nodes(task.body) if isinstance(node, StepIR))


def freeze_program_ir(
    tasks,
    resources,
    dependency_graph,
    *,
    cta_warps: int,
    default_num_registers: int,
) -> ProgramIR:
    """Normalize capture objects into an identity-free, frozen program view."""

    from .enums import Every, FIRST_ITER, LAST_ITER, OpaqueCondition, SKIPPABLE
    from .schedule_builder import ConditionalBlock, DomainLoop, Step, WorkTileLoop

    resource_ids = {resource: index for index, resource in enumerate(resources)}
    resource_irs = tuple(
        ResourceIR(
            resource_id=resource_ids[resource],
            name=resource.name,
            type_name=type(resource).__name__,
            pipeline_config=resource.pipeline_config,
            is_barrier=resource.is_barrier,
            ranges=tuple(resource.physical_ranges()),
        )
        for resource in resources
    )

    def freeze_guard(guard) -> GuardIR:
        if guard is FIRST_ITER:
            return GuardIR("first")
        if guard is LAST_ITER:
            return GuardIR("last")
        if guard is SKIPPABLE:
            return GuardIR("skippable")
        if isinstance(guard, Every):
            return GuardIR("every", period=guard.period, start=guard.start)
        if isinstance(guard, OpaqueCondition):
            return GuardIR(
                "opaque",
                key=guard.key,
                value_id=(
                    guard.route_token.value_id
                    if isinstance(guard.route_token, ScheduleValue)
                    else None
                ),
                negated=guard.negated,
            )
        raise TypeError(f"unsupported captured guard {type(guard).__name__}")

    def freeze_node(node) -> NodeIR:
        if isinstance(node, Step):
            return StepIR(
                resource_id=resource_ids[node.memory_resource],
                schedule_stage=node.schedule_stage,
                argument_order=node.argument_order,
                input_values=MappingProxyType(
                    {name: value.value_id for name, value in node.input_values.items()}
                ),
                static_values=MappingProxyType(dict(node.constexpr_kwargs)),
                output_values=tuple(value.value_id for value in node.output_values),
                label=node.label,
                unique_id=node.unique_id,
                stage_resource_id=(
                    resource_ids[node.stage_resource]
                    if node.stage_resource is not None
                    else None
                ),
            )
        if isinstance(node, ConditionalBlock):
            return ConditionalIR(
                tuple(freeze_node(child) for child in node.body),
                freeze_guard(node.condition),
            )
        if isinstance(node, DomainLoop):
            return DomainLoopIR(
                start=node.start,
                end=node.end,
                step=node.step,
                unroll=node.unroll,
                body=tuple(freeze_node(child) for child in node.body),
                carried_names=tuple(node.initial_values),
                initial_values=tuple(
                    value.value_id for value in node.initial_values.values()
                ),
                iter_values=tuple(
                    value.value_id for value in node.iter_values.values()
                ),
                yield_values=tuple(
                    value.value_id for value in node.yield_values.values()
                ),
                output_values=tuple(
                    value.value_id for value in node.result_values.values()
                ),
            )
        if isinstance(node, WorkTileLoop):
            return WorkTileLoopIR(
                tuple(freeze_node(child) for child in node.body),
                resource_ids[node.work_queue],
                node.skip_if,
            )
        raise TypeError(type(node).__name__)

    task_irs = []
    for task_id, task in enumerate(tasks):
        body = tuple(freeze_node(node) for node in task.schedule.body)
        consumed = {
            item
            for node in _iter_ir_nodes(body)
            if isinstance(node, StepIR)
            for item in node.input_values.values()
        }
        consumed.update(
            node.guard.value_id
            for node in _iter_ir_nodes(body)
            if isinstance(node, ConditionalIR) and node.guard.value_id is not None
        )
        for node in _iter_ir_nodes(body):
            if isinstance(node, DomainLoopIR):
                consumed.update(node.initial_values)
                consumed.update(node.yield_values)
        task_irs.append(
            TaskIR(
                task_id=task_id,
                name=task.name,
                body=body,
                resource_ids=tuple(resource_ids[item] for item in task.resources),
                consumer_resource_ids=tuple(
                    resource_ids[item] for item in task.consumer_resources
                ),
                producer_resource_ids=tuple(
                    resource_ids[item] for item in task.producer_resources
                ),
                warp_start=task.warp_start,
                warp_end=task.warp_end,
                num_registers=task.num_registers or default_num_registers,
                run_only_on_cta_id=task.run_only_on_cta_id,
                live_value_ids=frozenset(consumed),
            )
        )

    edges = []
    seen_edges = set()
    for downstream, upstreams in dependency_graph.items():
        for upstream in upstreams:
            edge = (resource_ids[upstream], resource_ids[downstream])
            if edge not in seen_edges:
                seen_edges.add(edge)
                edges.append(DependencyEdgeIR(*edge))
    return ProgramIR(resource_irs, tuple(task_irs), tuple(edges), cta_warps)
