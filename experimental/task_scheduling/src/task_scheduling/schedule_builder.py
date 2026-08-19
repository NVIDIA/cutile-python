# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture an immutable task schedule without creating compiler IR values."""

import functools
import inspect
import itertools
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Hashable, Mapping

from .enums import (
    BlockGuard,
    Every,
    FIRST_ITER,
    IterationPredicate,
    LAST_ITER,
    OpaqueCondition,
    SKIPPABLE,
    ScheduleStage,
)
from .ir import ScheduleValue
from .resources import (
    MemoryResource,
    WorkQueue,
    _collect_named_work_fns,
    _CONSUMER_WORK_FNS_ATTR,
    _PRODUCER_WORK_FNS_ATTR,
)


@dataclass(kw_only=True, eq=False, frozen=True)
class Step:
    memory_resource: MemoryResource
    schedule_stage: ScheduleStage
    argument_order: tuple[str, ...] = ()
    input_values: Mapping[str, ScheduleValue] = field(default_factory=dict)
    constexpr_kwargs: Mapping[str, object] = field(default_factory=dict)
    output_values: tuple[ScheduleValue, ...] = ()
    stage_resource: MemoryResource | None = None
    label: str | None = None
    unique_id: int


@dataclass(kw_only=True, eq=False, frozen=True)
class ConditionalBlock:
    body: tuple["Node", ...] | list["Node"]
    condition: BlockGuard


@dataclass(kw_only=True, eq=False, frozen=True)
class DomainLoop:
    start: object | Callable[..., object]
    end: object | Callable[..., object]
    step: object | Callable[..., object]
    unroll: int | None
    body: tuple["Node", ...] | list["Node"]
    initial_values: Mapping[str, ScheduleValue] = field(default_factory=dict)
    iter_values: Mapping[str, ScheduleValue] = field(default_factory=dict)
    yield_values: Mapping[str, ScheduleValue] = field(default_factory=dict)
    result_values: Mapping[str, ScheduleValue] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicDomainBound:
    """A schedule bound resolved from manager-owned values on the device.

    ``resolver`` receives ``ExecutionContext.tasks_inputs`` and returns the
    runtime scalar used by ``range``. The object remains host-visible in the
    captured schedule so analysis can use its representative-domain fallback
    without pretending that the device trip count is static.
    """

    resolver: Callable[[object], object]
    type_name: str = "Int32"

    def __post_init__(self) -> None:
        if not callable(self.resolver):
            raise TypeError("dynamic domain resolver must be callable")
        if not isinstance(self.type_name, str) or not self.type_name:
            raise ValueError("dynamic domain type_name must be a non-empty string")

    def resolve(self, tasks_inputs: object) -> object:
        return self.resolver(tasks_inputs)

    def __str__(self) -> str:
        return f"<dynamic {self.type_name}>"


@dataclass(frozen=True)
class _TasksInputsFieldResolver:
    name: str

    def __call__(self, tasks_inputs: object) -> object:
        return getattr(tasks_inputs, self.name)


class _TasksInputsProxy:
    def __getattr__(self, name: str) -> DynamicDomainBound:
        if name.startswith("_"):
            raise AttributeError(name)
        return DynamicDomainBound(_TasksInputsFieldResolver(name))


@dataclass(frozen=True)
class _ScheduleContextProxy:
    tasks_inputs: object = field(default_factory=_TasksInputsProxy)


@dataclass(frozen=True)
class ScheduleStageInfo:
    """Host proxy injected into the first ``stage_info`` schedule parameter."""

    context: object = field(default_factory=_ScheduleContextProxy)


@dataclass(kw_only=True, eq=False, frozen=True)
class WorkTileLoop:
    body: tuple["Node", ...] | list["Node"]
    work_queue: WorkQueue
    skip_if: Callable[..., object] | None = None


Node = Step | ConditionalBlock | DomainLoop | WorkTileLoop


@dataclass(kw_only=True, frozen=True)
class Schedule:
    name: str
    body: tuple[Node, ...] | list[Node]
    resources: tuple[MemoryResource, ...] | list[MemoryResource]

    def __str__(self) -> str:
        return _format_schedule(self)

    def visit(self, visitor: Callable[[object], None]) -> None:
        """Visit actual schedule nodes in deterministic preorder."""
        visitor(self)
        for node in self.body:
            _visit_node(node, visitor)


def _visit_node(node: Node, visitor: Callable[[object], None]) -> None:
    visitor(node)
    for child in _child_nodes(node):
        _visit_node(child, visitor)


def _child_nodes(node: Node) -> tuple[Node, ...] | list[Node]:
    if isinstance(node, Step):
        return ()
    return node.body


def _iter_nodes(nodes: tuple[Node, ...] | list[Node]) -> Iterator[Node]:
    for node in nodes:
        yield node
        yield from _iter_nodes(_child_nodes(node))


def _iter_steps(schedule: Schedule) -> Iterator[Step]:
    return (node for node in _iter_nodes(schedule.body) if isinstance(node, Step))


def is_queue_advance_step(step: Step) -> bool:
    return (
        isinstance(step.memory_resource, WorkQueue)
        and step.schedule_stage is ScheduleStage.ConsumerWork
    )


def validate_queue_advance_placement(
    nodes: tuple[Node, ...] | list[Node], *, parent_is_work_tile_loop: bool = False
) -> None:
    for node in nodes:
        if isinstance(node, Step):
            if is_queue_advance_step(node) and not parent_is_work_tile_loop:
                raise ScheduleError(
                    "get_and_advance_work_tile() must be a direct child of "
                    "work_tile_loop()."
                )
        elif isinstance(node, WorkTileLoop):
            validate_queue_advance_placement(node.body, parent_is_work_tile_loop=True)
        else:
            validate_queue_advance_placement(node.body)


def _format_guard(guard: BlockGuard) -> str:
    if guard is SKIPPABLE:
        return "Skippable"
    if guard is FIRST_ITER:
        return "FirstIter"
    if guard is LAST_ITER:
        return "LastIter"
    if isinstance(guard, Every):
        return f"Every(period={guard.period}, start={guard.start})"
    if isinstance(guard, OpaqueCondition):
        api = "when_false" if guard.negated else "when_true"
        return f"{api}(key={guard.key!r})"
    return repr(guard)


def _format_schedule(schedule: Schedule, *, show_routes: bool = True) -> str:
    resources = ", ".join(
        f"{resource.name}: {type(resource).__name__}" for resource in schedule.resources
    )
    lines = [f"Schedule({schedule.name!r}, resources=[{resources}])"]

    def emit(node: Node, indent: int) -> None:
        pad = " " * indent
        if isinstance(node, Step):
            label = f", label={node.label!r}" if node.label else ""
            lines.append(
                f"{pad}[#{node.unique_id}] Step({node.memory_resource.name}, "
                f"{node.schedule_stage.name}{label})"
            )
            if show_routes:
                for argument, value in node.input_values.items():
                    lines.append(f"{pad}      {argument} <- %{value.value_id}")
                for value in node.output_values:
                    lines.append(f"{pad}      -> %{value.value_id}")
            return
        if isinstance(node, ConditionalBlock):
            lines.append(
                f"{pad}ConditionalBlock(condition={_format_guard(node.condition)})"
            )
        elif isinstance(node, DomainLoop):
            carried = ", ".join(
                f"{name}=%{value.value_id}"
                for name, value in node.initial_values.items()
            )
            carried_suffix = f", carried=[{carried}]" if carried else ""
            lines.append(
                f"{pad}DomainLoop(start={node.start}, end={node.end}, "
                f"step={node.step}, unroll={node.unroll}{carried_suffix})"
            )
        else:
            suffix = (
                "" if node.skip_if is None else f", skip_if={node.skip_if.__name__}"
            )
            lines.append(f"{pad}WorkTileLoop({node.work_queue.name}{suffix})")
        for child in node.body:
            emit(child, indent + 2)

    for node in schedule.body:
        emit(node, 2)
    return "\n".join(lines)


class ScheduleError(ValueError):
    """A captured schedule violates the task-scheduling grammar."""


_active_builder: ContextVar["ScheduleBuilder | None"] = ContextVar(
    "_active_builder", default=None
)

_STANDARD_STAGE_METHODS = {
    "try_acquire": ScheduleStage.ProducerTryAcquire,
    "acquire": ScheduleStage.ProducerAcquire,
    "commit": ScheduleStage.ProducerCommit,
    "producer_work": ScheduleStage.ProducerWork,
    "producer_aux_work": ScheduleStage.ProducerAuxWork,
    "try_wait": ScheduleStage.ConsumerTryWait,
    "wait": ScheduleStage.ConsumerWait,
    "release": ScheduleStage.ConsumerRelease,
    "consumer_work": ScheduleStage.ConsumerWork,
    "consumer_aux_work": ScheduleStage.ConsumerAuxWork,
}


@dataclass(frozen=True)
class _WorkInfo:
    stage: ScheduleStage
    label: str | None
    parameter_names: tuple[str, ...] = ()
    routed_names: tuple[str, ...] = ()
    static_names: tuple[str, ...] = ()
    output_count: int = 0


class ScheduleBuilder:
    def __init__(self, name: str, resources: list[MemoryResource]) -> None:
        self.schedule = Schedule(name=name, body=[], resources=resources)
        self.stack: list[Schedule | ConditionalBlock | DomainLoop | WorkTileLoop] = [
            self.schedule
        ]
        self.values: set[ScheduleValue] = set()
        self.next_id = 0
        self.next_value_id = 0
        self.next_scope_id = 0
        self.scope_ids: dict[int, int] = {}
        self.scope_nodes: dict[int, object] = {}
        self.has_work_tile_loop = False
        self.after_work_tile_loop = False

    def current_scope(self) -> tuple[int, ...]:
        """Return stable lexical IDs for the currently open control-flow path."""
        result = []
        for item in self.stack[1:]:
            object_id = id(item)
            if object_id not in self.scope_ids:
                self.scope_ids[object_id] = self.next_scope_id
                self.scope_nodes[self.next_scope_id] = item
                self.next_scope_id += 1
            result.append(self.scope_ids[object_id])
        return tuple(result)

    def validate_value_use(self, value: ScheduleValue, argument: str) -> None:
        if value not in self.values:
            raise ScheduleError(
                f"input {argument!r} uses a value from another schedule"
            )
        current = self.current_scope()
        defining = value.scope
        if current[: len(defining)] == defining:
            return
        message = (
            f"input {argument!r} uses %{value.value_id} outside the "
            "control-flow scope that defines it"
        )
        escaped_loop = any(
            isinstance(self.scope_nodes.get(scope_id), DomainLoop)
            and scope_id not in current
            for scope_id in defining
        )
        if escaped_loop:
            message += (
                "; Python reassignment inside domain_loop() does not create "
                "loop-carried state—declare the initial value with carried=... "
                "and read and assign it through the loop handle"
            )
        raise ScheduleError(message)

    def make_value(
        self,
        producer: Step | DomainLoop,
        *,
        stage_resource: MemoryResource | None,
    ) -> ScheduleValue:
        value = ScheduleValue(
            value_id=self.next_value_id,
            producer_step=producer,
            stage_resource=stage_resource,
            scope=self.current_scope(),
        )
        self.next_value_id += 1
        self.values.add(value)
        return value

    def _inside(self, kind: type) -> bool:
        return any(isinstance(item, kind) for item in self.stack)

    def _in_condition(self) -> bool:
        return any(
            isinstance(item, ConditionalBlock)
            and isinstance(item.condition, (IterationPredicate, OpaqueCondition))
            for item in self.stack
        )

    def open_scope(self, node: ConditionalBlock | DomainLoop | WorkTileLoop) -> None:
        if self.after_work_tile_loop:
            raise ScheduleError("no blocks may follow work_tile_loop()")
        if isinstance(node, WorkTileLoop):
            if len(self.stack) != 1:
                raise ScheduleError("work_tile_loop() must be a top-level block")
            if self.has_work_tile_loop:
                raise ScheduleError("at most one work_tile_loop() is allowed")
            self.has_work_tile_loop = True
        elif isinstance(node, DomainLoop) and self._inside(DomainLoop):
            raise ScheduleError("domain_loop() cannot be nested")
        elif isinstance(node, ConditionalBlock):
            if node.condition is SKIPPABLE:
                loops = [item for item in self.stack if isinstance(item, WorkTileLoop)]
                if not loops or loops[-1].skip_if is None:
                    raise ScheduleError(
                        "skippable() requires work_tile_loop(skip_if=...)"
                    )
            elif isinstance(node.condition, IterationPredicate) and not self._inside(
                DomainLoop
            ):
                raise ScheduleError("iteration predicates require domain_loop()")
            if self._in_condition():
                raise ScheduleError("conditional scheduling blocks cannot be nested")
        self.stack[-1].body.append(node)
        self.stack.append(node)

    def close_scope(self, node: ConditionalBlock | DomainLoop | WorkTileLoop) -> None:
        if self.stack[-1] is not node:
            raise ScheduleError("schedule blocks closed out of order")
        self.stack.pop()
        if isinstance(node, WorkTileLoop):
            self.after_work_tile_loop = True

    def record_step(self, resource: MemoryResource, info: _WorkInfo) -> Step:
        if isinstance(resource, WorkQueue) and info.stage is ScheduleStage.ConsumerWork:
            if not isinstance(self.stack[-1], WorkTileLoop):
                raise ScheduleError(
                    "get_and_advance_work_tile() must be a direct child of "
                    "work_tile_loop()."
                )
        step = Step(
            memory_resource=resource,
            schedule_stage=info.stage,
            argument_order=info.parameter_names,
            label=info.label,
            unique_id=self.next_id,
        )
        self.next_id += 1
        self.stack[-1].body.append(step)
        return step

    def finalize(self) -> Schedule:
        if len(self.stack) != 1:
            raise ScheduleError("schedule ended with an unclosed block")
        validate_queue_advance_placement(self.schedule.body)

        def freeze(node: Node) -> Node:
            if isinstance(node, Step):
                return replace(
                    node,
                    input_values=MappingProxyType(dict(node.input_values)),
                    constexpr_kwargs=MappingProxyType(dict(node.constexpr_kwargs)),
                )
            if isinstance(node, DomainLoop):
                return replace(
                    node,
                    body=tuple(freeze(child) for child in node.body),
                    initial_values=MappingProxyType(dict(node.initial_values)),
                    iter_values=MappingProxyType(dict(node.iter_values)),
                    yield_values=MappingProxyType(dict(node.yield_values)),
                    result_values=MappingProxyType(dict(node.result_values)),
                )
            return replace(node, body=tuple(freeze(child) for child in node.body))

        return replace(
            self.schedule,
            body=tuple(freeze(node) for node in self.schedule.body),
            resources=tuple(self.schedule.resources),
        )


class ResourceProxy:
    def __init__(self, resource: MemoryResource, builder: ScheduleBuilder) -> None:
        self._resource = resource
        self._builder = builder
        self._work_info: dict[str, _WorkInfo] = {}
        self._discover_work()

    def _discover_work(self) -> None:
        cls = type(self._resource)
        for producer, attr in (
            (False, _CONSUMER_WORK_FNS_ATTR),
            (True, _PRODUCER_WORK_FNS_ATTR),
        ):
            for label, method_name in _collect_named_work_fns(cls, attr).items():
                if label in self._work_info:
                    raise ScheduleError(f"duplicate work method label {label!r}")
                method = getattr(cls, method_name)
                prefix = "producer" if producer else "consumer"
                self._work_info[label] = _WorkInfo(
                    stage=getattr(
                        method,
                        f"_{prefix}_work_stage",
                        ScheduleStage.ProducerWork
                        if producer
                        else ScheduleStage.ConsumerWork,
                    ),
                    label=label,
                    parameter_names=getattr(
                        method, f"_{prefix}_parameter_names", ()
                    ),
                    routed_names=getattr(method, f"_{prefix}_routed_names", ()),
                    static_names=getattr(method, f"_{prefix}_static_names", ()),
                    output_count=getattr(method, "_ts_route_output_count", 0),
                )

    def __getattr__(self, name: str) -> Callable[..., object]:
        if name.startswith("_"):
            raise AttributeError(name)
        info = self._work_info.get(name)
        if info is None and name in _STANDARD_STAGE_METHODS:
            if isinstance(self._resource, WorkQueue) and name in (
                "consumer_work",
                "producer_work",
            ):
                raise ScheduleError("WorkQueue requires its named work methods")
            info = _WorkInfo(stage=_STANDARD_STAGE_METHODS[name], label=None)
        if info is None:
            available = ", ".join(sorted({*self._work_info, *_STANDARD_STAGE_METHODS}))
            raise AttributeError(f"no work method {name!r}; available: {available}")

        def record(*args: object, **kwargs: object) -> object:
            step = self._builder.record_step(self._resource, info)
            parameter_names = info.parameter_names
            bound = dict(zip(parameter_names, args))
            for key, value in kwargs.items():
                if key in bound:
                    raise ScheduleError(f"multiple values for input {key!r}")
                bound[key] = value
            if set(bound) != set(parameter_names):
                missing = set(parameter_names) - set(bound)
                extra = set(bound) - set(parameter_names)
                raise ScheduleError(
                    f"invalid routed inputs; missing={missing}, extra={extra}"
                )
            explicit_static = set(info.static_names)
            for argument in parameter_names:
                value = bound[argument]
                if argument in explicit_static:
                    if isinstance(value, ScheduleValue):
                        raise ScheduleError(
                            f"static input {argument!r} cannot be a dataflow token"
                        )
                    step.constexpr_kwargs[argument] = value
                    continue
                if not isinstance(value, ScheduleValue):
                    step.constexpr_kwargs[argument] = value
                    continue
                self._builder.validate_value_use(value, argument)
                step.input_values[argument] = value
            stage_resource = (
                self._resource if self._resource.pipeline_config is not None else None
            )
            object.__setattr__(step, "stage_resource", stage_resource)
            output_values = tuple(
                self._builder.make_value(step, stage_resource=stage_resource)
                for _ in range(info.output_count)
            )
            object.__setattr__(step, "output_values", output_values)
            if not output_values:
                return None
            return output_values[0] if len(output_values) == 1 else output_values

        return record


def _require_active_builder(api: str) -> ScheduleBuilder:
    builder = _active_builder.get()
    if builder is None:
        raise ScheduleError(f"{api} must be called inside an @schedule function")
    return builder


_opaque_counter = itertools.count()


def _resolve_guard(
    builder: ScheduleBuilder,
    cond: object,
    key: Hashable | None,
    negated: bool,
) -> BlockGuard:
    if isinstance(cond, BlockGuard):
        if negated and not isinstance(cond, OpaqueCondition):
            raise ScheduleError("iteration and skippable predicates cannot be negated")
        return replace(cond, negated=True) if negated else cond
    if isinstance(cond, ScheduleValue):
        builder.validate_value_use(cond, "condition")
        producer = cond.producer_step
        schedule_stage = getattr(producer, "schedule_stage", None)
        label = getattr(producer, "label", None) or getattr(
            schedule_stage, "value", "domain_loop"
        )
        resource = getattr(producer, "memory_resource", None)
        return OpaqueCondition(
            key=key
            if key is not None
            else (
                getattr(resource, "name", "__loop__"),
                label,
                cond.value_id,
            ),
            negated=negated,
            resource=resource,
            method_label=label,
            route_token=cond,
        )
    return OpaqueCondition(
        key=key if key is not None else ("__opaque__", next(_opaque_counter)),
        negated=negated,
    )


@contextmanager
def _conditional(cond: object, key: Hashable | None, negated: bool):
    builder = _require_active_builder("when_false()" if negated else "when_true()")
    guard = _resolve_guard(builder, cond, key, negated)
    node = ConditionalBlock(body=[], condition=guard)
    builder.open_scope(node)
    try:
        yield
    finally:
        builder.close_scope(node)


def when_true(cond: object, *, key: Hashable | None = None):
    return _conditional(cond, key, False)


def when_false(cond: object, *, key: Hashable | None = None):
    return _conditional(cond, key, True)


class DomainLoopProxy:
    _RESERVED_NAMES = frozenset(
        {"builder", "node", "first_iter", "last_iter", "every"}
    )

    def __init__(self, builder: ScheduleBuilder, node: DomainLoop) -> None:
        object.__setattr__(self, "builder", builder)
        object.__setattr__(self, "node", node)

    def __getattr__(self, name: str) -> ScheduleValue:
        if name not in self.node.initial_values:
            raise AttributeError(name)
        if self.node in self.builder.stack:
            value = self.node.yield_values.get(name, self.node.iter_values[name])
        else:
            try:
                value = self.node.result_values[name]
            except KeyError as error:
                raise ScheduleError("domain loop did not finish capturing") from error
        return value

    def __setattr__(self, name: str, value: ScheduleValue) -> None:
        if name in {"builder", "node"}:
            object.__setattr__(self, name, value)
            return
        if name not in self.node.initial_values:
            raise AttributeError(name)
        self._assign(name, value)

    def _assign(self, name: str, value: ScheduleValue) -> None:
        self._check()
        if self.builder.stack[-1] is not self.node:
            raise ScheduleError(
                "domain-loop carried assignments must be unconditional"
            )
        if not isinstance(value, ScheduleValue):
            raise ScheduleError(
                f"carried value {name!r} must be a work-call output token"
            )
        self.builder.validate_value_use(value, name)
        initial = self.node.initial_values[name]
        if value.stage_resource is not initial.stage_resource:
            raise ScheduleError(
                f"carried value {name!r} changes pipeline stage provenance"
            )
        updated = dict(self.node.yield_values)
        updated[name] = value
        object.__setattr__(self.node, "yield_values", updated)

    def _create_iter_values(self) -> None:
        values = {}
        for name, initial in self.node.initial_values.items():
            values[name] = self.builder.make_value(
                self.node,
                stage_resource=initial.stage_resource,
            )
        object.__setattr__(self.node, "iter_values", values)

    def _finish(self) -> None:
        values = {
            name: self.node.yield_values.get(name, self.node.iter_values[name])
            for name in self.node.initial_values
        }
        object.__setattr__(self.node, "yield_values", values)

    def _create_results(self) -> None:
        values = {}
        for name, initial in self.node.initial_values.items():
            values[name] = self.builder.make_value(
                self.node,
                stage_resource=initial.stage_resource,
            )
        object.__setattr__(self.node, "result_values", values)

    def _check(self) -> None:
        if self.node not in self.builder.stack:
            raise ScheduleError("domain-loop handle has expired")

    def first_iter(self):
        self._check()
        return when_true(FIRST_ITER)

    def last_iter(self):
        self._check()
        return when_true(LAST_ITER)

    def every(self, period: int, *, start: int = 0):
        self._check()
        return when_true(Every(period, start))


class WorkTileLoopProxy:
    def __init__(self, builder: ScheduleBuilder, node: WorkTileLoop) -> None:
        self.builder = builder
        self.node = node

    def skippable(self) -> AbstractContextManager[None]:
        if self.node not in self.builder.stack:
            raise ScheduleError("work-tile-loop handle has expired")
        return _conditional(SKIPPABLE, None, False)


@contextmanager
def domain_loop(
    *bounds: object,
    unroll: int | None = None,
    carried: Mapping[str, ScheduleValue] | None = None,
):
    """Capture a range-like loop with optional named loop-carried SSA values."""
    builder = _require_active_builder("domain_loop()")
    if len(bounds) == 1:
        start, end, step = 0, bounds[0], 1
    elif len(bounds) == 2:
        start, end, step = bounds[0], bounds[1], 1
    elif len(bounds) == 3:
        start, end, step = bounds
    else:
        raise ScheduleError("domain_loop() takes 1 to 3 bounds like range()")
    if step == 0:
        raise ScheduleError("domain_loop() step must be non-zero")
    if unroll is not None and (type(unroll) is not int or unroll < 1):
        raise ScheduleError("domain_loop() unroll must be a positive integer")
    if carried is None:
        initial_values = {}
    elif not isinstance(carried, Mapping):
        raise ScheduleError("domain_loop carried values must be a mapping")
    else:
        initial_values = dict(carried)
    for name, value in initial_values.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            or name in DomainLoopProxy._RESERVED_NAMES
        ):
            raise ScheduleError(
                "domain-loop carried names must be usable loop attributes"
            )
        if not isinstance(value, ScheduleValue):
            raise ScheduleError(
                f"carried value {name!r} must be a work-call output token"
            )
        builder.validate_value_use(value, name)
    node = DomainLoop(
        start=start,
        end=end,
        step=step,
        unroll=unroll,
        body=[],
        initial_values=initial_values,
    )
    builder.open_scope(node)
    proxy = DomainLoopProxy(builder, node)
    proxy._create_iter_values()
    try:
        yield proxy
        proxy._finish()
    finally:
        builder.close_scope(node)
    proxy._create_results()


def dynamic_domain_bound(
    resolver: Callable[[object], object], *, type_name: str = "Int32"
) -> DynamicDomainBound:
    """Create a device-resolved bound for :func:`domain_loop`.

    The resolver is invoked inside the kernel with the manager's named
    ``tasks_inputs`` object. For example, ``lambda inputs: inputs.k // 64``
    creates a runtime K-tile count while preserving a stable host schedule.
    """

    return DynamicDomainBound(resolver, type_name)


@contextmanager
def work_tile_loop(wq: object, *, skip_if: Callable[..., object] | None = None):
    builder = _require_active_builder("work_tile_loop()")
    if not isinstance(wq, ResourceProxy) or not isinstance(wq._resource, WorkQueue):
        raise ScheduleError("work_tile_loop() requires a traced WorkQueue")
    if skip_if is not None and not callable(skip_if):
        raise ScheduleError("work_tile_loop(skip_if=...) must be callable")
    node = WorkTileLoop(body=[], work_queue=wq._resource, skip_if=skip_if)
    builder.open_scope(node)
    try:
        yield WorkTileLoopProxy(builder, node)
    finally:
        builder.close_scope(node)


def schedule(fn: Callable[..., None]) -> Callable[..., Schedule]:
    """Trace ``fn`` once on the host and return its immutable schedule tree."""

    @functools.wraps(fn)
    def traced(*resources: object) -> Schedule:
        parameters = tuple(inspect.signature(fn).parameters)
        inject_stage_info = bool(parameters) and parameters[0] == "stage_info"
        if "stage_info" in parameters[1:]:
            raise ScheduleError("stage_info must be the first @schedule parameter")
        expected_resources = len(parameters) - int(inject_stage_info)
        if len(resources) != expected_resources:
            raise ScheduleError(
                f"{fn.__name__} expects {expected_resources} "
                f"resources, got {len(resources)}"
            )
        from .pipeline_group import _GroupMemberRef

        members = [
            resource.member if isinstance(resource, _GroupMemberRef) else resource
            for resource in resources
        ]
        if not all(isinstance(resource, MemoryResource) for resource in members):
            raise ScheduleError("@schedule arguments must be MemoryResource instances")
        for original, member in zip(resources, members):
            if member.pipeline_group is not None and not isinstance(
                original, _GroupMemberRef
            ):
                raise ScheduleError(
                    f"grouped resource {member.name!r} must be passed as group.member"
                )
        builder = ScheduleBuilder(fn.__name__, list(members))
        token = _active_builder.set(builder)
        try:
            proxies = tuple(ResourceProxy(resource, builder) for resource in members)
            if inject_stage_info:
                fn(ScheduleStageInfo(), *proxies)
            else:
                fn(*proxies)
        finally:
            _active_builder.reset(token)
        return builder.finalize()

    return traced
