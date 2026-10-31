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
from types import MappingProxyType, TracebackType
from typing import Callable, Hashable, Mapping, NoReturn

from .enums import (
    BlockGuard,
    ComplementedIterationPredicate,
    Every,
    FIRST_ITER,
    GuardAnd,
    GuardOr,
    IterationPredicate,
    LAST_ITER,
    OpaqueCondition,
    SKIPPABLE,
    ScheduleStage,
    iter_leaves,
)
from .ir import ScheduleValue, _NegatedScheduleValue
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
class BreakLoop:
    """Exit the enclosing loop, skipping the remaining iteration body."""

    exit_values: tuple[ScheduleValue, ...] = ()


@dataclass(kw_only=True, eq=False, frozen=True)
class ConditionalBlock:
    """A region executed only when its guard holds.

    ``verify_num_fires`` is a verifier-only assertion for blocks inside
    ``while_true_loop()``: the exact number of modeled rounds on which the
    effective guard, including enclosing guards, holds. It constrains the
    opaque exit histories the checker explores and has no device effect.
    """

    body: tuple["Node", ...] | list["Node"]
    condition: BlockGuard
    verify_num_fires: int | None = None


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


@dataclass(kw_only=True, eq=False, frozen=True)
class WhileTrueLoop:
    """A dynamic loop that repeats its body until a ``break_loop()`` runs.

    ``init`` and ``advance`` are consumer-work hook steps outside ``body``.
    Each returns the loop cursor, which device work reads as
    ``stage_info.loop_offset``: ``init`` runs once before the first round and
    ``advance`` after every round that does not break. The exit formula is
    the disjunction of the guards that dominate each ``break_loop()``.
    ``verify_num_iters`` lists the modeled round counts the exhaustive checker
    explores; a count ``N`` is checked as ``N - 1`` retries and one break.
    """

    init: Step
    advance: Step
    verify_num_iters: tuple[int, ...]
    body: tuple["Node", ...] | list["Node"]


Node = Step | BreakLoop | ConditionalBlock | DomainLoop | WorkTileLoop | WhileTrueLoop


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
    if isinstance(node, (Step, BreakLoop)):
        return ()
    return node.body


def _iter_nodes(nodes: tuple[Node, ...] | list[Node]) -> Iterator[Node]:
    """Yield nodes in execution preorder, including while-loop hook steps."""
    for node in nodes:
        yield node
        if isinstance(node, WhileTrueLoop):
            yield node.init
            yield from _iter_nodes(node.body)
            yield node.advance
        else:
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
            validate_queue_advance_placement(_child_nodes(node))


def validate_break_placement(nodes, *, in_domain_loop=False) -> None:
    for node in nodes:
        if isinstance(node, BreakLoop) and not in_domain_loop:
            raise ScheduleError(
                "break_loop() must be called inside domain_loop() or "
                "while_true_loop()"
            )
        validate_break_placement(
            _child_nodes(node),
            in_domain_loop=in_domain_loop
            or isinstance(node, (DomainLoop, WhileTrueLoop)),
        )


def contains_break(nodes) -> bool:
    """Whether this scope can exit its enclosing domain loop."""
    return any(
        isinstance(node, BreakLoop)
        or (isinstance(node, ConditionalBlock) and contains_break(node.body))
        for node in nodes
    )


def while_loop_break_clauses(
    loop: WhileTrueLoop,
) -> tuple[tuple[BlockGuard, ...], ...]:
    """Return the while-loop exit formula in source order.

    Each clause is the conjunction of the guards that dominate one
    ``break_loop()``; the loop exits when any clause holds. An empty clause is
    an unconditional break.
    """
    clauses = []

    def walk(nodes, chain):
        for node in nodes:
            if isinstance(node, BreakLoop):
                clauses.append(chain)
            elif isinstance(node, ConditionalBlock):
                inner = chain
                if node.condition is not SKIPPABLE:
                    inner = chain + (node.condition,)
                walk(node.body, inner)

    walk(loop.body, ())
    return tuple(clauses)


def _format_exit_formula(clauses: tuple[tuple[BlockGuard, ...], ...]) -> str:
    if not clauses:
        return "never"
    rendered = [
        " and ".join(_format_guard(guard) for guard in clause) if clause else "true"
        for clause in clauses
    ]
    if len(rendered) == 1:
        return rendered[0]
    return " or ".join(
        f"({clause})" if " and " in clause else clause for clause in rendered
    )


def _format_guard_expression(guard: BlockGuard) -> str:
    if isinstance(guard, OpaqueCondition):
        value = f"key={guard.key!r}"
        return f"~({value})" if guard.negated else value
    if isinstance(guard, (GuardAnd, GuardOr)):
        operator = " and " if isinstance(guard, GuardAnd) else " or "
        return (
            "("
            + operator.join(_format_guard_expression(item) for item in guard.operands)
            + ")"
        )
    return _format_guard(guard)


def _format_guard(guard: BlockGuard) -> str:
    if guard is SKIPPABLE:
        return "Skippable"
    if isinstance(guard, (GuardAnd, GuardOr)):
        return f"when_true({_format_guard_expression(guard)})"
    if isinstance(guard, ComplementedIterationPredicate):
        return f"~{_format_guard(guard.predicate)}"
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

    def emit_step(node: Step, pad: str, role: str = "") -> None:
        label = f", label={node.label!r}" if node.label else ""
        lines.append(
            f"{pad}[#{node.unique_id}] {role}Step({node.memory_resource.name}, "
            f"{node.schedule_stage.name}{label})"
        )
        if show_routes:
            for argument, value in node.input_values.items():
                lines.append(f"{pad}      {argument} <- %{value.value_id}")
            for value in node.output_values:
                lines.append(f"{pad}      -> %{value.value_id}")

    def emit(node: Node, indent: int) -> None:
        pad = " " * indent
        if isinstance(node, BreakLoop):
            values = ", ".join(f"%{value.value_id}" for value in node.exit_values)
            suffix = f"values=[{values}]" if values and show_routes else ""
            lines.append(f"{pad}BreakLoop({suffix})")
            return
        if isinstance(node, Step):
            emit_step(node, pad)
            return
        if isinstance(node, WhileTrueLoop):
            lines.append(
                f"{pad}WhileTrueLoop(init={node.init.memory_resource.name}."
                f"{node.init.label}, advance={node.advance.memory_resource.name}."
                f"{node.advance.label}, verify_num_iters={node.verify_num_iters}, "
                f"exit={_format_exit_formula(while_loop_break_clauses(node))})"
            )
            emit_step(node.init, pad + "  ", "init ")
            for child in node.body:
                emit(child, indent + 2)
            emit_step(node.advance, pad + "  ", "advance ")
            return
        if isinstance(node, ConditionalBlock):
            fires = (
                ""
                if node.verify_num_fires is None
                else f", verify_num_fires={node.verify_num_fires}"
            )
            lines.append(
                f"{pad}ConditionalBlock(condition={_format_guard(node.condition)}"
                f"{fires})"
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
    output_count: int = 0


class ScheduleBuilder:
    def __init__(self, name: str, resources: list[MemoryResource]) -> None:
        self.schedule = Schedule(name=name, body=[], resources=resources)
        self.stack: list[
            Schedule | ConditionalBlock | DomainLoop | WorkTileLoop | WhileTrueLoop
        ] = [self.schedule]
        self.values: set[ScheduleValue] = set()
        self.next_id = 0
        self.next_value_id = 0
        self.next_scope_id = 0
        self.scope_ids: dict[int, int] = {}
        self.scope_nodes: dict[int, object] = {}
        self.has_work_tile_loop = False
        self.after_work_tile_loop = False
        self.has_while_true_loop = False

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
                "loop-carried state—use the functional domain_loop() form, "
                "pass the initial value to loop_body, and return its next value"
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

    def _in_runtime_condition(self) -> bool:
        return any(
            isinstance(item, ConditionalBlock) and item.condition is not SKIPPABLE
            for item in self.stack
        )

    def open_scope(
        self, node: ConditionalBlock | DomainLoop | WorkTileLoop | WhileTrueLoop
    ) -> None:
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
        elif isinstance(node, DomainLoop) and self._inside(WhileTrueLoop):
            raise ScheduleError("domain_loop() cannot be nested inside while_true_loop()")
        elif isinstance(node, WhileTrueLoop):
            if self.has_while_true_loop:
                raise ScheduleError("at most one while_true_loop() is allowed per schedule")
            if self._inside(DomainLoop):
                raise ScheduleError("while_true_loop() cannot be nested inside domain_loop()")
            if self._in_runtime_condition():
                raise ScheduleError(
                    "while_true_loop() cannot be nested inside a condition block"
                )
            loops = [item for item in self.stack if isinstance(item, WorkTileLoop)]
            if (
                loops
                and loops[-1].skip_if is not None
                and not any(
                    isinstance(item, ConditionalBlock) and item.condition is SKIPPABLE
                    for item in self.stack
                )
            ):
                raise ScheduleError(
                    "with work_tile_loop(skip_if=...), while_true_loop() must be "
                    "inside wtl.skippable()"
                )
            self.has_while_true_loop = True
        elif isinstance(node, ConditionalBlock):
            if node.verify_num_fires is not None:
                count = node.verify_num_fires
                if type(count) is not int or count < 0:
                    raise ScheduleError(
                        f"verify_num_fires must be a non-negative int, got {count!r}"
                    )
                if not self._inside(WhileTrueLoop):
                    raise ScheduleError(
                        "verify_num_fires= is only valid inside while_true_loop(); "
                        "a domain_loop() already has a static trip count"
                    )
            if node.condition is SKIPPABLE:
                loops = [item for item in self.stack if isinstance(item, WorkTileLoop)]
                if not loops or loops[-1].skip_if is None:
                    raise ScheduleError(
                        "skippable() requires work_tile_loop(skip_if=...)"
                    )
            elif any(
                isinstance(leaf, IterationPredicate)
                for leaf in iter_leaves(node.condition)
            ) and not self._inside(DomainLoop):
                raise ScheduleError("iteration predicates require domain_loop()")
        self.stack[-1].body.append(node)
        self.stack.append(node)

    def close_scope(
        self, node: ConditionalBlock | DomainLoop | WorkTileLoop | WhileTrueLoop
    ) -> None:
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

    def allocate_hook_step(self, resource: MemoryResource, info: _WorkInfo) -> Step:
        """Number a while-loop hook step without placing it in the body."""
        step = Step(
            memory_resource=resource,
            schedule_stage=info.stage,
            argument_order=info.parameter_names,
            stage_resource=(
                resource if resource.pipeline_config is not None else None
            ),
            label=info.label,
            unique_id=self.next_id,
        )
        self.next_id += 1
        return step

    def finalize(self) -> Schedule:
        if len(self.stack) != 1:
            raise ScheduleError("schedule ended with an unclosed block")
        validate_queue_advance_placement(self.schedule.body)
        validate_break_placement(self.schedule.body)
        loop_kinds = {
            type(node)
            for node in _iter_nodes(self.schedule.body)
            if isinstance(node, (DomainLoop, WhileTrueLoop))
        }
        if len(loop_kinds) > 1:
            raise ScheduleError(
                "domain_loop() and while_true_loop() cannot both appear in one "
                "schedule"
            )

        def freeze(node: Node) -> Node:
            if isinstance(node, BreakLoop):
                return node
            if isinstance(node, Step):
                return replace(
                    node,
                    input_values=MappingProxyType(dict(node.input_values)),
                    constexpr_kwargs=MappingProxyType(dict(node.constexpr_kwargs)),
                )
            if isinstance(node, WhileTrueLoop):
                return replace(
                    node,
                    init=freeze(node.init),
                    advance=freeze(node.advance),
                    body=tuple(freeze(child) for child in node.body),
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
            for argument in parameter_names:
                value = bound[argument]
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

        # Lets while_true_loop(init=smem.method) resolve the uncalled method.
        record._ts_loop_hook = (self._builder, self._resource, info)
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
    api: str,
    key_from_composite: bool = False,
) -> BlockGuard:
    """Resolve a captured condition to a guard in negation-normal form.

    Composite operands resolve recursively. A ``key=`` on a composite applies
    to each opaque leaf and leaves iteration predicates unchanged.
    """
    # Must precede the literal fallback below: an iteration-block context is
    # not a guard, and would otherwise capture as a value-less opaque guard.
    if isinstance(cond, _IterationBlockContext):
        cond._operand_error()
    if cond is SKIPPABLE:
        if negated or key_from_composite:
            raise ScheduleError(
                "skippable() cannot be negated or combined with other conditions"
            )
        return cond
    if isinstance(cond, (GuardAnd, GuardOr)):
        if (
            key is not None
            and not key_from_composite
            and all(isinstance(leaf, IterationPredicate) for leaf in iter_leaves(cond))
        ):
            raise ScheduleError(
                f"{api}(): key= on a composite condition requires at least one "
                "opaque leaf to correlate"
            )
        guard = type(cond)(
            tuple(
                _resolve_guard(
                    builder, operand, key, False, api, key_from_composite=True
                )
                for operand in cond.operands
            )
        )
        return ~guard if negated else guard
    if isinstance(cond, IterationPredicate):
        return ~cond if negated else cond
    if isinstance(cond, OpaqueCondition):
        if key is not None:
            cond = replace(cond, key=key)
        return ~cond if negated else cond
    if isinstance(cond, _NegatedScheduleValue):
        cond = cond.value
        negated = not negated
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
    if isinstance(cond, (BlockGuard, AbstractContextManager)):
        raise ScheduleError(
            f"{api}() cannot use {type(cond).__name__} as a condition operand"
        )
    return OpaqueCondition(
        key=key if key is not None else ("__opaque__", next(_opaque_counter)),
        negated=negated,
    )


@contextmanager
def _conditional(
    cond: object,
    key: Hashable | None,
    negated: bool,
    verify_num_fires: int | None = None,
):
    api = "when_false" if negated else "when_true"
    builder = _require_active_builder(f"{api}()")
    guard = _resolve_guard(builder, cond, key, negated, api)
    node = ConditionalBlock(
        body=[], condition=guard, verify_num_fires=verify_num_fires
    )
    builder.open_scope(node)
    try:
        yield
    finally:
        builder.close_scope(node)


def when_true(
    cond: object,
    *,
    key: Hashable | None = None,
    verify_num_fires: int | None = None,
):
    """Capture a guarded region, evaluated only when all enclosing guards fire.

    ``cond`` is a work-call output, an iteration predicate (``d.first``,
    ``d.last``, ``d.every_pred(...)``, ``FIRST_ITER``, ``LAST_ITER``), or an
    expression combining them with ``&``, ``|``, and ``~``. Python ``and``,
    ``or``, and ``not`` are rejected because they cannot be captured. A
    ``key=`` on a composite correlates every opaque leaf.

    Inside ``while_true_loop()``, ``verify_num_fires=N`` asserts that the block
    runs on exactly ``N`` of the loop's modeled rounds. The exhaustive checker
    uses it to discard impossible opaque exit histories; it emits no code.
    """
    return _conditional(cond, key, False, verify_num_fires)


def when_false(
    cond: object,
    *,
    key: Hashable | None = None,
    verify_num_fires: int | None = None,
):
    """Capture a negated guarded region; nested values retain lexical scope.

    This is ``when_true(~cond)``; composite guards are converted to
    negation-normal form. ``verify_num_fires`` behaves as in :func:`when_true`.
    """
    return _conditional(cond, key, True, verify_num_fires)


def _require_active_domain_loop(api: str) -> None:
    builder = _require_active_builder(f"{api}()")
    if not any(isinstance(node, DomainLoop) for node in builder.stack):
        raise ScheduleError(f"{api}() must be called inside domain_loop()")


def first_iter():
    """Capture a block that runs only on the first domain-loop iteration."""
    _require_active_domain_loop("first_iter")
    return _IterationBlockContext(FIRST_ITER, "first_iter")


def break_loop(*values: ScheduleValue) -> None:
    """Exit the enclosing loop when execution reaches this operation.

    Remaining work and the functional loop's backedge are skipped. Explicit
    values become the loop's results, in carried-input order. With no arguments,
    results retain their values at entry to the interrupted iteration.
    Inside ``while_true_loop()``, the guards that dominate this call form one
    clause of the loop's exit formula; that loop carries no values.
    """
    builder = _require_active_builder("break_loop()")
    loop = next(
        (
            node
            for node in reversed(builder.stack)
            if isinstance(node, (DomainLoop, WhileTrueLoop))
        ),
        None,
    )
    if loop is None:
        raise ScheduleError(
            "break_loop() must be called inside domain_loop() or while_true_loop()"
        )
    if isinstance(loop, WhileTrueLoop):
        if values:
            raise ScheduleError("while_true_loop() break_loop() takes no values")
        builder.stack[-1].body.append(BreakLoop())
        return
    if values and len(values) != len(loop.initial_values):
        raise ScheduleError(
            "break_loop() must supply one value for every carried input; "
            f"expected {len(loop.initial_values)}, got {len(values)}"
        )
    for (name, initial), value in zip(loop.initial_values.items(), values):
        if not isinstance(value, ScheduleValue):
            raise ScheduleError(f"break value {name!r} must be a routed schedule value")
        builder.validate_value_use(value, name)
        if value.stage_resource is not initial.stage_resource:
            raise ScheduleError(f"break value {name!r} changes pipeline stage provenance")
    builder.stack[-1].body.append(BreakLoop(exit_values=values))


def last_iter():
    """Capture a block that runs only on the last domain-loop iteration."""
    _require_active_domain_loop("last_iter")
    return _IterationBlockContext(LAST_ITER, "last_iter")


def every(period: int, *, start: int = 0):
    """Capture a block that runs periodically within the current domain loop."""
    _require_active_domain_loop("every")
    return _IterationBlockContext(Every(period, start), "every")


class _IterationBlockContext(AbstractContextManager):
    """Context-manager spelling of a predicate, never a boolean operand."""

    def __init__(self, predicate: IterationPredicate, api: str) -> None:
        self._context = when_true(predicate)
        self._api = api

    def __enter__(self) -> None:
        return self._context.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._context.__exit__(exc_type, exc_value, traceback)

    def _operand_error(self) -> NoReturn:
        raise ScheduleError(
            f"{self._api}() is a context manager, not a condition operand; use "
            "d.first / d.last / d.every_pred(...), or FIRST_ITER / LAST_ITER"
        )

    def __and__(self, _other: object) -> NoReturn:
        self._operand_error()

    def __rand__(self, _other: object) -> NoReturn:
        self._operand_error()

    def __or__(self, _other: object) -> NoReturn:
        self._operand_error()

    def __ror__(self, _other: object) -> NoReturn:
        self._operand_error()

    def __invert__(self) -> NoReturn:
        self._operand_error()

    def __bool__(self) -> NoReturn:
        self._operand_error()


class DomainLoopProxy:
    __slots__ = ("builder", "node")

    def __init__(self, builder: ScheduleBuilder, node: DomainLoop) -> None:
        object.__setattr__(self, "builder", builder)
        object.__setattr__(self, "node", node)

    def _set_yield_values(self, values: tuple[object, ...]) -> None:
        self._check()
        if self.builder.stack[-1] is not self.node:
            raise ScheduleError(
                "functional domain-loop carried results must be unconditional"
            )
        updated = {}
        for (name, initial), value in zip(self.node.initial_values.items(), values):
            if not isinstance(value, ScheduleValue):
                raise ScheduleError(
                    f"carried value {name!r} must be a work-call output token"
                )
            self.builder.validate_value_use(value, name)
            if value.stage_resource is not initial.stage_resource:
                raise ScheduleError(
                    f"carried value {name!r} changes pipeline stage provenance"
                )
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
        return first_iter()

    def break_loop(self, *values: ScheduleValue) -> None:
        self._check()
        if _active_builder.get() is not self.builder:
            raise ScheduleError("domain-loop handle belongs to another schedule")
        break_loop(*values)

    def last_iter(self):
        self._check()
        return last_iter()

    def every(self, period: int, *, start: int = 0):
        self._check()
        return every(period, start=start)

    @property
    def first(self) -> IterationPredicate:
        """First-iteration predicate, usable in ``&`` / ``|`` / ``~``."""
        self._check()
        return FIRST_ITER

    @property
    def last(self) -> IterationPredicate:
        """Last-iteration predicate, usable in ``&`` / ``|`` / ``~``."""
        self._check()
        return LAST_ITER

    def every_pred(self, period: int, *, start: int = 0) -> IterationPredicate:
        """Periodic predicate, usable in ``&`` / ``|`` / ``~``."""
        self._check()
        return Every(period, start)


class WorkTileLoopProxy:
    def __init__(self, builder: ScheduleBuilder, node: WorkTileLoop) -> None:
        self.builder = builder
        self.node = node

    def skippable(self) -> AbstractContextManager[None]:
        if self.node not in self.builder.stack:
            raise ScheduleError("work-tile-loop handle has expired")
        return _conditional(SKIPPABLE, None, False)


@contextmanager
def _domain_loop_scope(
    *bounds: object,
    unroll: int | None = None,
    initial_values: Mapping[str, ScheduleValue] | None = None,
):
    """Capture a range-like loop with internal loop-carried SSA values."""
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
    if initial_values is None:
        initial_values = {}
    elif not isinstance(initial_values, Mapping):
        raise ScheduleError("domain_loop initial values must be a mapping")
    else:
        initial_values = dict(initial_values)
    for name, value in initial_values.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ScheduleError("domain-loop carried names must be valid identifiers")
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


def _functional_domain_loop(
    start: object,
    end: object,
    step: object,
    body: Callable[..., object],
    initial_values: tuple[object, ...],
    *,
    unroll: int | None,
):
    """Capture a callback body and return its loop-carried results."""
    parameters = tuple(inspect.signature(body).parameters.values())
    expected_parameters = len(initial_values)
    if len(parameters) != expected_parameters or any(
        parameter.kind
        not in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    ):
        raise ScheduleError(
            "functional domain_loop body must take exactly "
            f"{len(initial_values)} carried values"
        )
    carried_names = tuple(parameter.name for parameter in parameters)
    initial_values_by_name = dict(zip(carried_names, initial_values))
    with _domain_loop_scope(
        start,
        end,
        step,
        unroll=unroll,
        initial_values=initial_values_by_name,
    ) as loop:
        iter_values = tuple(loop.node.iter_values[name] for name in carried_names)
        yielded = body(*iter_values)
        if len(initial_values) == 0:
            yielded_values = () if yielded is None else (yielded,)
        elif len(initial_values) == 1 and isinstance(yielded, ScheduleValue):
            yielded_values = (yielded,)
        elif isinstance(yielded, tuple):
            yielded_values = yielded
        else:
            yielded_values = (yielded,)
        if len(yielded_values) != len(initial_values):
            raise ScheduleError(
                "functional domain_loop body must return one value for every "
                f"carried input; expected {len(initial_values)}, got "
                f"{len(yielded_values)}"
            )
        loop._set_yield_values(yielded_values)
    results = tuple(loop.node.result_values[name] for name in carried_names)
    if not results:
        return None
    return results[0] if len(results) == 1 else results


def domain_loop(
    *args: object,
    unroll: int | None = None,
):
    """Capture a range-like loop in context-manager or functional form.

    ``domain_loop(start, end, step, body, *initial_values)`` invokes ``body``
    once during schedule capture with the iteration values. Its returned
    work-call outputs form the loop backedge and are returned as the post-loop
    results. Device work reads the current offset from ``stage_info.loop_offset``.
    The context-manager form captures loops without loop-carried values.
    ``unroll=N`` groups N device iterations in a main loop followed by a tail;
    runtime bounds, carried values, and iteration guards retain their semantics.
    Domains containing ``break_loop()`` retain an ordinary loop so exits are
    immediate even when an unroll hint is supplied.
    """
    if len(args) >= 4:
        start, end, step, body, *initial_values = args
        if not callable(body):
            raise ScheduleError(
                "the fourth functional domain_loop argument must be callable"
            )
        return _functional_domain_loop(
            start,
            end,
            step,
            body,
            tuple(initial_values),
            unroll=unroll,
        )
    return _domain_loop_scope(*args, unroll=unroll)


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


class WhileTrueLoopProxy:
    """Yielded by ``while_true_loop()``; records the loop's break points."""

    def __init__(self, builder: ScheduleBuilder, node: WhileTrueLoop) -> None:
        self.builder = builder
        self.node = node

    def break_loop(self) -> None:
        if self.node not in self.builder.stack:
            raise ScheduleError("while-true-loop handle has expired")
        if _active_builder.get() is not self.builder:
            raise ScheduleError("while-true-loop handle belongs to another schedule")
        break_loop()


def _normalize_verify_num_iters(verify_num_iters: object) -> tuple[int, ...]:
    """Return modeled round counts; ``None`` checks 1 and 2 rounds."""
    if verify_num_iters is None:
        return (1, 2)
    counts = (
        (verify_num_iters,)
        if type(verify_num_iters) is int
        else verify_num_iters
        if isinstance(verify_num_iters, tuple)
        else None
    )
    if counts is None:
        raise ScheduleError(
            "while_true_loop() verify_num_iters must be an int, a tuple of ints, "
            "or None"
        )
    if not counts:
        raise ScheduleError("while_true_loop() verify_num_iters must be non-empty")
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ScheduleError(
            "while_true_loop() verify_num_iters must contain only positive ints"
        )
    return counts


def _resolve_loop_hook(
    builder: ScheduleBuilder, api: str, hook: object
) -> tuple[MemoryResource, _WorkInfo]:
    """Validate an uncalled traced work method used as a loop-cursor hook."""
    traced = getattr(hook, "_ts_loop_hook", None)
    if traced is None:
        raise ScheduleError(
            f"while_true_loop() {api}= must be a traced resource work method such "
            f"as smem.retry_{api}, not {type(hook).__name__}"
        )
    hook_builder, resource, info = traced
    if hook_builder is not builder:
        raise ScheduleError(f"while_true_loop() {api}= belongs to another schedule")
    if info.label is None or info.stage not in (
        ScheduleStage.ConsumerWork,
        ScheduleStage.ConsumerAuxWork,
    ):
        raise ScheduleError(
            f"while_true_loop() {api}= must name a consumer work method that "
            "returns the loop cursor"
        )
    if info.output_count != 1:
        raise ScheduleError(
            f"while_true_loop() {api}= must declare outputs=1 for the loop cursor"
        )
    if info.parameter_names:
        raise ScheduleError(
            f"while_true_loop() {api}= cannot take routed inputs; read the current "
            "cursor from stage_info.loop_offset"
        )
    return resource, info


@contextmanager
def while_true_loop(
    *,
    init: Callable[..., object],
    advance: Callable[..., object],
    verify_num_iters: int | tuple[int, ...] | None = None,
):
    """Capture a dynamic loop that repeats until a ``break_loop()`` runs.

    ``init`` and ``advance`` are uncalled consumer work methods declared with
    ``outputs=1`` and no routed inputs, for example ``smem.retry_init``. Their
    result is the loop cursor seen by device work as
    ``stage_info.loop_offset``: ``init`` seeds it before the first round and
    ``advance`` computes the next cursor after a round without a break.

    Each ``loop.break_loop()`` contributes the conjunction of its dominating
    ``when_true`` / ``when_false`` guards to the exit formula, and work after it
    in the same round does not run. The exhaustive checker models each count
    ``N`` in ``verify_num_iters`` as ``N - 1`` retry rounds followed by a
    break round; ``None`` checks ``(1, 2)``.

    A schedule may contain either a ``domain_loop()`` or a
    ``while_true_loop()``, and neither may be nested inside the other.
    """
    builder = _require_active_builder("while_true_loop()")
    init_resource, init_info = _resolve_loop_hook(builder, "init", init)
    advance_resource, advance_info = _resolve_loop_hook(builder, "advance", advance)
    node = WhileTrueLoop(
        init=builder.allocate_hook_step(init_resource, init_info),
        advance=None,
        verify_num_iters=_normalize_verify_num_iters(verify_num_iters),
        body=[],
    )
    builder.open_scope(node)
    try:
        yield WhileTrueLoopProxy(builder, node)
    finally:
        # Number the advance hook after the body to keep IDs in execution order.
        object.__setattr__(
            node,
            "advance",
            builder.allocate_hook_step(advance_resource, advance_info),
        )
        builder.close_scope(node)
    clauses = while_loop_break_clauses(node)
    if not clauses:
        raise ScheduleError("while_true_loop() requires at least one loop.break_loop()")
    if any(count > 1 for count in node.verify_num_iters) and any(
        not clause for clause in clauses
    ):
        raise ScheduleError(
            "while_true_loop() with verify_num_iters > 1 requires a retry "
            "assignment, but a break_loop() is unconditional"
        )


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
