# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded exhaustive interleaving checker for captured schedules."""

import itertools
from collections import deque
from dataclasses import dataclass, field, replace

from .enums import (
    FIRST_ITER,
    LAST_ITER,
    BlockGuard,
    Every,
    IterationPredicate,
    OpaqueCondition,
    PipelineType,
    SKIPPABLE,
    ScheduleStage,
    guard_fires,
    iter_leaves,
)
from .schedule_builder import (
    BreakLoop,
    ConditionalBlock,
    DomainLoop,
    Step,
    WhileTrueLoop,
    WorkTileLoop,
    _format_guard,
    _iter_nodes,
    validate_break_placement,
    while_loop_break_clauses,
)
from .resources import WorkQueue
from .task import Task


@dataclass(frozen=True)
class FlatScheduleOp:
    resource: object
    schedule_stage: ScheduleStage
    unique_id: int
    phase: str
    domain_start: object | None = None
    domain_end: object | None = None
    domain_step: object | None = None
    domain_entry: int | None = None
    sync_phase: str | None = None


def _bound(value: object, task: Task, fallback: int) -> int:
    if type(value) is int:
        return value
    if callable(value):
        try:
            folded = value(task, (0, 0, 0))
        except (AttributeError, TypeError):
            return fallback
        if type(folded) is int:
            return folded
    return fallback


def _domain_trip_count(
    node: DomainLoop,
    task: Task,
    dynamic_domain_fallback: int,
    representative_domain: bool,
) -> int:
    start = _bound(node.start, task, 0)
    end = (
        dynamic_domain_fallback
        if representative_domain
        else _bound(node.end, task, dynamic_domain_fallback)
    )
    stride = _bound(node.step, task, 1)
    if stride <= 0:
        raise ValueError("exhaustive checker requires positive domain step")
    return len(range(start, end, stride))


def _guard_phase(guard: object, phase: str) -> str:
    """Return the trace phase tag for operations inside a guarded block."""
    leaves = tuple(iter_leaves(guard))
    if any(isinstance(leaf, OpaqueCondition) for leaf in leaves):
        return "I"
    if any(leaf is FIRST_ITER for leaf in leaves):
        return "F"
    if any(leaf is LAST_ITER for leaf in leaves):
        return "L"
    if any(isinstance(leaf, Every) for leaf in leaves):
        return "P"
    return phase


OpaqueAssignment = dict[object, bool]
OpaqueAssignmentSequence = list[OpaqueAssignment]


@dataclass(frozen=True)
class OpaqueAssignmentScenario:
    """Per-task opaque assignments for one joint interleaving check.

    A task with a ``while_true_loop()`` receives an assignment sequence, one
    dictionary per modeled round. Other tasks receive ``default_assignment``,
    the final (break) round of the first while-loop task, so they observe the
    exit-round values of shared opaque keys.
    """

    default_assignment: OpaqueAssignment
    task_assignments: dict[int, OpaqueAssignmentSequence]
    task_names: dict[int, str] = field(default_factory=dict)

    def for_task(self, task: Task) -> OpaqueAssignment | OpaqueAssignmentSequence:
        return self.task_assignments.get(id(task), self.default_assignment)


# Stages that make progress in a retry round. Auxiliary work, such as an exit
# probe, does not: a round that only polls can spin forever.
_WHILE_RETRY_PROGRESS_STAGES = frozenset(
    {
        ScheduleStage.ProducerTryAcquire,
        ScheduleStage.ProducerAcquire,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ProducerWork,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerRelease,
        ScheduleStage.ConsumerWork,
    }
)


def format_opaque_assignment(assignment: object) -> str:
    """Render an assignment, a while-loop sequence, or a joint scenario."""
    if isinstance(assignment, OpaqueAssignmentScenario):
        parts = [f"default={format_opaque_assignment(assignment.default_assignment)}"]
        for task_id, sequence in assignment.task_assignments.items():
            name = assignment.task_names.get(task_id, str(task_id))
            parts.append(f"{name}: {format_opaque_assignment(sequence)}")
        return "; ".join(parts)
    if isinstance(assignment, list):
        return " -> ".join(format_opaque_assignment(item) for item in assignment)
    if not assignment:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in assignment.items())


def expand_task(
    task: Task,
    *,
    skipped_tile: bool = False,
    opaque_assignment: (
        OpaqueAssignment | OpaqueAssignmentSequence | OpaqueAssignmentScenario | None
    ) = None,
    num_tiles: int = 1,
    dynamic_domain_fallback: int = 1,
    representative_domain: bool = False,
) -> list[FlatScheduleOp]:
    """Flatten one task schedule into the operation order the checker explores.

    ``opaque_assignment`` is a dictionary, a while-loop sequence, or an
    :class:`OpaqueAssignmentScenario`. A sequence unrolls the task's
    ``while_true_loop()`` once per round; guards outside that loop observe its
    final (break) round. Init/advance hooks are not emitted: they compute the
    cursor and do not touch pipeline credits.
    """
    validate_break_placement(task.schedule.body)
    if isinstance(opaque_assignment, OpaqueAssignmentScenario):
        opaque_assignment = opaque_assignment.for_task(task)
    while_sequence = None
    if isinstance(opaque_assignment, list):
        while_sequence = opaque_assignment
        opaque_assignment = opaque_assignment[-1] if opaque_assignment else {}
    result = []
    domain_entry = itertools.count()

    def emit(nodes, assignment, iteration=None, count=None, phase="O", skipped=False):
        for node in nodes:
            if isinstance(node, BreakLoop):
                return True
            if isinstance(node, Step):
                # Static work-queue stages only update task-local scheduler
                # state; unlike a CLC queue, they do not synchronize tasks.
                if (
                    isinstance(node.memory_resource, WorkQueue)
                    and node.memory_resource.pipeline_config is None
                ):
                    continue
                result.append(
                    FlatScheduleOp(
                        node.memory_resource,
                        node.schedule_stage,
                        node.unique_id,
                        phase,
                    )
                )
            elif isinstance(node, DomainLoop):
                if skipped:
                    continue
                count = _domain_trip_count(
                    node, task, dynamic_domain_fallback, representative_domain
                )
                next(domain_entry)
                for index in range(count):
                    if emit(node.body, assignment, index, count, "", skipped):
                        break
            elif isinstance(node, WhileTrueLoop):
                if skipped:
                    continue
                emit_while(node, assignment)
            elif isinstance(node, WorkTileLoop):
                for _ in range(num_tiles):
                    emit(node.body, assignment, iteration, count, phase, skipped_tile)
            elif isinstance(node, ConditionalBlock):
                guard = node.condition
                if guard is SKIPPABLE:
                    active = not skipped
                else:
                    active = guard_fires(
                        guard,
                        iteration or 0,
                        count or 0,
                        opaque_assignment=assignment,
                    )
                if active:
                    guard_phase = _guard_phase(guard, phase)
                    if emit(node.body, assignment, iteration, count, guard_phase, skipped):
                        return True
        return False

    def emit_while(node, assignment):
        # Without a sequence, the round's assignment is the plain dictionary;
        # the structural round checks apply only to modeled sequences.
        sequence = while_sequence if while_sequence is not None else [assignment or {}]
        if not sequence:
            raise ValueError(
                f"Task {task.name!r}: while_true_loop() received an empty "
                "assignment sequence"
            )
        for index, round_assignment in enumerate(sequence):
            start = len(result)
            broke = emit(node.body, round_assignment, index, len(sequence), "")
            if while_sequence is None:
                if broke:
                    break
                continue
            where = f"while_true_loop() round {index + 1} of {len(sequence)}"
            if index == len(sequence) - 1:
                if not broke:
                    raise ValueError(
                        f"Task {task.name!r}: {where} does not reach break_loop()"
                    )
            elif broke:
                raise ValueError(
                    f"Task {task.name!r}: retry {where} reaches break_loop()"
                )
            elif not any(
                op.schedule_stage in _WHILE_RETRY_PROGRESS_STAGES
                for op in result[start:]
            ):
                raise ValueError(
                    f"Task {task.name!r}: retry {where} has no blocking or "
                    "state-changing pipeline operation"
                )

    emit(task.schedule.body, opaque_assignment)
    return result


def collect_opaque_keys(tasks: list[Task]) -> list[object]:
    """Return distinct opaque leaf keys of every guard, in first-seen order.

    Values used only as routed data are not verifier keys.
    """
    result = []
    for task in tasks:
        for node in _iter_nodes(task.schedule.body):
            if isinstance(node, ConditionalBlock):
                for leaf in iter_leaves(node.condition):
                    if isinstance(leaf, OpaqueCondition) and leaf.key not in result:
                        result.append(leaf.key)
    return result


GuardObservation = tuple[BlockGuard, int, int]


def collect_guard_observations(
    tasks: list[Task], *, representative_domain: bool = False
) -> list[GuardObservation]:
    """Collect each guard at every enclosing domain iteration it can observe.

    Each observation is ``(guard, iter_idx, num_iters)``, using the trip counts
    :func:`expand_task` models. Observations that agree on every iteration leaf
    are kept once, since they evaluate identically under any assignment.
    """
    fallback = dynamic_domain_fallback_for(tasks)
    observations = []
    seen = set()

    def observe(guard, contexts):
        iteration_leaves = [
            leaf for leaf in iter_leaves(guard) if isinstance(leaf, IterationPredicate)
        ]
        for index, count in contexts:
            signature = (
                id(guard),
                tuple(leaf.fires(index, count) for leaf in iteration_leaves),
            )
            if signature not in seen:
                seen.add(signature)
                observations.append((guard, index, count))

    def walk(task, nodes, contexts):
        for node in nodes:
            if isinstance(node, (Step, BreakLoop)):
                continue
            if isinstance(node, DomainLoop):
                count = _domain_trip_count(node, task, fallback, representative_domain)
                walk(task, node.body, tuple((index, count) for index in range(count)))
                continue
            if isinstance(node, ConditionalBlock) and node.condition is not SKIPPABLE:
                observe(node.condition, contexts)
            walk(task, node.body, contexts)

    for task in tasks:
        walk(task, task.schedule.body, ((0, 0),))
    return observations


def find_while_true_loop(task: Task) -> WhileTrueLoop | None:
    """Return the task's ``while_true_loop()``; the builder allows at most one."""
    return next(
        (
            node
            for node in _iter_nodes(task.schedule.body)
            if isinstance(node, WhileTrueLoop)
        ),
        None,
    )


def _all_opaque_assignments(
    keys: list[object],
    observations: list[GuardObservation] | None = None,
) -> list[OpaqueAssignment]:
    """Return one assignment per observably distinct control-flow class.

    Every key maps to ``True`` or ``False``. With ``observations``, two
    assignments are equivalent when every observed guard evaluates the same
    under both; the first of each class represents it. For example, a schedule
    that observes only ``a & b`` keeps two classes, not four.
    """
    assignments = [
        dict(zip(keys, values))
        for values in itertools.product((False, True), repeat=len(keys))
    ]
    if not observations:
        return assignments
    distinct = {}
    for assignment in assignments:
        signature = tuple(
            guard_fires(guard, index, count, opaque_assignment=assignment)
            for guard, index, count in observations
        )
        distinct.setdefault(signature, assignment)
    return list(distinct.values())


def _clause_holds(clause: tuple, assignment: OpaqueAssignment) -> bool:
    """An empty clause is an unconditional break and always holds."""
    return all(
        guard_fires(guard, 0, 0, opaque_assignment=assignment) for guard in clause
    )


def _fire_count_hints(loop: WhileTrueLoop) -> list[tuple[tuple, int]]:
    """Collect ``verify_num_fires`` assertions with their effective guards.

    A nested block runs only when every enclosing runtime guard holds, so its
    effective guard is the conjunction of the whole chain.
    """
    hints = []

    def walk(nodes, chain):
        for node in nodes:
            if isinstance(node, ConditionalBlock):
                inner = chain
                if node.condition is not SKIPPABLE:
                    inner = chain + (node.condition,)
                if node.verify_num_fires is not None:
                    hints.append((inner, node.verify_num_fires))
                walk(node.body, inner)

    walk(loop.body, ())
    return hints


def _while_task_sequences(
    task: Task,
    loop: WhileTrueLoop,
    keys: list[object],
    observations: list[GuardObservation] | None = None,
) -> list[OpaqueAssignmentSequence]:
    """Enumerate ``retry^(N-1) + break`` round histories for one task.

    A break assignment satisfies at least one exit clause; a retry assignment
    satisfies none. ``verify_num_fires`` assertions discard histories in which
    a block would run a different number of times.
    """
    clauses = while_loop_break_clauses(loop)
    assignments = _all_opaque_assignments(keys, observations)
    breaks = [a for a in assignments if any(_clause_holds(c, a) for c in clauses)]
    retries = [a for a in assignments if not any(_clause_holds(c, a) for c in clauses)]
    if not breaks:
        raise ValueError(
            f"Task {task.name!r}: while_true_loop() has no opaque assignment that "
            "satisfies a break_loop() clause"
        )
    hints = _fire_count_hints(loop)
    sequences = []
    for count in loop.verify_num_iters:
        if count > 1 and not retries:
            raise ValueError(
                f"Task {task.name!r}: while_true_loop() verify_num_iters={count} "
                "requires a retry assignment, but every opaque assignment "
                "satisfies a break_loop() clause"
            )
        for history in itertools.product(retries, repeat=count - 1):
            for final in breaks:
                sequence = [*history, final]
                if all(
                    sum(_clause_holds(guards, a) for a in sequence) == declared
                    for guards, declared in hints
                ):
                    sequences.append(sequence)
    if not sequences:
        widest = max(loop.verify_num_iters)
        details = []
        for guards, declared in hints:
            retry_fires = [_clause_holds(guards, a) for a in retries]
            break_fires = [_clause_holds(guards, a) for a in breaks]
            low = (widest - 1) * bool(retries and all(retry_fires)) + all(break_fires)
            high = (widest - 1) * any(retry_fires) + any(break_fires)
            reach = f"exactly {low}" if low == high else f"between {low} and {high}"
            rendered = " and ".join(_format_guard(guard) for guard in guards)
            details.append(
                f"{rendered} declares verify_num_fires={declared} but fires on "
                f"{reach} of {widest} modeled rounds"
            )
        raise ValueError(
            f"Task {task.name!r}: no while_true_loop() round history satisfies "
            f"verify_num_fires over verify_num_iters={loop.verify_num_iters}: "
            f"{'; '.join(details)}. Raise verify_num_iters to model more rounds, "
            "or correct the assertion"
        )
    return sequences


def opaque_assignment_sequences(
    tasks: list[Task],
    keys: list[object],
    *,
    representative_domain: bool = False,
) -> list[OpaqueAssignment | OpaqueAssignmentScenario]:
    """Enumerate the opaque inputs for every exhaustive interleaving check.

    Without a ``while_true_loop()`` this is one boolean assignment of ``keys``
    per observably distinct control-flow class. Otherwise each result is a
    scenario combining one round history per while-loop task.
    """
    observations = collect_guard_observations(
        tasks, representative_domain=representative_domain
    )
    while_tasks = [
        (task, loop)
        for task in tasks
        if (loop := find_while_true_loop(task)) is not None
    ]
    if not while_tasks:
        return _all_opaque_assignments(keys, observations)
    names = {id(task): task.name for task, _ in while_tasks}
    return [
        OpaqueAssignmentScenario(
            default_assignment=histories[0][-1],
            task_assignments={
                id(task): history
                for (task, _), history in zip(while_tasks, histories)
            },
            task_names=names,
        )
        for histories in itertools.product(
            *(
                _while_task_sequences(task, loop, keys, observations)
                for task, loop in while_tasks
            )
        )
    ]


def dynamic_domain_fallback_for(tasks: list[Task]) -> int:
    starts = [
        node.start
        for task in tasks
        for node in _iter_nodes(task.schedule.body)
        if isinstance(node, DomainLoop) and type(node.start) is int
    ]
    return max(starts, default=0) + 1


def _explode_sync_work(schedules: list[list[FlatScheduleOp]]) -> list[list[FlatScheduleOp]]:
    """Expand non-pipelined work into two logical synchronization phases."""
    result = []
    for schedule in schedules:
        expanded = []
        for operation in schedule:
            resource = operation.resource
            stage = operation.schedule_stage
            is_nonpipeline = not resource.is_barrier and resource.pipeline_config is None
            if is_nonpipeline and stage is ScheduleStage.ConsumerWork:
                expanded.append(replace(operation, sync_phase="cons_write_phase"))
                expanded.append(replace(operation, sync_phase="cons_read_phase"))
            elif is_nonpipeline and stage is ScheduleStage.ProducerWork:
                expanded.extend((operation, operation))
            else:
                expanded.append(operation)
        result.append(expanded)
    return result


@dataclass(slots=True)
class DeadlockInfo:
    cursors: tuple[int, ...]
    blocked_tasks: tuple[str, ...]
    blocked_details: tuple[str, ...] = ()

    def format_lines(self) -> list[str]:
        return [
            f"deadlock witness at cursors {self.cursors}: "
            f"blocked tasks {', '.join(self.blocked_tasks)}",
            *self.blocked_details,
        ]


@dataclass(slots=True)
class RaceInfo:
    cursors: tuple[int, ...]
    writer_task: str
    writer_resource: str
    victim_task: str
    victim_resource: str
    victim_access: str
    overlap_desc: str


@dataclass(slots=True)
class PdlOrderInfo:
    cursors: tuple[int, ...]
    launch_task: str
    launch_resource: str


@dataclass(slots=True)
class CheckResult:
    deadlock_states: list[DeadlockInfo]
    race_states: list[RaceInfo]
    states_explored: int
    complete_count: int
    is_safe: bool
    pdl_order_states: list[PdlOrderInfo] = field(default_factory=list)
    hit_state_limit: bool = False


@dataclass(slots=True, frozen=True)
class _TraceNode:
    parent: "_TraceNode | None"
    task_index: int
    operation: str


def _trace_items(node: _TraceNode | None) -> list[tuple[int, str]]:
    items = []
    while node is not None:
        items.append((node.task_index, node.operation))
        node = node.parent
    items.reverse()
    return items


class _VerboseReporter:
    """Render the expanded schedules and representative BFS witnesses."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks

    def print_header(
        self,
        schedules: list[list[FlatScheduleOp]],
        *,
        skipped_tile: bool,
    ) -> None:
        print("=" * 72)
        print("Exhaustive interleaving checker - BFS exploration")
        if skipped_tile:
            print("Variant: skipped-tile execution")
        print("=" * 72)
        for task_index, task in enumerate(self.tasks):
            schedule = ", ".join(
                f"{op.schedule_stage.name}({op.resource.name})"
                for op in schedules[task_index]
            )
            print(f"  Task {task_index} ({task.name}): [{schedule}]")
        print("-" * 72)

    @staticmethod
    def print_model(lines: list[str]) -> None:
        print("  barrier model:")
        for line in lines:
            print(f"    {line}")
        print("-" * 72)

    def print_timeline(
        self,
        trace: _TraceNode | None,
        *,
        suffix_label: str = "",
    ) -> None:
        path = _trace_items(trace)
        if not path:
            print("  timeline: (initial state)")
            if suffix_label:
                print(f"  {suffix_label}")
            return

        column_width = (
            max(
                max((len(operation) for _, operation in path), default=0),
                max((len(task.name) for task in self.tasks), default=0),
                5,
            )
            + 2
        )
        header = "    t  "
        separator = "    -  "
        for task in self.tasks:
            header += task.name.ljust(column_width)
            separator += ("-" * len(task.name)).ljust(column_width)
        print(f"  timeline ({len(path)} steps):")
        print(header)
        print(separator)
        for time, (active_task, operation) in enumerate(path):
            row = f"    {time:<3d}"
            for task_index in range(len(self.tasks)):
                cell = operation if task_index == active_task else "."
                row += cell.ljust(column_width)
            print(row)
        if suffix_label:
            print(f"  {suffix_label}")

    def format_held(self, held: tuple, resources: dict[int, object]) -> str:
        if not held:
            return "{}"
        entries = []
        for task_index, resource_id, access, slot, tag in held:
            entry = (
                f"{resources[resource_id].name}:{access}@"
                f"{self.tasks[task_index].name}"
            )
            if slot != _SLOTLESS:
                entry += f"[stage {slot}]"
            if tag != _UNTAGGED:
                entry += f"#commit{tag}"
            entries.append(entry)
        return "{" + ", ".join(entries) + "}"

    def print_complete(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        trace: _TraceNode | None,
    ) -> None:
        print(f"State #{state_number}  cursors={cursors}  COMPLETE")
        self.print_timeline(trace)
        print()

    def print_deadlock(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        held: tuple,
        resources: dict[int, object],
        trace: _TraceNode | None,
        blocked_tasks: tuple[str, ...],
        blocked_details: tuple[str, ...] = (),
    ) -> None:
        print(
            f"State #{state_number}  cursors={cursors}  "
            f"held={self.format_held(held, resources)}"
        )
        self.print_timeline(
            trace,
            suffix_label=f"*** DEADLOCK: blocked tasks {', '.join(blocked_tasks)}",
        )
        for detail in blocked_details:
            print(f"      {detail}")
        print()

    def print_race(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        held: tuple,
        resources: dict[int, object],
        trace: _TraceNode | None,
        race: RaceInfo,
    ) -> None:
        print(
            f"State #{state_number}  cursors={cursors}  "
            f"held={self.format_held(held, resources)}"
        )
        self.print_timeline(
            trace,
            suffix_label=(
                f"*** RACE: {race.writer_task} writes {race.writer_resource} vs "
                f"{race.victim_task} {race.victim_access} "
                f"{race.victim_resource} ({race.overlap_desc})"
            ),
        )
        print()

    @staticmethod
    def print_summary(result: CheckResult, max_states: int) -> None:
        print("=" * 72)
        print(
            f"BFS complete: {result.states_explored} states explored, "
            f"{result.complete_count} complete, "
            f"{len(result.deadlock_states)} deadlock(s), "
            f"{len(result.race_states)} race(s), "
            f"{len(result.pdl_order_states)} PDL order violation(s)"
        )
        if result.hit_state_limit:
            print(
                f"WARNING: exhaustive checker hit max_states={max_states} "
                "before completing the search; no concrete issue was found."
            )
        print(f"Result: {'SAFE' if result.is_safe else 'UNSAFE'}")
        print("=" * 72)


def _physical_ranges(resource: object) -> list[tuple[str, int, int]]:
    return getattr(resource, "physical_ranges", lambda: [])()


def _overlap(left: object, right: object) -> str | None:
    for left_space, left_start, left_end in _physical_ranges(left):
        for right_space, right_start, right_end in _physical_ranges(right):
            if left_space == right_space and max(left_start, right_start) < min(
                left_end, right_end
            ):
                return (
                    f"{left_space}[{max(left_start, right_start)}:"
                    f"{min(left_end, right_end)}]"
                )
    return None


# ── Barrier-phase model ──────────────────────────────────────────────────
#
# Every resource that a schedule acquires, commits, waits on or releases owns
# one record per ring stage: the full barrier's phase and pending arrival
# count, then the empty barrier's phase and pending arrival count.  The
# records of all resources are flattened into one tuple of ints so that a
# successor state is a single tuple copy.  Full barriers start in phase 0 and
# empty barriers in phase 1; software cursors start in phase 0, so the first
# ``num_stages`` producer acquires pass and the first consumer wait blocks.
#
# A task's k-th acquire (wait) on a resource addresses stage ``k % num_stages``
# and names phase ``(k // num_stages) % 2``; its k-th commit (release) arrives
# on the same stage.  This is the implicit cursor of a schedule without
# explicit pipeline state: the leading barrier op claims the stage and the
# trailing op advances past it.  A ``ConsumerWait`` is enabled once the stage's
# full barrier has left the phase the cursor names, a ``ProducerAcquire`` once
# the empty barrier has.  A full barrier flips after every producer task of the
# resource has committed the stage and an empty barrier after every consumer
# task has released it, which is what the hardware arrival counts encode: one
# producer feeding several consumers is a broadcast, several producers feeding
# one consumer must all arrive, and a task that commits or releases fewer
# times than its partners is a deadlock rather than a spare credit.
_B_FULL_PHASE = 0
_B_FULL_ARRIVALS = 1
_B_EMPTY_PHASE = 2
_B_EMPTY_ARRIVALS = 3
_BARRIER_FIELDS = 4
_INITIAL_RECORD = (0, 0, 1, 0)

_GATE_NONE = 0
_GATE_FULL = 1  # ConsumerWait reads the full barrier's phase
_GATE_EMPTY = 2  # ProducerAcquire reads the empty barrier's phase

_BARRIER_STAGES = frozenset(
    {
        ScheduleStage.ProducerTryAcquire,
        ScheduleStage.ProducerAcquire,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerRelease,
    }
)

# Pipeline types whose consumer is the tensor-core MMA.  The software
# ``ConsumerRelease`` is a ``tcgen05.commit`` that arrives on the empty barrier
# only when the MMA has finished reading, so the operand stays live past the
# release.  The checker drops that read hold at the next ``ProducerAcquire`` of
# the same stage (the acquire passed the empty barrier, so every MMA reading
# the stage completed) or at a paired completion wait, see
# ``_COMPLETION_PRODUCER_TYPES``.
_DMA_CONSUMER_TYPES = frozenset(
    {
        PipelineType.TmaUmma,
        PipelineType.AsyncUmma,
        PipelineType.UmmaUmma,
    }
)

# Pipeline types whose full barrier is arrived by ``tcgen05.commit``.  A
# ``ConsumerWait`` on such a resource proves that every MMA the producing task
# issued before the matching commit has completed, including its reads of the
# DMA-consumer operands it waited on earlier.
_COMPLETION_PRODUCER_TYPES = frozenset(
    {
        PipelineType.UmmaAsync,
        PipelineType.UmmaUmma,
    }
)

_UNTAGGED = -1
_SLOTLESS = -1

# A held entry is ``(task_index, resource_id, access, stage, commit_tag)``.
# ``write`` holds span ProducerAcquire..ProducerCommit, ``read`` holds span
# ConsumerWait..ConsumerRelease (or longer for DMA consumers).  ``commit_tag``
# is the generation of the paired completion commit that covers a DMA-consumer
# read, or ``_UNTAGGED``.
HeldEntry = tuple[int, int, str, int, int]


def _pipeline_type(resource: object) -> PipelineType | None:
    config = getattr(resource, "pipeline_config", None)
    return None if config is None else config.pipeline_type


def _is_dma_consumer(resource: object) -> bool:
    return _pipeline_type(resource) in _DMA_CONSUMER_TYPES


def _is_completion_producer(resource: object) -> bool:
    return _pipeline_type(resource) in _COMPLETION_PRODUCER_TYPES


def _num_stages(resource: object) -> int:
    config = getattr(resource, "pipeline_config", None)
    return 1 if config is None else config.num_stages


@dataclass(frozen=True, slots=True)
class _StepInfo:
    """Static facts about one expanded schedule op."""

    rid: int
    stage: ScheduleStage
    label: str
    slot: int = _SLOTLESS
    record: int = -1
    gate: int = _GATE_NONE
    gate_phase: int = 0
    arrive_full: bool = False
    arrive_empty: bool = False
    expected_full: int = 1
    expected_empty: int = 1
    access: str | None = None
    dma_consumer: bool = False
    completion_rids: frozenset[int] = frozenset()
    completion_gen: int = -1


@dataclass(frozen=True, slots=True)
class _BarrierModel:
    """Barrier layout, participant counts and per-op metadata for one check."""

    offsets: dict[int, int]
    initial: tuple[int, ...]
    producers: dict[int, int]
    consumers: dict[int, int]
    dma_rids: frozenset[int]
    completion_pairs: dict[int, frozenset[int]]
    resources: dict[int, object]
    steps: list[list[_StepInfo]]

    def describe(self) -> list[str]:
        lines = []
        for rid in self.offsets:
            resource = self.resources[rid]
            pipeline_type = _pipeline_type(resource)
            kind = "unpipelined" if pipeline_type is None else pipeline_type.name
            line = (
                f"{resource.name}: {kind}, {_num_stages(resource)} stage(s), "
                f"{self.producers.get(rid, 1)} producer task(s), "
                f"{self.consumers.get(rid, 1)} consumer task(s)"
            )
            if rid in self.dma_rids:
                line += "; operand read hold persists past release"
            paired = self.completion_pairs.get(rid)
            if paired:
                names = ", ".join(sorted(self.resources[x].name for x in paired))
                line += f"; a wait on it completes the reads of {names}"
            lines.append(line)
        return lines


def _static_roles(
    task: Task,
) -> tuple[set[int], set[int], set[int], dict[int, object]]:
    """Return the resources a task commits, releases and waits on, statically.

    Hardware arrival counts are fixed when the barriers are initialized, so a
    task counts as a participant whenever its captured schedule contains the
    op, whether or not the current opaque assignment or tile variant reaches
    it.
    """
    committed: set[int] = set()
    released: set[int] = set()
    waited: set[int] = set()
    by_id: dict[int, object] = {}
    for node in _iter_nodes(task.schedule.body):
        if not isinstance(node, Step):
            continue
        rid = id(node.memory_resource)
        by_id[rid] = node.memory_resource
        stage = node.schedule_stage
        if stage is ScheduleStage.ProducerCommit:
            committed.add(rid)
        elif stage is ScheduleStage.ConsumerRelease:
            released.add(rid)
        elif stage is ScheduleStage.ConsumerWait:
            waited.add(rid)
    return committed, released, waited, by_id


def _build_model(
    tasks: list[Task],
    schedules: list[list[FlatScheduleOp]],
    resources: dict[int, object],
) -> _BarrierModel:
    offsets: dict[int, int] = {}
    initial: list[int] = []
    for schedule in schedules:
        for op in schedule:
            rid = id(op.resource)
            if op.schedule_stage in _BARRIER_STAGES and rid not in offsets:
                offsets[rid] = len(initial) // _BARRIER_FIELDS
                initial.extend(_INITIAL_RECORD * _num_stages(op.resource))

    producers: dict[int, int] = {}
    consumers: dict[int, int] = {}
    roles = []
    by_id = dict(resources)
    for task in tasks:
        committed, released, waited, task_resources = _static_roles(task)
        by_id.update(task_resources)
        for rid in committed:
            producers[rid] = producers.get(rid, 0) + 1
        for rid in released:
            consumers[rid] = consumers.get(rid, 0) + 1
        roles.append((committed, waited))

    dma_rids = frozenset(
        rid for rid, resource in by_id.items() if _is_dma_consumer(resource)
    )
    # A task that waits on a DMA-consumer operand and commits a completion
    # producer pairs the two: the commit covers the reads issued before it.
    pairs: dict[int, set[int]] = {}
    for committed, waited in roles:
        operands = waited & dma_rids
        if not operands:
            continue
        for rid in committed:
            if _is_completion_producer(by_id[rid]):
                pairs.setdefault(rid, set()).update(operands)
    completion_pairs = {rid: frozenset(operands) for rid, operands in pairs.items()}

    steps = [
        _precompute_steps(
            schedule, offsets, producers, consumers, dma_rids, completion_pairs
        )
        for schedule in schedules
    ]
    return _BarrierModel(
        offsets,
        tuple(initial),
        producers,
        consumers,
        dma_rids,
        completion_pairs,
        by_id,
        steps,
    )


def _precompute_steps(
    schedule: list[FlatScheduleOp],
    offsets: dict[int, int],
    producers: dict[int, int],
    consumers: dict[int, int],
    dma_rids: frozenset[int],
    completion_pairs: dict[int, frozenset[int]],
) -> list[_StepInfo]:
    """Resolve the stage, gate, arrivals and holds of every op of one task."""
    ordinals: dict[tuple[int, str], int] = {}
    steps = []
    for op in schedule:
        resource = op.resource
        rid = id(resource)
        stage = op.schedule_stage
        fields: dict[str, object] = {
            "rid": rid,
            "stage": stage,
            "label": f"{stage.name}({resource.name})",
            "dma_consumer": rid in dma_rids,
        }
        if stage is ScheduleStage.ProducerWork:
            fields["access"] = "write"
        elif stage is ScheduleStage.ConsumerWork:
            fields["access"] = (
                "write" if op.sync_phase == "cons_write_phase" else "read"
            )
        base = offsets.get(rid)
        if base is not None and stage in _BARRIER_STAGES:
            num_stages = _num_stages(resource)
            fields["expected_full"] = producers.get(rid, 1)
            fields["expected_empty"] = consumers.get(rid, 1)
            if stage in (
                ScheduleStage.ProducerTryAcquire,
                ScheduleStage.ProducerAcquire,
            ):
                ordinal = ordinals.get((rid, "acquire"), 0)
                fields["slot"] = ordinal % num_stages
                if stage is ScheduleStage.ProducerAcquire:
                    fields["gate"] = _GATE_EMPTY
                    fields["gate_phase"] = (ordinal // num_stages) % 2
                    ordinals[(rid, "acquire")] = ordinal + 1
            elif stage is ScheduleStage.ProducerCommit:
                ordinal = ordinals.get((rid, "commit"), 0)
                ordinals[(rid, "commit")] = ordinal + 1
                fields["slot"] = ordinal % num_stages
                fields["arrive_full"] = True
                paired = completion_pairs.get(rid)
                if paired:
                    gen = ordinals.get((rid, "completion commit"), 0)
                    ordinals[(rid, "completion commit")] = gen + 1
                    fields["completion_rids"] = paired
                    fields["completion_gen"] = gen
            elif stage in (
                ScheduleStage.ConsumerTryWait,
                ScheduleStage.ConsumerWait,
            ):
                ordinal = ordinals.get((rid, "wait"), 0)
                fields["slot"] = ordinal % num_stages
                if stage is ScheduleStage.ConsumerWait:
                    fields["gate"] = _GATE_FULL
                    fields["gate_phase"] = (ordinal // num_stages) % 2
                    ordinals[(rid, "wait")] = ordinal + 1
                    paired = completion_pairs.get(rid)
                    if paired:
                        gen = ordinals.get((rid, "completion wait"), 0)
                        ordinals[(rid, "completion wait")] = gen + 1
                        fields["completion_rids"] = paired
                        fields["completion_gen"] = gen
            else:
                ordinal = ordinals.get((rid, "release"), 0)
                ordinals[(rid, "release")] = ordinal + 1
                fields["slot"] = ordinal % num_stages
                fields["arrive_empty"] = True
            fields["record"] = base + fields["slot"]
        steps.append(_StepInfo(**fields))
    return steps


def _step_enabled(step: _StepInfo, barriers: tuple[int, ...]) -> bool:
    if step.gate == _GATE_NONE:
        return True
    phase_field = _B_FULL_PHASE if step.gate == _GATE_FULL else _B_EMPTY_PHASE
    return barriers[step.record * _BARRIER_FIELDS + phase_field] != step.gate_phase


def _arrive(
    values: list[int], base: int, phase_field: int, count_field: int, expected: int
) -> None:
    """Count one arrival on a barrier; flip its phase once all have arrived."""
    values[base + count_field] += 1
    if values[base + count_field] >= expected:
        values[base + count_field] = 0
        values[base + phase_field] ^= 1


def _remove_first(
    entries: list[HeldEntry], task_index: int, rid: int, access: str, slot: int
) -> None:
    for index, entry in enumerate(entries):
        if entry[:4] == (task_index, rid, access, slot):
            del entries[index]
            return


def _apply_step(
    step: _StepInfo,
    task_index: int,
    barriers: tuple[int, ...],
    held: tuple[HeldEntry, ...],
) -> tuple[tuple[int, ...], tuple[HeldEntry, ...]]:
    """Return the barrier vector and held entries after ``step`` executes."""
    if step.arrive_full or step.arrive_empty:
        values = list(barriers)
        base = step.record * _BARRIER_FIELDS
        if step.arrive_full:
            _arrive(values, base, _B_FULL_PHASE, _B_FULL_ARRIVALS, step.expected_full)
        if step.arrive_empty:
            _arrive(
                values, base, _B_EMPTY_PHASE, _B_EMPTY_ARRIVALS, step.expected_empty
            )
        barriers = tuple(values)
    if step.record < 0:
        return barriers, held
    stage = step.stage
    if stage is ScheduleStage.ProducerAcquire:
        # Passing the empty barrier of a DMA-consumer stage proves that every
        # MMA reading it has completed, whichever task issued the read.
        entries = [
            entry
            for entry in held
            if not (
                step.dma_consumer
                and entry[1] == step.rid
                and entry[2] == "read"
                and entry[3] == step.slot
            )
        ]
        entries.append((task_index, step.rid, "write", step.slot, _UNTAGGED))
    elif stage is ScheduleStage.ProducerCommit:
        entries = list(held)
        _remove_first(entries, task_index, step.rid, "write", step.slot)
        if step.completion_rids:
            # ``tcgen05.commit`` tracks the issuing thread's own MMAs, so only
            # this task's untagged operand reads are covered by this commit.
            entries = [
                (entry[0], entry[1], entry[2], entry[3], step.completion_gen)
                if entry[0] == task_index
                and entry[2] == "read"
                and entry[4] == _UNTAGGED
                and entry[1] in step.completion_rids
                else entry
                for entry in entries
            ]
    elif stage is ScheduleStage.ConsumerWait:
        entries = list(held)
        if step.completion_rids:
            entries = [
                entry
                for entry in entries
                if not (
                    entry[2] == "read"
                    and entry[4] == step.completion_gen
                    and entry[1] in step.completion_rids
                )
            ]
        entries.append((task_index, step.rid, "read", step.slot, _UNTAGGED))
    elif stage is ScheduleStage.ConsumerRelease:
        if step.dma_consumer:
            return barriers, held
        entries = list(held)
        _remove_first(entries, task_index, step.rid, "read", step.slot)
    else:
        return barriers, held
    return barriers, tuple(sorted(entries))


def _blocked_details(
    cursors: tuple[int, ...],
    barriers: tuple[int, ...],
    model: _BarrierModel,
    tasks: list[Task],
) -> tuple[str, ...]:
    details = []
    for task_index, cursor in enumerate(cursors):
        steps = model.steps[task_index]
        if cursor >= len(steps):
            continue
        step = steps[cursor]
        if step.gate == _GATE_NONE:
            continue
        base = step.record * _BARRIER_FIELDS
        if step.gate == _GATE_FULL:
            kind = "full"
            phase = barriers[base + _B_FULL_PHASE]
            arrivals = barriers[base + _B_FULL_ARRIVALS]
            expected = step.expected_full
        else:
            kind = "empty"
            phase = barriers[base + _B_EMPTY_PHASE]
            arrivals = barriers[base + _B_EMPTY_ARRIVALS]
            expected = step.expected_empty
        details.append(
            f"{tasks[task_index].name}: {step.label} on stage {step.slot} waits "
            f"for the {kind} barrier to leave phase {phase} "
            f"({arrivals} of {expected} arrivals)"
        )
    return tuple(details)


def check_all_interleavings(
    tasks: list[Task],
    alias_map=None,
    prod_alias_map=None,
    cons_alias_map=None,
    overlap_descs=None,
    max_states: int = 1_000_000,
    num_tiles: int = 1,
    skipped_tile: bool = False,
    assume_pdl_wait_completed: bool = False,
    verbose: bool = False,
    opaque_assignment: (
        OpaqueAssignment | OpaqueAssignmentSequence | OpaqueAssignmentScenario | None
    ) = None,
    early_exit: bool = True,
    cursor_only_visited: bool = True,
    representative_domain: bool = False,
) -> CheckResult:
    """Explore every interleaving of the expanded schedules.

    Shared state is the barrier-phase model described above plus the held
    accesses used for alias races.  Both are functions of the cursor tuple
    (arrival counts are sums over task prefixes and holds are decided by which
    waits, commits and acquires each task has executed), so the default
    cursor-only visited key is exact; ``cursor_only_visited=False`` spells the
    shared state out, which lets a test check that claim.
    """
    del alias_map, prod_alias_map, cons_alias_map, overlap_descs
    del assume_pdl_wait_completed
    if not tasks:
        result = CheckResult([], [], 0, 1, True)
        if verbose:
            _VerboseReporter.print_summary(result, max_states)
        return result
    fallback = dynamic_domain_fallback_for(tasks)
    schedules = [
        expand_task(
            task,
            skipped_tile=skipped_tile,
            opaque_assignment=opaque_assignment,
            num_tiles=num_tiles,
            dynamic_domain_fallback=fallback,
            representative_domain=representative_domain,
        )
        for task in tasks
    ]
    schedules = _explode_sync_work(schedules)
    resources = {
        id(op.resource): op.resource for schedule in schedules for op in schedule
    }
    model = _build_model(tasks, schedules, resources)
    reporter = _VerboseReporter(tasks) if verbose else None
    if reporter is not None:
        reporter.print_header(schedules, skipped_tile=skipped_tile)
        reporter.print_model(model.describe())
    queue = deque([((0,) * len(tasks), model.initial, (), None)])
    visited = set()
    deadlocks = []
    races = []
    explored = complete = 0

    while queue and explored < max_states:
        cursors, barriers, held, trace = queue.popleft()
        key = cursors if cursor_only_visited else (cursors, barriers, held)
        if key in visited:
            continue
        visited.add(key)
        explored += 1
        if all(cursors[i] == len(schedules[i]) for i in range(len(tasks))):
            complete += 1
            if reporter is not None:
                reporter.print_complete(explored, cursors, trace)
            continue
        enabled = 0
        for task_index, task in enumerate(tasks):
            cursor = cursors[task_index]
            if cursor >= len(schedules[task_index]):
                continue
            step = model.steps[task_index][cursor]
            if not _step_enabled(step, barriers):
                continue
            enabled += 1
            if step.access:
                for owner, held_rid, held_access, _slot, _tag in held:
                    if (
                        owner == task_index
                        or step.rid == held_rid
                        or (step.access == held_access == "read")
                    ):
                        continue
                    description = _overlap(resources[step.rid], resources[held_rid])
                    if description:
                        is_writer = step.access == "write"
                        races.append(
                            RaceInfo(
                                cursors,
                                task.name if is_writer else tasks[owner].name,
                                resources[step.rid].name
                                if is_writer
                                else resources[held_rid].name,
                                tasks[owner].name if is_writer else task.name,
                                resources[held_rid].name
                                if is_writer
                                else resources[step.rid].name,
                                held_access,
                                description,
                            )
                        )
                        if early_exit:
                            result = CheckResult(
                                deadlocks, races, explored, complete, False
                            )
                            if reporter is not None:
                                reporter.print_race(
                                    explored,
                                    cursors,
                                    held,
                                    resources,
                                    trace,
                                    races[-1],
                                )
                                reporter.print_summary(result, max_states)
                            return result
            next_barriers, next_held = _apply_step(step, task_index, barriers, held)
            next_cursors = list(cursors)
            next_cursors[task_index] += 1
            next_trace = (
                _TraceNode(trace, task_index, step.label) if verbose else None
            )
            queue.append((tuple(next_cursors), next_barriers, next_held, next_trace))
        if enabled == 0:
            blocked = tuple(
                tasks[i].name
                for i, cursor in enumerate(cursors)
                if cursor < len(schedules[i])
            )
            details = _blocked_details(cursors, barriers, model, tasks)
            deadlocks.append(DeadlockInfo(cursors, blocked, details))
            if reporter is not None:
                reporter.print_deadlock(
                    explored,
                    cursors,
                    held,
                    resources,
                    trace,
                    blocked,
                    details,
                )
            if early_exit:
                break
    safe = not deadlocks and not races and bool(complete)
    result = CheckResult(
        deadlocks,
        races,
        explored,
        complete,
        safe,
        hit_state_limit=bool(queue and explored >= max_states),
    )
    if reporter is not None:
        reporter.print_summary(result, max_states)
    return result
