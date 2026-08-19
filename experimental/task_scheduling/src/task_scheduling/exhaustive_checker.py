# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded exhaustive interleaving checker for captured schedules."""

import itertools
from collections import deque
from dataclasses import dataclass, field, replace

from .enums import (
    FIRST_ITER,
    LAST_ITER,
    Every,
    OpaqueCondition,
    SKIPPABLE,
    ScheduleStage,
    guard_fires,
)
from .schedule_builder import (
    ConditionalBlock,
    DomainLoop,
    Step,
    WorkTileLoop,
    _iter_nodes,
)
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
        folded = value(task, (0, 0, 0))
        if type(folded) is int:
            return folded
    return fallback


def expand_task(
    task: Task,
    *,
    skipped_tile: bool = False,
    opaque_assignment: dict | None = None,
    num_tiles: int = 1,
    dynamic_domain_fallback: int = 1,
    representative_domain: bool = False,
) -> list[FlatScheduleOp]:
    result = []
    domain_entry = itertools.count()

    def emit(nodes, iteration=None, count=None, phase="O", skipped=False):
        for node in nodes:
            if isinstance(node, Step):
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
                start = _bound(node.start, task, 0)
                end = (
                    dynamic_domain_fallback
                    if representative_domain
                    else _bound(node.end, task, dynamic_domain_fallback)
                )
                stride = _bound(node.step, task, 1)
                if stride <= 0:
                    raise ValueError("exhaustive checker requires positive domain step")
                values = tuple(range(start, end, stride)) or (start,)
                next(domain_entry)
                for index, _ in enumerate(values):
                    emit(node.body, index, len(values), "", skipped)
            elif isinstance(node, WorkTileLoop):
                for _ in range(num_tiles):
                    emit(node.body, iteration, count, phase, skipped_tile)
            elif isinstance(node, ConditionalBlock):
                guard = node.condition
                if guard is SKIPPABLE:
                    active = not skipped
                else:
                    active = guard_fires(
                        guard,
                        iteration or 0,
                        count or 0,
                        opaque_assignment=opaque_assignment,
                    )
                if active:
                    guard_phase = (
                        "P"
                        if isinstance(guard, Every)
                        else "I"
                        if isinstance(guard, OpaqueCondition)
                        else "F"
                        if guard is FIRST_ITER
                        else "L"
                        if guard is LAST_ITER
                        else phase
                    )
                    emit(node.body, iteration, count, guard_phase, skipped)

    emit(task.schedule.body)
    return result


def collect_opaque_keys(tasks: list[Task]) -> list[object]:
    result = []
    for task in tasks:
        for node in _iter_nodes(task.schedule.body):
            if (
                isinstance(node, ConditionalBlock)
                and isinstance(node.condition, OpaqueCondition)
                and node.condition.key not in result
            ):
                result.append(node.condition.key)
    return result


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

    def format_lines(self) -> list[str]:
        return [
            f"deadlock witness at cursors {self.cursors}: "
            f"blocked tasks {', '.join(self.blocked_tasks)}"
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
        for task_index, resource_id, access in held:
            entries.append(
                f"{resources[resource_id].name}:{access}@"
                f"{self.tasks[task_index].name}"
            )
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
    ) -> None:
        print(
            f"State #{state_number}  cursors={cursors}  "
            f"held={self.format_held(held, resources)}"
        )
        self.print_timeline(
            trace,
            suffix_label=f"*** DEADLOCK: blocked tasks {', '.join(blocked_tasks)}",
        )
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
    opaque_assignment: dict | None = None,
    early_exit: bool = True,
    cursor_only_visited: bool = True,
    representative_domain: bool = False,
) -> CheckResult:
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
    reporter = _VerboseReporter(tasks) if verbose else None
    if reporter is not None:
        reporter.print_header(schedules, skipped_tile=skipped_tile)
    initial_prod = tuple(
        (
            id(resource),
            resource.pipeline_config.num_stages
            if resource.pipeline_config is not None
            else 1,
        )
        for resource in resources.values()
    )
    initial_cons = tuple((rid, 0) for rid in resources)
    queue = deque([((0,) * len(tasks), initial_prod, initial_cons, (), None)])
    visited = set()
    deadlocks = []
    races = []
    explored = complete = 0

    while queue and explored < max_states:
        cursors, frozen_prod, frozen_cons, held, trace = queue.popleft()
        key = (
            cursors
            if cursor_only_visited
            else (cursors, frozen_prod, frozen_cons, held)
        )
        if key in visited:
            continue
        visited.add(key)
        explored += 1
        if all(cursors[i] == len(schedules[i]) for i in range(len(tasks))):
            complete += 1
            if reporter is not None:
                reporter.print_complete(explored, cursors, trace)
            continue
        prod, cons = dict(frozen_prod), dict(frozen_cons)
        enabled = 0
        for task_index, task in enumerate(tasks):
            cursor = cursors[task_index]
            if cursor >= len(schedules[task_index]):
                continue
            op = schedules[task_index][cursor]
            rid = id(op.resource)
            stage = op.schedule_stage
            if stage is ScheduleStage.ProducerAcquire and prod[rid] <= 0:
                continue
            if stage is ScheduleStage.ConsumerWait and cons[rid] <= 0:
                continue
            enabled += 1
            next_prod, next_cons = dict(prod), dict(cons)
            next_held = list(held)
            if stage is ScheduleStage.ProducerAcquire:
                next_prod[rid] -= 1
                next_held.append((task_index, rid, "write"))
            elif stage is ScheduleStage.ProducerCommit:
                next_cons[rid] += 1
                next_held = [h for h in next_held if h != (task_index, rid, "write")]
            elif stage is ScheduleStage.ConsumerWait:
                next_cons[rid] -= 1
                next_held.append((task_index, rid, "read"))
            elif stage is ScheduleStage.ConsumerRelease:
                next_prod[rid] += 1
                next_held = [h for h in next_held if h != (task_index, rid, "read")]
            access = None
            if stage is ScheduleStage.ProducerWork:
                access = "write"
            elif stage is ScheduleStage.ConsumerWork:
                access = (
                    "write"
                    if op.sync_phase == "cons_write_phase"
                    else "read"
                )
            if access:
                for owner, held_rid, held_access in held:
                    if (
                        owner == task_index
                        or rid == held_rid
                        or (access == held_access == "read")
                    ):
                        continue
                    description = _overlap(op.resource, resources[held_rid])
                    if description:
                        races.append(
                            RaceInfo(
                                cursors,
                                task.name if access == "write" else tasks[owner].name,
                                op.resource.name
                                if access == "write"
                                else resources[held_rid].name,
                                tasks[owner].name if access == "write" else task.name,
                                resources[held_rid].name
                                if access == "write"
                                else op.resource.name,
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
            next_cursors = list(cursors)
            next_cursors[task_index] += 1
            next_trace = (
                _TraceNode(
                    trace,
                    task_index,
                    f"{stage.name}({op.resource.name})",
                )
                if verbose
                else None
            )
            queue.append(
                (
                    tuple(next_cursors),
                    tuple(sorted(next_prod.items())),
                    tuple(sorted(next_cons.items())),
                    tuple(sorted(next_held)),
                    next_trace,
                )
            )
        if enabled == 0:
            blocked = tuple(
                tasks[i].name
                for i, cursor in enumerate(cursors)
                if cursor < len(schedules[i])
            )
            deadlocks.append(DeadlockInfo(cursors, blocked))
            if reporter is not None:
                reporter.print_deadlock(
                    explored,
                    cursors,
                    held,
                    resources,
                    trace,
                    blocked,
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
