# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only resource-graph, schedule, and budget validation."""

import io
import itertools
import warnings
from contextlib import redirect_stdout

from .enums import ScheduleStage
from .exhaustive_checker import (
    check_all_interleavings,
    collect_opaque_keys,
    dynamic_domain_fallback_for,
    expand_task,
)
from .ir import freeze_program_ir
from .memory import BarrierAllocator, SmemAllocator, TmemAllocator
from .pipeline import DevicePipelineBinding
from .schedule_builder import Schedule, _iter_steps
from .task import (
    DeviceTask,
    DeviceTaskManager,
    Task,
    barrier_arena_factory_for,
    smem_base_factory_for,
)


class TaskManager:
    _DEFAULT_SMEM_CAPACITY_BYTES = 232448
    _DEFAULT_TMEM_CAPACITY_COLUMNS = 512
    _REGISTER_CAPACITY = 65536

    def __init__(
        self,
        tasks: list[Task],
        resource_dependency_graph: dict[object, list[object]] | None = None,
        *,
        skip_validation: bool = False,
        smem_allocator=None,
        tmem_allocator=None,
        barrier_allocator=None,
        verbose: bool = True,
        smem_capacity_bytes: int | None = None,
        tmem_capacity_columns: int | None = None,
        exhaustive_deadlock_race_check: bool = True,
        exhaustive_representative_domain: bool = False,
        cta_warps: int | None = None,
        infer_allocators: bool = True,
        **kwargs: object,
    ) -> None:
        del kwargs
        if not tasks:
            raise ValueError("TaskManager requires at least one task")
        if any(not isinstance(task, Task) for task in tasks):
            raise TypeError("TaskManager tasks must all be Task instances")
        self.user_tasks = tuple(tasks)
        minimum_cta_warps = max(task.warp_end for task in tasks)
        self.cta_warps = minimum_cta_warps if cta_warps is None else cta_warps
        if type(self.cta_warps) is not int or self.cta_warps <= 0:
            raise ValueError("cta_warps must be a positive integer")
        if self.cta_warps < minimum_cta_warps:
            raise ValueError(
                f"cta_warps={self.cta_warps} does not cover task warp end "
                f"{minimum_cta_warps}"
            )
        self.tasks = [*tasks, *self._make_padding_tasks(tasks)]
        self.resource_dependency_graph = (
            {} if resource_dependency_graph is None else resource_dependency_graph
        )
        self.skip_validation = skip_validation
        self.smem_allocator = smem_allocator
        self.tmem_allocator = tmem_allocator
        self.barrier_allocator = barrier_allocator
        self.verbose = verbose
        self.smem_capacity_bytes = (
            smem_capacity_bytes
            if smem_capacity_bytes is not None
            else self._DEFAULT_SMEM_CAPACITY_BYTES
        )
        self.tmem_capacity_columns = (
            tmem_capacity_columns
            if tmem_capacity_columns is not None
            else self._DEFAULT_TMEM_CAPACITY_COLUMNS
        )
        self.exhaustive_deadlock_race_check = exhaustive_deadlock_race_check
        self.exhaustive_representative_domain = exhaustive_representative_domain
        self._interleaving_results = []
        seen = set()
        self.resources = []
        for task in tasks:
            for resource in task.resources:
                if id(resource) not in seen:
                    seen.add(id(resource))
                    self.resources.append(resource)
        self._validate_resource_names()
        if infer_allocators:
            self._derive_allocators()
        self.default_task_registers = self._default_task_registers()
        for task in self.tasks:
            task.default_num_registers = self.default_task_registers
        self.program_ir = freeze_program_ir(
            self.tasks,
            self.resources,
            self.resource_dependency_graph,
            cta_warps=self.cta_warps,
            default_num_registers=self.default_task_registers,
        )
        self._validate(exhaustive_deadlock_race_check)

    def _make_padding_tasks(self, tasks: list[Task]) -> list[Task]:
        occupied = {
            warp
            for task in tasks
            for warp in range(task.warp_start, task.warp_end)
        }
        padding_registers = min(
            (
                task.num_registers
                for task in tasks
                if task.num_registers is not None
            ),
            default=40,
        )
        result = []
        start = None
        for warp in range(self.cta_warps + 1):
            free = warp < self.cta_warps and warp not in occupied
            if free and start is None:
                start = warp
            if not free and start is not None:
                index = len(result)
                result.append(
                    Task(
                        start,
                        warp - start,
                        schedule=Schedule(
                            name=f"__padding_{index}", body=(), resources=()
                        ),
                        num_registers=padding_registers,
                        name=f"PaddingTask{index}",
                    )
                )
                start = None
        return result

    def _validate_resource_names(self) -> None:
        names = {}
        for resource in self.resources:
            if not resource.name:
                raise ValueError("TaskManager resources require non-empty names")
            if resource.name in names and names[resource.name] is not resource:
                raise ValueError(
                    f"duplicate TaskManager resource name {resource.name!r}"
                )
            names[resource.name] = resource

    def _derive_allocators(self) -> None:
        if self.smem_allocator is None and any(
            resource.get_smem_requirements() for resource in self.resources
        ):
            self.smem_allocator = SmemAllocator(default_add_barriers=False)
            for resource in self.resources:
                self.smem_allocator.add_resource(resource, add_barriers=False)
            self.smem_allocator.compute_layout()
        if self.tmem_allocator is None and any(
            resource.get_tmem_requirements() for resource in self.resources
        ):
            self.tmem_allocator = TmemAllocator()
            for resource in self.resources:
                self.tmem_allocator.add_resource(resource)
            self.tmem_allocator.compute_layout()
        pipeline_resources = [
            resource
            for resource in self.resources
            if resource.pipeline_config is not None
        ]
        if self.barrier_allocator is None and len(pipeline_resources) > 1:
            self.barrier_allocator = BarrierAllocator()
            for resource in pipeline_resources:
                self.barrier_allocator.add_resource(resource)
            self.barrier_allocator.compute_layout()

    def freeze(self):
        """Return an immutable IR snapshot of the validated manager state."""
        return self.program_ir

    def _run_check(self, function) -> None:
        try:
            function()
        except (TypeError, ValueError, RuntimeError) as error:
            if not self.skip_validation:
                raise
            warnings.warn(str(error), stacklevel=3)

    def _validate(self, exhaustive: bool) -> None:
        for check in (
            self._verify_warp_ranges,
            self._verify_register_budget,
            self._verify_resource_coverage,
            self._verify_dependency_graph,
            self._verify_bracketing,
            self._verify_memory_budgets,
        ):
            self._run_check(check)
        if exhaustive:
            self._run_check(self._verify_all_interleavings)

    def _verify_warp_ranges(self) -> None:
        occupied = {}
        for task in self.tasks:
            for warp in range(task.warp_start, task.warp_end):
                if warp in occupied:
                    raise ValueError(
                        f"task warp ranges overlap: {task.name!r} and "
                        f"{occupied[warp]!r} both own warp {warp}"
                    )
                occupied[warp] = task.name

    def _default_task_registers(self) -> int:
        """Compute the CTA's warp-group-rounded per-thread register budget."""
        threads_per_warp = 32
        warp_group_size = 4
        setmaxnreg_granularity = 8
        total_warps = self.cta_warps
        warp_groups = (total_warps + warp_group_size - 1) // warp_group_size
        rounded_warps = warp_groups * warp_group_size
        return min(
            256,
            self._REGISTER_CAPACITY
            // (rounded_warps * threads_per_warp * setmaxnreg_granularity)
            * setmaxnreg_granularity,
        )

    def _verify_register_budget(self) -> None:
        total = 0
        for task in self.tasks:
            registers = task.num_registers or task.default_num_registers
            total += registers * task.num_warps * 32
        if total > self._REGISTER_CAPACITY:
            raise ValueError(
                f"register budget exceeded: {total} registers requested, "
                f"capacity {self._REGISTER_CAPACITY}"
            )

    def _verify_resource_coverage(self) -> None:
        managed = {id(resource) for resource in self.resources}
        graph = {
            id(resource)
            for downstream, upstreams in self.resource_dependency_graph.items()
            for resource in (downstream, *upstreams)
        }
        missing = graph - managed
        if missing:
            raise ValueError("dependency graph references resources not owned by tasks")
        for task in self.tasks:
            declared = {id(resource) for resource in task.resources}
            used = {id(step.memory_resource) for step in _iter_steps(task.schedule)}
            if not used <= declared:
                raise ValueError(
                    f"task {task.name!r} schedule uses undeclared resources"
                )

    def _verify_dependency_graph(self) -> None:
        for downstream, upstreams in self.resource_dependency_graph.items():
            for upstream in upstreams:
                if not any(
                    any(item is upstream for item in task.src_resources)
                    and any(item is downstream for item in task.dst_resources)
                    for task in self.tasks
                ):
                    raise ValueError(
                        f"dependency edge {upstream.name} --> {downstream.name} "
                        "is not backed by a task"
                    )

    def _verify_bracketing(self) -> None:
        producer_open = {
            ScheduleStage.ProducerAcquire: False,
            ScheduleStage.ConsumerWait: False,
        }
        for task in self.tasks:
            state = {}
            tried = set()
            for step in _iter_steps(task.schedule):
                rid = id(step.memory_resource)
                stage = step.schedule_stage
                if stage is ScheduleStage.ProducerTryAcquire:
                    tried.add((rid, "producer"))
                elif stage is ScheduleStage.ConsumerTryWait:
                    tried.add((rid, "consumer"))
                elif stage is ScheduleStage.ProducerAcquire:
                    config = step.memory_resource.pipeline_config
                    if (
                        config is not None
                        and config.supports_try_probe_ops
                        and (rid, "producer") not in tried
                    ):
                        raise ValueError(
                            "ProducerAcquire must be preceded by "
                            "ProducerTryAcquire on the same resource"
                        )
                    tried.discard((rid, "producer"))
                    if state.get((rid, "producer")):
                        raise ValueError("nested ProducerAcquire without commit")
                    state[(rid, "producer")] = True
                elif stage is ScheduleStage.ProducerWork:
                    if step.memory_resource.pipeline_config and not state.get(
                        (rid, "producer")
                    ):
                        raise ValueError(
                            f"{task.name}: ProducerWork on {step.memory_resource.name} "
                            "is not bracketed by acquire/commit"
                        )
                elif stage is ScheduleStage.ProducerCommit:
                    if not state.pop((rid, "producer"), False):
                        raise ValueError("ProducerCommit without ProducerAcquire")
                elif stage is ScheduleStage.ConsumerWait:
                    config = step.memory_resource.pipeline_config
                    if (
                        config is not None
                        and config.supports_try_probe_ops
                        and (rid, "consumer") not in tried
                    ):
                        raise ValueError(
                            "ConsumerWait must be preceded by ConsumerTryWait "
                            "on the same resource"
                        )
                    tried.discard((rid, "consumer"))
                    if state.get((rid, "consumer")):
                        raise ValueError("nested ConsumerWait without release")
                    state[(rid, "consumer")] = True
                elif stage is ScheduleStage.ConsumerWork:
                    if step.memory_resource.pipeline_config and not state.get(
                        (rid, "consumer")
                    ):
                        raise ValueError(
                            f"{task.name}: ConsumerWork on {step.memory_resource.name} "
                            "is not bracketed by wait/release"
                        )
                elif stage is ScheduleStage.ConsumerRelease:
                    if not state.pop((rid, "consumer"), False):
                        raise ValueError("ConsumerRelease without ConsumerWait")
            if state:
                raise ValueError(f"{task.name}: unterminated pipeline work group")
        del producer_open

    def _verify_memory_budgets(self) -> None:
        if self.smem_allocator is not None:
            if not self.smem_allocator.layout_computed:
                raise ValueError("SMEM allocator layout has not been computed")
        if (
            self.barrier_allocator is not None
            and not self.barrier_allocator.layout_computed
        ):
            raise ValueError("barrier allocator layout has not been computed")
        data_bytes, barrier_bytes = self._device_smem_usage()
        total_smem_bytes = data_bytes + barrier_bytes
        if total_smem_bytes > self.smem_capacity_bytes:
            raise ValueError(
                f"SMEM budget exceeded: {total_smem_bytes} "
                f"> {self.smem_capacity_bytes}"
            )
        if self.tmem_allocator is not None:
            if not self.tmem_allocator.layout_computed:
                raise ValueError("TMEM allocator layout has not been computed")
            if self.tmem_allocator.total_tmem_columns > self.tmem_capacity_columns:
                raise ValueError(
                    f"TMEM budget exceeded: {self.tmem_allocator.total_tmem_columns} "
                    f"> {self.tmem_capacity_columns}"
                )

    def _device_smem_usage(self) -> tuple[int, int]:
        """Return physical data and barrier bytes materialized by the manager."""
        data_bytes = (
            self.smem_allocator.data_smem_bytes
            if self.smem_allocator is not None
            else 0
        )
        if self.barrier_allocator is not None:
            barrier_bytes = self.barrier_allocator.padded_size_bytes
        else:
            barrier_bytes = sum(
                16 * resource.pipeline_config.num_stages
                for resource in self.resources
                if resource.pipeline_config is not None
            )
        return data_bytes, barrier_bytes

    def _verify_all_interleavings(self) -> None:
        self._interleaving_results.clear()
        keys = collect_opaque_keys(self.tasks)
        assignments = (
            [
                dict(zip(keys, values))
                for values in itertools.product((False, True), repeat=len(keys))
            ]
            if keys
            else [{}]
        )
        for assignment in assignments:
            verbose_output = ""
            if self.verbose:
                stream = io.StringIO()
                with redirect_stdout(stream):
                    result = check_all_interleavings(
                        self.tasks,
                        opaque_assignment=assignment,
                        early_exit=True,
                        verbose=True,
                        representative_domain=self.exhaustive_representative_domain,
                    )
                verbose_output = stream.getvalue()
            else:
                result = check_all_interleavings(
                    self.tasks,
                    opaque_assignment=assignment,
                    early_exit=True,
                    representative_domain=self.exhaustive_representative_domain,
                )
            self._interleaving_results.append(
                (assignment, result, verbose_output)
            )
            if not result.is_safe:
                witness = (
                    result.deadlock_states[0].format_lines()[0]
                    if result.deadlock_states
                    else f"aliasing race: {result.race_states[0].overlap_desc}"
                )
                raise ValueError(
                    f"exhaustive schedule validation failed for opaque "
                    f"assignment {assignment}: {witness}"
                )

    def get_mermaid_string_dependency_graph(self) -> str:
        lines = ["flowchart LR"]
        for downstream, upstreams in self.resource_dependency_graph.items():
            for upstream in upstreams:
                lines.append(
                    f"  r{id(upstream)}[{upstream.name}] --> r{id(downstream)}[{downstream.name}]"
                )
        return "\n".join(lines)

    def get_marmaid_string_dependency_graph(self, *args, **kwargs) -> str:
        del args, kwargs
        return self.get_mermaid_string_dependency_graph()

    def _print_expanded_schedule_table(self) -> None:
        """Print the side-by-side representative schedule view."""
        task_line = "|"
        for task in self.tasks:
            name = f"{task.name}[{task.warp_start}:{task.warp_end})"
            task_line += f" {name:^28} |"
        table_width = len(task_line)

        def section_title(label: str) -> str:
            title = f"> {label} <"
            padding = max(table_width - len(title), 0)
            left = "=" * (padding // 2)
            right = "=" * (padding - padding // 2)
            return f"{left}{title}{right}"

        fallback = dynamic_domain_fallback_for(self.tasks)
        opaque_assignment = {key: True for key in collect_opaque_keys(self.tasks)}
        schedules = [
            expand_task(
                task,
                opaque_assignment=opaque_assignment,
                dynamic_domain_fallback=fallback,
                representative_domain=self.exhaustive_representative_domain,
            )
            for task in self.tasks
        ]
        indices = [0] * len(self.tasks)
        full_credits = {
            id(resource): 0
            for resource in self.resources
            if resource.pipeline_config is not None
        }
        empty_credits = {
            id(resource): resource.pipeline_config.num_stages
            for resource in self.resources
            if resource.pipeline_config is not None
        }

        print(f"\n{'=' * table_width}\n")
        print(task_line)
        print(f"\n{section_title('  Full (Once + Loop)  ')}\n")
        while any(index < len(schedule) for index, schedule in zip(indices, schedules)):
            line = "|"
            made_progress = False
            for task_index, schedule in enumerate(schedules):
                if indices[task_index] >= len(schedule):
                    line += " " * 30 + "|"
                    continue
                operation = schedule[indices[task_index]]
                resource = operation.resource
                stage = operation.schedule_stage
                resource_id = id(resource)
                blocked = (
                    stage is ScheduleStage.ProducerAcquire
                    and empty_credits[resource_id] <= 0
                ) or (
                    stage is ScheduleStage.ConsumerWait
                    and full_credits[resource_id] <= 0
                )
                if blocked:
                    line += " " * 30 + "|"
                    continue

                name = resource.name[:10]
                line += (
                    f" {name:10} {str(stage):12}{operation.phase:>2} "
                    f"{operation.unique_id:2} |"
                )
                indices[task_index] += 1
                made_progress = True
                if stage is ScheduleStage.ProducerAcquire:
                    empty_credits[resource_id] -= 1
                elif stage is ScheduleStage.ProducerCommit:
                    full_credits[resource_id] += 1
                elif stage is ScheduleStage.ConsumerWait:
                    full_credits[resource_id] -= 1
                elif stage is ScheduleStage.ConsumerRelease:
                    empty_credits[resource_id] += 1
            if not made_progress:
                raise ValueError("representative schedule table is blocked")
            print(line)

        print(f"\n{section_title('End Full (Once + Loop)')}\n")
        print(
            "Phase tags:  O — Once (outside any loop)"
            "  |  F — FirstIter (first iteration only)"
            "  |  L — LastIter (last iteration only)"
            "  |  P — Periodic (Every)  |  I — Opaque when_true/when_false"
        )

    def print_verbose_report(self) -> None:
        """Print task budgets, captured schedules, and host safety results."""
        if not self.verbose:
            return

        num_registers = 0
        register_budget = 0
        default_registers = self.default_task_registers
        for index, task in enumerate(self.tasks):
            registers = task.num_registers or default_registers
            contribution = task.num_warps * (default_registers - registers)
            num_registers += task.num_warps * 32 * registers
            register_budget += contribution
            print(
                f"Task {index:>2} '{task.name:<20}', "
                "numWarps x (defaultMaxNumRegs - numRegsPerThread) = "
                f"{task.num_warps} x ({default_registers:>3} - "
                f"{registers:>3}) = {contribution}"
            )
        free_register_blocks = max(
            0, (self._REGISTER_CAPACITY - num_registers) // (4 * 32 * 8)
        )
        print(
            f"Num. regs: {num_registers} "
            f"(free reg blocks: {free_register_blocks})"
        )
        print(f"Reg budget: {register_budget}")

        data_bytes, barrier_bytes = self._device_smem_usage()
        if data_bytes or barrier_bytes:
            total_bytes = data_bytes + barrier_bytes
            print(
                f"SMEM usage: {total_bytes} B "
                f"(data {data_bytes} B + "
                f"barriers {barrier_bytes} B) / "
                f"{self.smem_capacity_bytes} B capacity"
            )
        if self.tmem_allocator is not None:
            print(
                f"TMEM usage: {self.tmem_allocator.total_tmem_columns} columns "
                f"/ {self.tmem_capacity_columns} columns capacity"
            )

        self._print_expanded_schedule_table()
        for task in self.tasks:
            print(f"\n[{task.name} warps {task.warp_start}:{task.warp_end}]")
            print(task.schedule)

        for assignment, _, verbose_output in self._interleaving_results:
            print(verbose_output, end="")

        if self.smem_allocator is not None:
            self.smem_allocator.print_usage_report()
        if self.tmem_allocator is not None:
            self.tmem_allocator.print_usage_report()
        if self.barrier_allocator is not None:
            self.barrier_allocator.print_usage_report()

    def setup_resources(self) -> None:
        """Validate that every configured resource pipeline supports lowering."""
        for resource in self.resources:
            resource.create_pipeline()

    def to_device(
        self,
        task_callbacks: (
            dict[Task, dict[object, object]] | list[DeviceTask] | None
        ) = None,
    ) -> DeviceTaskManager:
        """Freeze the validated task set and its manager-owned resources.

        Directly decorated static work methods are inferred when
        ``task_callbacks`` is omitted. Explicit mappings by readable label (or
        ``"resource.label"`` when qualification is useful) override inferred
        callbacks. Task-local routing is automatic.
        Device pipeline bindings are derived from each resource's validated
        ``PipelineConfig``. Pipeline/barrier and SMEM storage is always dynamic
        and manager-owned. When an SMEM allocator is configured, its data arena
        is exposed through every task's ``ExecutionContext.smem_base``.
        Resource pipeline support is validated automatically before freezing.
        Passing an already lowered ``list[DeviceTask]`` remains supported for
        low-level adapters.
        """
        self.setup_resources()

        pipeline_bindings = {}
        pipeline_resources = [
            resource
            for resource in self.resources
            if resource.pipeline_config is not None
        ]
        if not isinstance(task_callbacks, list):
            for resource in pipeline_resources:
                pipeline_bindings.setdefault(
                    resource,
                    DevicePipelineBinding.from_config(resource.pipeline_config),
                )
        for resource in pipeline_resources:
            if resource not in pipeline_bindings:
                continue
            binding = pipeline_bindings[resource]
            if not isinstance(binding, DevicePipelineBinding):
                raise TypeError("pipeline bindings must be DevicePipelineBinding")
            if binding.num_stages != resource.pipeline_config.num_stages:
                raise ValueError(
                    f"pipeline binding for {resource.name!r} has "
                    f"{binding.num_stages} stages; expected "
                    f"{resource.pipeline_config.num_stages}"
                )

        barrier_offsets = {}
        barrier_initialization_runs = ()
        barrier_allocation_offsets = ()
        if self.barrier_allocator is not None:
            if not self.barrier_allocator.layout_computed:
                raise ValueError("barrier allocator layout has not been computed")
            for resource in pipeline_resources:
                try:
                    barrier_offsets[resource] = (
                        self.barrier_allocator.offset_of(f"{resource.name}.full"),
                        self.barrier_allocator.offset_of(f"{resource.name}.empty"),
                    )
                except KeyError as error:
                    raise ValueError(
                        "BarrierAllocator is missing full/empty storage for "
                        f"pipeline resource {resource.name!r}"
                    ) from error
            barrier_arena_size = self.barrier_allocator.padded_size
            barrier_initialization_runs = (
                self.barrier_allocator.initialization_runs
            )
            barrier_allocation_offsets = (
                self.barrier_allocator.allocation_offsets
            )
        else:
            barrier_arena_size = 0
            for resource in pipeline_resources:
                num_stages = resource.pipeline_config.num_stages
                barrier_offsets[resource] = (
                    barrier_arena_size,
                    barrier_arena_size + num_stages,
                )
                barrier_arena_size += 2 * num_stages
            raw_runs = []
            for resource in pipeline_resources:
                binding = pipeline_bindings.get(resource)
                if binding is None:
                    continue
                full, empty = barrier_offsets[resource]
                raw_runs.extend(
                    (
                        (full, full + binding.num_stages, binding.producer_arrivals),
                        (empty, empty + binding.num_stages, binding.consumer_arrivals),
                    )
                )
            barrier_initialization_runs = tuple(raw_runs)
            barrier_allocation_offsets = tuple(
                item
                for resource, (full, empty) in barrier_offsets.items()
                for item in (
                    (f"{resource.name}.full", full),
                    (f"{resource.name}.empty", empty),
                )
            )

        bound_pipeline_bindings = {
            resource: pipeline_bindings[resource].at_offsets(
                *barrier_offsets[resource]
            )
            for resource in pipeline_resources
            if resource in pipeline_bindings
        }
        if not bound_pipeline_bindings and self.barrier_allocator is None:
            barrier_arena_size = 0
            barrier_allocation_offsets = ()
        if isinstance(task_callbacks, list):
            tasks = task_callbacks
        else:
            task_callbacks = task_callbacks or {}
            unknown_tasks = set(task_callbacks) - set(self.tasks)
            if unknown_tasks:
                raise ValueError("callback mappings contain tasks not owned by manager")
            tasks = [
                task.to_device(
                    task_callbacks.get(task, {}),
                    pipeline_bindings=bound_pipeline_bindings,
                )
                for task in self.tasks
            ]

        if len(tasks) != len(self.tasks):
            raise ValueError(
                f"expected {len(self.tasks)} device tasks, got {len(tasks)}"
            )
        for host_task, device_task in zip(self.tasks, tasks):
            if not isinstance(device_task, DeviceTask):
                raise TypeError("tasks must contain only DeviceTask objects")
            if (
                device_task.warp_start != host_task.warp_start
                or device_task.warp_end != host_task.warp_end
            ):
                raise ValueError(
                    f"device task order does not match host task {host_task.name!r}"
                )
        smem_size_bytes = 0
        smem_alignment = 128
        smem_allocation_offsets = ()
        if self.smem_allocator is not None:
            smem_size_bytes = self.smem_allocator.data_smem_bytes
            smem_alignment = self.smem_allocator.data_alignment
            smem_allocation_offsets = self.smem_allocator.allocation_offsets
        manager = DeviceTaskManager(
            tasks=tuple(tasks),
            pipeline_bindings=tuple(
                bound_pipeline_bindings[resource]
                for resource in pipeline_resources
                if resource in bound_pipeline_bindings
            ),
            dynamic_pipeline_storage=True,
            barrier_initialization_runs=barrier_initialization_runs,
            barrier_arena_size=barrier_arena_size,
            barrier_allocation_offsets=barrier_allocation_offsets,
            barrier_arena_factory=barrier_arena_factory_for(
                barrier_arena_size,
                any(binding.uses_cluster for binding in bound_pipeline_bindings.values()),
            ),
            barrier_uses_cluster=any(
                binding.uses_cluster for binding in bound_pipeline_bindings.values()
            ),
            smem_size_bytes=smem_size_bytes,
            smem_alignment=smem_alignment,
            smem_allocation_offsets=smem_allocation_offsets,
            smem_base_factory=smem_base_factory_for(smem_size_bytes),
        )
        return manager

    def run(self) -> None:
        for task in self.tasks:
            task()
