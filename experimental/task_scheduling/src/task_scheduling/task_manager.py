# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only resource-graph, schedule, and budget validation."""

import io
import warnings
from contextlib import redirect_stdout

from .enums import PipelineType, ScheduleStage, SignalingThreads
from .exhaustive_checker import (
    check_all_interleavings,
    collect_opaque_keys,
    dynamic_domain_fallback_for,
    expand_task,
    format_opaque_assignment,
    opaque_assignment_sequences,
)
from .ir import freeze_program_ir
from .memory import BarrierAllocator, SmemAllocator, TmemAllocator
from .pipeline import DevicePipelineBinding
from .pipeline_group import PipelineGroup
from .schedule_builder import Schedule, _iter_steps
from .task import (
    DeviceTask,
    DeviceTaskManager,
    Task,
    barrier_arena_factory_for,
    smem_base_factory_for,
)

_WARP_SIZE = 32
_TMA_PIPELINE_TYPES = (PipelineType.TmaAsync, PipelineType.TmaUmma)


def _signaling_label(signaling: SignalingThreads) -> str:
    """Spell a SignalingThreads value the same way on every Python version."""
    names = [flag.name for flag in SignalingThreads if flag & signaling]
    return "|".join(names) or str(int(signaling))


def _static_cta_layout(config) -> tuple[int, int, int, int] | None:
    """Return the static VMNK cluster layout, or None when it is opaque."""
    layout = config.cta_layout_vmnk
    if layout is None:
        return (1, 1, 1, 1)
    if (
        isinstance(layout, (tuple, list))
        and len(layout) == 4
        and all(type(size) is int and size > 0 for size in layout)
    ):
        return tuple(layout)
    return None


def _rank_coordinates(rank: int, layout: tuple[int, int, int, int]) -> tuple[int, ...]:
    v_size, m_size, n_size, _ = layout
    return (
        rank % v_size,
        (rank // v_size) % m_size,
        (rank // (v_size * m_size)) % n_size,
        rank // (v_size * m_size * n_size),
    )


def _tcgen05_commit_sources(
    layout: tuple[int, int, int, int],
    mcast_mode_mn: tuple[int, int],
    *,
    cta_leader: bool,
    pinned: bool,
    vmnk_multicast: bool,
) -> int:
    """Count the CTAs whose ``tcgen05.commit`` lands on one CTA's barrier.

    A pinned task commits from one CTA and ``CtaLeader`` signaling from the
    leader of each V-group. The commit multicasts to the committing CTA's own
    V-group, or with ``vmnk_multicast`` (a ``TmaUmma`` release) to every CTA
    of the same K slice that shares the M coordinate when ``mcast_mode_mn[0]``
    is set or the N coordinate when ``mcast_mode_mn[1]`` is set, which mirrors
    ``DevicePipelineBinding.tma_umma_multicast_mask``.
    """
    v_size = layout[0]
    total = v_size * layout[1] * layout[2] * layout[3]
    if pinned:
        return 1
    if vmnk_multicast and total != v_size:
        multicast_m, multicast_n = mcast_mode_mn
        count = 0
        for rank in range(total):
            v, m, n, k = _rank_coordinates(rank, layout)
            if k != 0 or (cta_leader and v != 0):
                continue
            if (multicast_m and m == 0) or (multicast_n and n == 0):
                count += 1
        return count
    return 1 if cta_leader else v_size


def _tma_async_lanes_per_warp(
    layout: tuple[int, int, int, int], mcast_mode_mn: tuple[int, int]
) -> int:
    """Return the empty-arrive lanes CUTLASS's TmaAsync release selects per warp.

    One lane signals each cluster peer that multicasts into this CTA: the CTAs
    sharing the M coordinate (N peers) when ``mcast_mode_mn[0]`` is set and the
    CTAs sharing the N coordinate (M peers) when ``mcast_mode_mn[1]`` is set,
    counting this CTA once. This mirrors
    ``DevicePipelineBinding.tma_async_consumer_selected``.
    """
    _, m_size, n_size, _ = layout
    multicast_m, multicast_n = mcast_mode_mn
    lanes = 0
    if multicast_m:
        lanes += n_size
    if multicast_n:
        lanes += m_size
    if multicast_m and multicast_n:
        lanes -= 1
    return max(lanes, 1)


class TaskManager:
    _DEFAULT_SMEM_CAPACITY_BYTES = 232448
    _DEFAULT_TMEM_CAPACITY_COLUMNS = 512
    _REGISTER_CAPACITY = 65536

    def __init__(
        self,
        tasks: list[Task],
        resource_dependency_graph: dict[object, list[object]],
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
        if not isinstance(resource_dependency_graph, dict):
            raise TypeError("resource_dependency_graph must be a dictionary")
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
        self.resource_dependency_graph = resource_dependency_graph
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
        has_requested_barrier_offsets = any(
            resource.pipeline_config.storage_offset_full is not None
            for resource in pipeline_resources
        )
        if self.barrier_allocator is None and (
            len(pipeline_resources) > 1 or has_requested_barrier_offsets
        ):
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
            self._verify_pipeline_thread_counts,
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
            for step in _iter_steps(task.schedule):
                rid = id(step.memory_resource)
                stage = step.schedule_stage
                if stage is ScheduleStage.ProducerAcquire:
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
                    open_count = state.get((rid, "consumer"), 0)
                    if open_count and not (
                        config is not None and config.advance_on_wait
                    ):
                        raise ValueError("nested ConsumerWait without release")
                    state[(rid, "consumer")] = open_count + 1
                elif stage is ScheduleStage.ConsumerWork:
                    if step.memory_resource.pipeline_config and not state.get(
                        (rid, "consumer")
                    ):
                        raise ValueError(
                            f"{task.name}: ConsumerWork on {step.memory_resource.name} "
                            "is not bracketed by wait/release"
                        )
                elif stage is ScheduleStage.ConsumerRelease:
                    open_count = state.get((rid, "consumer"), 0)
                    if not open_count:
                        raise ValueError("ConsumerRelease without ConsumerWait")
                    if open_count == 1:
                        state.pop((rid, "consumer"))
                    else:
                        state[(rid, "consumer")] = open_count - 1
            if state:
                raise ValueError(f"{task.name}: unterminated pipeline work group")
        del producer_open

    def _verify_pipeline_thread_counts(self) -> None:
        """Verify cooperative-group sizes against the arrivals the lowering emits.

        Every full and empty barrier is initialized with ``group.size`` expected
        arrivals, so that size must equal the number of ``mbarrier.arrive``,
        ``arrive.expect_tx`` or ``tcgen05.commit`` signals one stage receives
        from the tasks that commit or release the resource:

        * AsyncThread sides (``AsyncAsync`` both sides, ``AsyncUmma`` producers,
          ``UmmaAsync`` and ``ClcFetchAsync`` consumers): every thread of every
          signaling task arrives, multiplied by the CTAs that route into the
          same barrier (the V-group of a UMMA pipeline, the whole cluster for
          CLC). ``CtaLeader`` signaling and ``run_only_on_cta_id`` tasks count
          one CTA. A size of 1 is accepted when exactly one warp on one CTA
          signals, because the lowering then elects one lane.
        * tcgen05 sides (``UmmaAsync``/``UmmaUmma`` producers, ``TmaUmma``/
          ``AsyncUmma``/``UmmaUmma`` consumers): one commit per signaling warp
          per CTA whose multicast mask covers the barrier, that is the V-group,
          or for ``TmaUmma`` releases the CTAs sharing the M or N coordinate
          selected by ``mcast_mode_mn``.
        * TMA producers: ``arrive.expect_tx`` is issued once per producer
          warp, or once per task under ``TaskWarpLeader``; a multi-warp TMA
          task needs one of ``TaskWarpLeader`` or ``interleave_stride``.
        * ``TmaAsync`` consumers: the CUTLASS lane model (one lane per
          multicast peer per warp) or an all-thread release.
        * CLC producers: one ``arrive.expect_tx`` per CTA barrier.

        A side with no committing or releasing task is skipped; the barrier
        model of the exhaustive checker reports that situation.
        """

        def signaling_tasks(resource, stage):
            return [
                task
                for task in self.tasks
                if any(
                    step.memory_resource is resource and step.schedule_stage is stage
                    for step in _iter_steps(task.schedule)
                )
            ]

        def describe(tasks):
            return ", ".join(
                f"'{task.name}' ({task.num_warps} warps"
                + (
                    f", CTA {task.run_only_on_cta_id}"
                    if task.run_only_on_cta_id is not None
                    else ""
                )
                + ")"
                for task in tasks
            )

        def mismatch(resource, side, size, expected, formula, tasks, signaling=None):
            message = (
                f"Resource '{resource.name}' "
                f"(pipeline={resource.pipeline_config.pipeline_type.value}): "
                f"{side}_group.size={size} does not match expected {expected} "
                f"(= {formula}). Task(s): [{describe(tasks)}]"
            )
            if signaling is not None:
                message += f", signaling={_signaling_label(signaling)}"
            raise ValueError(message + ".")

        def cta_scale(task, signaling, scale):
            if signaling.has_cta_leader() or task.run_only_on_cta_id is not None:
                return 1
            return scale

        def check_async_thread_side(resource, side, size, signaling, tasks, scale):
            """Every thread arrives, or one elected lane when a lone warp signals."""
            if not tasks:
                return
            warps = sum(task.num_warps * cta_scale(task, signaling, scale) for task in tasks)
            expected = warps * _WARP_SIZE
            if size == expected or (size == 1 and warps == 1):
                return
            formula = "Σ task.num_warps × 32"
            if scale > 1:
                formula += f" × ctas({scale})"
            mismatch(resource, side, size, expected, formula, tasks, signaling)

        def check_tcgen05_side(resource, side, size, signaling, tasks, layout, mcast):
            """One tcgen05.commit per warp per CTA whose multicast reaches the barrier."""
            if not tasks:
                return
            expected = 0
            for task in tasks:
                sources = _tcgen05_commit_sources(
                    layout,
                    resource.pipeline_config.mcast_mode_mn,
                    cta_leader=signaling.has_cta_leader(),
                    pinned=task.run_only_on_cta_id is not None,
                    vmnk_multicast=mcast,
                )
                expected += task.num_warps * sources
            if size != expected:
                mismatch(
                    resource,
                    side,
                    size,
                    expected,
                    "Σ task.num_warps × committing CTAs whose tcgen05.commit "
                    "multicast reaches this barrier",
                    tasks,
                    signaling,
                )

        def check_tma_producer_side(resource, size, signaling, tasks, stride):
            """arrive.expect_tx is elected once per warp, or once per task."""
            if not tasks:
                return
            config = resource.pipeline_config
            multi_warp = any(task.num_warps > 1 for task in tasks)
            if multi_warp and stride <= 1 and not signaling.has_task_warp_leader():
                raise ValueError(
                    f"Resource '{resource.name}' (pipeline={config.pipeline_type.value}) "
                    f"has TMA producer task(s) with more than one warp, but "
                    f"interleave_stride={stride} and producer_signaling_threads="
                    f"{_signaling_label(signaling)}. Multi-warp TMA producer tasks must either use "
                    f"interleave_stride > 1 for independent per-warp barrier stages or "
                    f"producer_signaling_threads=TaskWarpLeader for one aggregate "
                    f"transaction-barrier arrive. Task(s): [{describe(tasks)}]."
                )
            if multi_warp and stride > 1 and signaling.has_task_warp_leader():
                raise ValueError(
                    f"Resource '{resource.name}' (pipeline={config.pipeline_type.value}) "
                    f"combines interleave_stride={stride} with "
                    f"producer_signaling_threads={_signaling_label(signaling)}. "
                    f"Multi-warp TMA producer "
                    f"tasks must choose either interleaved independent per-warp "
                    f"arrivals or TaskWarpLeader aggregate arrivals."
                )
            if stride > 1:
                if size != 1:
                    raise ValueError(
                        f"Resource '{resource.name}' "
                        f"(pipeline={config.pipeline_type.value}): "
                        f"producer_group.size={size} but interleave_stride={stride} "
                        f"means each barrier stage is signaled by one elected warp "
                        f"thread. Expected producer_group.size=1."
                    )
                return
            if signaling.has_task_warp_leader():
                expected = len(tasks)
                formula = (
                    "one selected warp per task, because TaskWarpLeader guards "
                    "arrive.expect_tx"
                )
            else:
                expected = sum(task.num_warps for task in tasks)
                formula = (
                    "Σ producer_task.num_warps, because arrive.expect_tx is elected "
                    "once per warp"
                )
            if size != expected:
                mismatch(resource, "producer", size, expected, formula, tasks)

        def check_interleaved_release(resource, size, tasks, stride):
            if tasks and size != 1:
                raise ValueError(
                    f"Resource '{resource.name}' "
                    f"(pipeline={resource.pipeline_config.pipeline_type.value}): "
                    f"consumer_group.size={size} but interleave_stride={stride} means "
                    f"each empty barrier stage is released by one elected warp thread. "
                    f"Expected consumer_group.size=1."
                )

        def check_tma_async_consumer_side(resource, size, signaling, tasks, layout):
            """CUTLASS lane model (one lane per multicast peer) or all threads."""
            if not tasks:
                return
            config = resource.pipeline_config
            lanes = _tma_async_lanes_per_warp(layout, config.mcast_mode_mn)
            warps = sum(task.num_warps for task in tasks)
            all_threads = warps * _WARP_SIZE
            lane_expected = warps * lanes
            if size in (all_threads, lane_expected):
                return
            lane_formula = "Σ consumer_task.num_warps"
            if lanes != 1:
                lane_formula += f" × multicast lanes-per-warp({lanes})"
            raise ValueError(
                f"Resource '{resource.name}' (pipeline={config.pipeline_type.value}): "
                f"consumer_group.size={size} does not match expected {lane_expected} "
                f"for a per-warp lane release (= {lane_formula}) or {all_threads} for "
                f"an all-thread release (= Σ consumer_task.num_warps × 32). "
                f"Task(s): [{describe(tasks)}], signaling={_signaling_label(signaling)}."
            )

        def check_fixed(resource, side, size, expected, reason):
            if size != expected:
                raise ValueError(
                    f"Resource '{resource.name}' "
                    f"(pipeline={resource.pipeline_config.pipeline_type.value}): "
                    f"{side}_group.size={size} but {reason}. "
                    f"Expected {side}_group.size={expected}."
                )

        for resource in self.resources:
            config = resource.pipeline_config
            if config is None or isinstance(resource, PipelineGroup):
                continue
            ptype = config.pipeline_type
            layout = _static_cta_layout(config)
            if layout is None:
                warnings.warn(
                    f"resource {resource.name!r} has an opaque cta_layout_vmnk; "
                    "skipping pipeline thread-count validation",
                    stacklevel=2,
                )
                continue
            v_size = layout[0]
            cluster_total = layout[0] * layout[1] * layout[2] * layout[3]
            producers = signaling_tasks(resource, ScheduleStage.ProducerCommit)
            consumers = signaling_tasks(resource, ScheduleStage.ConsumerRelease)
            producer_size = config.producer_group.size
            consumer_size = config.consumer_group.size
            producer_signaling = config.producer_signaling_threads
            consumer_signaling = config.consumer_signaling_threads

            if producer_signaling.has_task_warp_leader() and ptype not in _TMA_PIPELINE_TYPES:
                raise ValueError(
                    f"Resource '{resource.name}' (pipeline={ptype.value}) uses "
                    f"producer_signaling_threads={_signaling_label(producer_signaling)}, but "
                    f"TaskWarpLeader is only supported for TMA producer pipelines."
                )
            if consumer_signaling.has_task_warp_leader():
                raise ValueError(
                    f"Resource '{resource.name}' (pipeline={ptype.value}) uses "
                    f"consumer_signaling_threads={_signaling_label(consumer_signaling)}, but "
                    f"TaskWarpLeader is producer-side only."
                )
            wait_signaling = config.consumer_wait_signaling_threads
            if wait_signaling is not None and wait_signaling.has_task_warp_leader():
                raise ValueError(
                    f"Resource '{resource.name}' (pipeline={ptype.value}) uses "
                    f"consumer_wait_signaling_threads={_signaling_label(wait_signaling)}, but "
                    f"TaskWarpLeader is producer-side only."
                )

            # Producer side: the full barrier.
            if ptype is PipelineType.AsyncAsync:
                check_async_thread_side(
                    resource, "producer", producer_size, producer_signaling, producers, 1
                )
            elif ptype is PipelineType.AsyncUmma:
                # Arrivals are routed to the V-group leader's full barrier.
                check_async_thread_side(
                    resource, "producer", producer_size, producer_signaling, producers, v_size
                )
            elif ptype in (PipelineType.UmmaAsync, PipelineType.UmmaUmma):
                check_tcgen05_side(
                    resource, "producer", producer_size, producer_signaling, producers,
                    layout, False,
                )
            elif ptype in _TMA_PIPELINE_TYPES:
                check_tma_producer_side(
                    resource, producer_size, producer_signaling, producers,
                    config.producer_commit_interleave_stride,
                )
            elif ptype is PipelineType.ClcFetchAsync:
                check_fixed(
                    resource, "producer", producer_size, 1,
                    "the scheduler thread arms each CTA's full barrier with one "
                    "arrive.expect_tx",
                )

            # Consumer side: the empty barrier.
            release_stride = config.consumer_release_interleave_stride
            if ptype is PipelineType.TmaUmma:
                if release_stride > 1:
                    check_interleaved_release(resource, consumer_size, consumers, release_stride)
                else:
                    check_tcgen05_side(
                        resource, "consumer", consumer_size, consumer_signaling, consumers,
                        layout, True,
                    )
            elif ptype in (PipelineType.AsyncUmma, PipelineType.UmmaUmma):
                check_tcgen05_side(
                    resource, "consumer", consumer_size, consumer_signaling, consumers,
                    layout, False,
                )
            elif ptype is PipelineType.AsyncAsync:
                check_async_thread_side(
                    resource, "consumer", consumer_size, consumer_signaling, consumers, 1
                )
            elif ptype is PipelineType.UmmaAsync:
                # Arrivals are routed to the V-group leader's empty barrier.
                check_async_thread_side(
                    resource, "consumer", consumer_size, consumer_signaling, consumers, v_size
                )
            elif ptype is PipelineType.ClcFetchAsync:
                # Every CTA releases into the cluster leader's empty barrier.
                check_async_thread_side(
                    resource, "consumer", consumer_size, consumer_signaling, consumers,
                    cluster_total,
                )
            elif ptype is PipelineType.TmaAsync:
                if release_stride > 1:
                    check_interleaved_release(resource, consumer_size, consumers, release_stride)
                else:
                    check_tma_async_consumer_side(
                        resource, consumer_size, consumer_signaling, consumers, layout
                    )

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
        # A while-loop task contributes retry^(N-1) + break round histories.
        assignments = opaque_assignment_sequences(
            self.tasks,
            collect_opaque_keys(self.tasks),
            representative_domain=self.exhaustive_representative_domain,
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
                if result.deadlock_states:
                    witness = "; ".join(result.deadlock_states[0].format_lines())
                elif result.race_states:
                    witness = (
                        f"aliasing race: {result.race_states[0].overlap_desc}"
                    )
                elif result.hit_state_limit:
                    witness = (
                        "inconclusive: exhaustive checker hit its state limit "
                        f"after {result.states_explored} states without reaching "
                        "a complete execution"
                    )
                else:
                    witness = "inconclusive: no complete execution or witness"
                raise ValueError(
                    f"exhaustive schedule validation failed for opaque "
                    f"assignment {format_opaque_assignment(assignment)}: {witness}"
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
        opaque_keys = collect_opaque_keys(self.tasks)
        opaque_assignment = {key: True for key in opaque_keys}
        assignments = opaque_assignment_sequences(
            self.tasks,
            opaque_keys,
            representative_domain=self.exhaustive_representative_domain,
        )
        if any(not isinstance(item, dict) for item in assignments):
            # Show the history with the most while-loop rounds.
            opaque_assignment = max(
                assignments,
                key=lambda item: max(map(len, item.task_assignments.values())),
            )
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

    def _consumer_release_uses_per_warp_election(self, resource) -> bool:
        """Return whether a TMA consumer contributes one arrival per warp."""
        config = resource.pipeline_config
        if config.pipeline_type is not PipelineType.TmaAsync:
            return False
        release_tasks = self._consumer_release_tasks(resource)
        return bool(release_tasks) and config.consumer_group.size == sum(
            task.num_warps for task in release_tasks
        )

    def _consumer_release_uses_all_threads(self, resource) -> bool:
        """Return whether a TMA consumer deliberately bypasses lane routing."""
        config = resource.pipeline_config
        if config.pipeline_type is not PipelineType.TmaAsync:
            return False
        release_tasks = self._consumer_release_tasks(resource)
        return bool(release_tasks) and config.consumer_group.size == sum(
            task.num_warps * 32 for task in release_tasks
        )

    def _consumer_release_tasks(self, resource) -> list[Task]:
        """Return tasks that release the specified resource's empty barrier."""
        return [
            task
            for task in self.tasks
            if any(
                step.memory_resource is resource
                and step.schedule_stage is ScheduleStage.ConsumerRelease
                for step in _iter_steps(task.schedule)
            )
        ]

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
                    DevicePipelineBinding.from_config(
                        resource.pipeline_config,
                        consumer_elected_per_warp=(
                            self._consumer_release_uses_per_warp_election(resource)
                        ),
                        consumer_all_threads=(
                            self._consumer_release_uses_all_threads(resource)
                        ),
                    ),
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
                    offsets = (
                        self.barrier_allocator.offset_of(f"{resource.name}.full"),
                        self.barrier_allocator.offset_of(f"{resource.name}.empty"),
                    )
                except KeyError as error:
                    raise ValueError(
                        "BarrierAllocator is missing full/empty storage for "
                        f"pipeline resource {resource.name!r}"
                    ) from error
                config = resource.pipeline_config
                requested_offsets = (
                    config.storage_offset_full,
                    config.storage_offset_empty,
                )
                if requested_offsets[0] is not None and offsets != requested_offsets:
                    raise ValueError(
                        f"BarrierAllocator assigned {resource.name!r} offsets "
                        f"{offsets}, but its PipelineConfig requests "
                        f"{requested_offsets}"
                    )
                barrier_offsets[resource] = offsets
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
