# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tree interpreter shared by host construction and CUDA Lang lowering."""

from dataclasses import dataclass, replace

import cuda.lang as cl

from .enums import OpaqueCondition, SKIPPABLE, ScheduleStage
from .ir import ScheduleValue
from .pipeline import DevicePipelineBinding, PipelineState, require_device_support
from .resources import MemoryResource, StageInfo, _get_static_work_fn
from .schedule_builder import (
    ConditionalBlock,
    DomainLoop,
    DynamicDomainBound,
    Node,
    Schedule,
    Step,
    WorkTileLoop,
    _iter_nodes,
    validate_queue_advance_placement,
)


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable runtime values; no compiler value is stored on host metadata."""

    loop_offset: object = 0
    iteration_index: object = 0
    num_iterations: object = 0
    loop_start: object = 0
    loop_end: object = 0
    loop_step: object = 1
    in_domain_loop: object = False
    work_tile: object | None = None
    skipped: object = False
    route_values: tuple[object, ...] = ()
    smem_base: object | None = None
    cluster_smem_base: object | None = None
    tmem_base: object | None = None
    tasks_inputs: object = ()
    warp_index: object | None = None
    pipeline_states: tuple[PipelineState, ...] = ()
    barrier_arena: object | None = None
    active_pipeline_slot: int = -1
    pipeline_tile_iteration: object = 0
    multistage_iterations: object = 0

    def route(self, index: int) -> object:
        return self.route_values[index]

    def append_routes(self, values: tuple[object, ...]) -> "ExecutionContext":
        return replace(self, route_values=self.route_values + values)

    def pop_routes(self, count: int) -> "ExecutionContext":
        if count == 0:
            return self
        return replace(self, route_values=self.route_values[:-count])

    def truncate_routes(self, depth: int) -> "ExecutionContext":
        return replace(self, route_values=self.route_values[:depth])

    def pipeline_state(self, slot: int | None = None) -> PipelineState:
        selected = self.active_pipeline_slot if slot is None else slot
        return self.pipeline_states[selected]

    def pipeline_index(self, slot: int | None = None) -> object:
        return self.pipeline_state(slot).index

    def store_pipeline_state(
        self, slot: int, state: PipelineState
    ) -> "ExecutionContext":
        states = (
            self.pipeline_states[:slot] + (state,) + self.pipeline_states[slot + 1:]
        )
        return replace(self, pipeline_states=states)


@dataclass(frozen=True)
class DeviceGuard:
    kind: str
    period: int
    start: int
    slot: int
    negated: bool

    def active(self, context: ExecutionContext) -> object:
        if self.kind == "first":
            return context.iteration_index == 0
        if self.kind == "last":
            return context.iteration_index == context.num_iterations - 1
        if self.kind == "every":
            return (
                context.iteration_index >= self.start
                and (context.iteration_index - self.start) % self.period == 0
            )
        if self.kind == "opaque":
            value = context.route(self.slot)
            active = value != 0
            return active != self.negated
        if self.kind == "skippable":
            return not context.skipped
        return False


@dataclass(frozen=True)
class DeviceStep:
    callback: object
    unique_id: int
    static_args: tuple[object, ...] = ()
    argument_order: tuple[int, ...] = ()
    pipeline_slot: int = -1
    input_slots: tuple[int, ...] = ()
    append_output_count: int = 0
    release_before_append: int = 0
    release_after_append: int = 0
    automatic_routing: bool = False
    pipeline_binding: DevicePipelineBinding | None = None
    pass_stage_info: bool = False
    label: object | None = None
    in_domain_loop: bool = False

    def _callback_argument(self, context: ExecutionContext):
        if not self.pass_stage_info:
            return context
        loop_offset = context.loop_offset if self.in_domain_loop else None
        loop_start = context.loop_start if self.in_domain_loop else None
        loop_end = context.loop_end if self.in_domain_loop else None
        loop_step = context.loop_step if self.in_domain_loop else None
        if self.pipeline_binding is None:
            return StageInfo(
                count=context.iteration_index,
                loop_offset=loop_offset,
                loop_start=loop_start,
                loop_end=loop_end,
                loop_step=loop_step,
                label=self.label,
                work_tile=context.work_tile,
                context=context,
            )
        state = context.pipeline_state(self.pipeline_slot)
        return StageInfo(
            index=state.index,
            stage_idx=state.index,
            phase=state.phase,
            barrier=self.pipeline_binding.work_barrier(
                context.barrier_arena, state.index
            ),
            count=context.iteration_index,
            loop_offset=loop_offset,
            loop_start=loop_start,
            loop_end=loop_end,
            loop_step=loop_step,
            label=self.label,
            work_tile=context.work_tile,
            context=context,
        )

    def _call_with_routed_values(self, context: ExecutionContext):
        callback_argument = self._callback_argument(context)
        routed_values = tuple(
            context.route(self.input_slots[index])
            for index in cl.static_iter(range(len(self.input_slots)))
        )
        unordered_values = routed_values + self.static_args
        ordered_values = tuple(
            unordered_values[index]
            for index in cl.static_iter(self.argument_order)
        )
        return self.callback(
            callback_argument,
            *ordered_values,
        )

    def _store_routed_values(self, context: ExecutionContext, result):
        context = context.pop_routes(self.release_before_append)
        if self.append_output_count:
            values = result if self.append_output_count > 1 else (result,)
            context = context.append_routes(values)
        return context.pop_routes(self.release_after_append)

    def __call__(self, context: ExecutionContext) -> ExecutionContext:
        cl.ptx_comment(f"task_scheduling step #{self.unique_id}")
        context = replace(context, active_pipeline_slot=self.pipeline_slot)
        if self.automatic_routing:
            result = self._call_with_routed_values(context)
            return self._store_routed_values(context, result)
        callback_argument = self._callback_argument(context)
        return self.callback(callback_argument, *self.static_args)


@dataclass(frozen=True)
class DevicePipelineStep:
    TRY_ACQUIRE = 0
    ACQUIRE = 1
    COMMIT = 2
    TRY_WAIT = 3
    WAIT = 4
    RELEASE = 5

    action: int
    binding: DevicePipelineBinding
    state_slot: int
    unique_id: int

    def _signal_selected(self, producer: bool):
        """Select one fixed lane when a barrier expects one signaling thread."""
        elected = (
            self.binding.producer_elected if producer else self.binding.consumer_elected
        )
        selected = cl.elect_sync() if elected else True
        # A clustered TMA->UMMA pipeline aggregates both CTAs' TMA
        # transactions in the leader CTA's full barrier.  Every producer CTA
        # still waits for its local empty barrier, but only the leader CTA may
        # arm the shared transaction barrier.  Arming the peer barrier leaves
        # it transaction-incomplete and makes the first stage reuse invalid.
        if (
            producer
            and self.binding.kind == DevicePipelineBinding.TMA_UMMA
            and self.binding.uses_cluster
        ):
            selected = selected and self.binding.is_leader_cta()
        cta_leader = (
            self.binding.producer_cta_leader
            if producer
            else self.binding.consumer_cta_leader
        )
        if cta_leader:
            selected = selected and self.binding.is_leader_cta()
        return selected

    def __call__(self, context: ExecutionContext) -> ExecutionContext:
        cl.ptx_comment(f"task_scheduling pipeline step #{self.unique_id}")
        state = context.pipeline_state(self.state_slot)
        full = self.binding.full_barrier(context.barrier_arena, state.index)
        empty = self.binding.empty_barrier(context.barrier_arena, state.index)

        if self.action == self.TRY_ACQUIRE:
            status = cl.mbarrier_try_wait_parity(empty, state.phase)
            return context.store_pipeline_state(
                self.state_slot, state.with_status(status)
            )

        if self.action == self.ACQUIRE:
            if not state.status:
                cl.mbarrier_wait_parity(
                    empty,
                    state.phase,
                    time_hint=10_000_000,
                )
            if self.binding.kind in (
                DevicePipelineBinding.TMA_ASYNC,
                DevicePipelineBinding.TMA_UMMA,
            ):
                selected = self._signal_selected(True)
                if selected:
                    cl.mbarrier_arrive_expect_transaction(full, self.binding.num_bytes)
            return context.store_pipeline_state(
                self.state_slot, state.with_status(False)
            )

        if self.action == self.COMMIT:
            if self.binding.kind in (
                DevicePipelineBinding.ASYNC_ASYNC,
                DevicePipelineBinding.ASYNC_UMMA,
            ):
                if self._signal_selected(True):
                    cl.mbarrier_arrive(full)
            elif self.binding.kind == DevicePipelineBinding.UMMA_ASYNC:
                if self._signal_selected(True):
                    if self.binding.uses_cluster:
                        cl.tcgen05_commit(
                            full,
                            multicast_mask=self.binding.multicast_mask(),
                            cta_group=cl.CTAGroup.CTA_2,
                        )
                    else:
                        cl.tcgen05_commit(full, cta_group=cl.CTAGroup.CTA_1)
            return context.store_pipeline_state(
                self.state_slot, state.advance(self.binding.num_stages)
            )

        if self.action == self.TRY_WAIT:
            status = cl.mbarrier_try_wait_parity(full, state.phase)
            return context.store_pipeline_state(
                self.state_slot, state.with_status(status)
            )

        if self.action == self.WAIT:
            if not state.status:
                cl.mbarrier_wait_parity(
                    full,
                    state.phase,
                    time_hint=10_000_000,
                )
            if self.binding.kind in (
                DevicePipelineBinding.TMA_UMMA,
                DevicePipelineBinding.ASYNC_UMMA,
                DevicePipelineBinding.UMMA_ASYNC,
            ):
                cl.tcgen05_fence_after_thread_sync()
            return context.store_pipeline_state(
                self.state_slot, state.with_status(False)
            )

        if self._signal_selected(False):
            if self.binding.kind in (
                DevicePipelineBinding.TMA_UMMA,
                DevicePipelineBinding.ASYNC_UMMA,
            ):
                if self.binding.uses_cluster:
                    cl.tcgen05_commit(
                        empty,
                        multicast_mask=self.binding.multicast_mask(),
                        cta_group=cl.CTAGroup.CTA_2,
                    )
                else:
                    cl.tcgen05_commit(empty, cta_group=cl.CTAGroup.CTA_1)
            else:
                cl.mbarrier_arrive(
                    self.binding.consumer_release_barrier(
                        context.barrier_arena, state.index
                    ),
                    scope=cl.MbarrierScope.BLOCK,
                )
        return context.store_pipeline_state(
            self.state_slot, state.advance(self.binding.num_stages)
        )


@dataclass(frozen=True)
class DeviceConditional:
    body: tuple[object, ...]
    guard: DeviceGuard
    release_count: int = 0

    def __call__(self, context: ExecutionContext) -> ExecutionContext:
        if self.guard.active(context):
            route_depth = len(context.route_values)
            body_context = _run_device_nodes(self.body, context)
            context = body_context.pop_routes(
                len(body_context.route_values) - route_depth
            )
        return context.pop_routes(self.release_count)


@dataclass(frozen=True)
class DeviceDomainLoop:
    start: int
    end: int
    step: int
    num_iterations: int
    body: tuple[object, ...]
    dynamic_start: bool = False
    dynamic_end: bool = False
    start_resolver: object = None
    end_resolver: object = None
    release_count: int = 0
    initial_route_positions: tuple[int, ...] = ()
    yield_route_positions: tuple[int, ...] = ()
    result_indices: tuple[int, ...] = ()

    def __call__(self, context: ExecutionContext) -> ExecutionContext:
        start = self.start
        if self.dynamic_start:
            start = self.start_resolver(context.tasks_inputs)
        end = self.end
        if self.dynamic_end:
            end = self.end_resolver(context.tasks_inputs)
        num_iterations = self.num_iterations
        if self.num_iterations < 0:
            distance = end - start
            num_iterations = (
                (distance + self.step - 1) // self.step if distance > 0 else 0
            )
        outer_loop_offset = context.loop_offset
        outer_iteration_index = context.iteration_index
        outer_num_iterations = context.num_iterations
        outer_loop_start = context.loop_start
        outer_loop_end = context.loop_end
        outer_loop_step = context.loop_step
        outer_in_domain_loop = context.in_domain_loop
        route_depth = len(context.route_values)
        carried = tuple(
            context.route(position)
            for position in cl.static_iter(self.initial_route_positions)
        )
        for loop_offset in range(start, end, self.step):
            iteration_index = (loop_offset - start) // self.step
            iteration_context = replace(
                context,
                loop_offset=loop_offset,
                iteration_index=iteration_index,
                num_iterations=num_iterations,
                loop_start=start,
                loop_end=end,
                loop_step=self.step,
                in_domain_loop=True,
                route_values=context.route_values + carried,
            )
            body_context = _run_device_nodes(self.body, iteration_context)
            carried = tuple(
                body_context.route(position)
                for position in cl.static_iter(self.yield_route_positions)
            )
            context = replace(body_context, route_values=context.route_values)
        context = replace(
            context,
            loop_offset=outer_loop_offset,
            iteration_index=outer_iteration_index,
            num_iterations=outer_num_iterations,
            loop_start=outer_loop_start,
            loop_end=outer_loop_end,
            loop_step=outer_loop_step,
            in_domain_loop=outer_in_domain_loop,
        )
        results = tuple(
            carried[index] for index in cl.static_iter(self.result_indices)
        )
        context = replace(
            context,
            route_values=context.route_values[:route_depth] + results,
        )
        return context.pop_routes(self.release_count)


def _run_device_nodes(
    nodes: tuple[object, ...], context: ExecutionContext, index: int = 0
):
    """Statically walk nodes while allowing anonymous route tuple growth.

    A ``cl.static_iter`` loop makes ``context`` a loop-carried value, whose type
    cannot change when a producer appends a routed value. Recursive traversal
    gives every schedule position its own statically known context type. The
    schedule tree is frozen and finite, so this recursion is resolved during
    lowering and does not exist in generated device code.
    """
    if index == len(nodes):
        return context
    return _run_device_nodes(nodes, nodes[index](context), index + 1)


def _task_warp_index():
    """Return a warp-uniform physical index for one-dimensional task blocks."""
    warp_index = cl.thread_index(0) // cl.lane_count()
    return cl.shfl_sync(warp_index, 0)


@dataclass(frozen=True)
class DeviceTask:
    body: tuple[object, ...]
    warp_start: int
    warp_end: int
    num_registers: int
    run_only_on_cta_id: int
    initial_pipeline_states: tuple[PipelineState, ...] = ()
    default_num_registers: int = 128
    pipeline_stage_counts: tuple[int, ...] = ()
    pipeline_advances_per_domain: tuple[bool, ...] = ()

    def make_context(
        self,
        tasks_inputs: object,
        barrier_arena: object | None = None,
        smem_base: object | None = None,
        cluster_smem_base: object | None = None,
        tmem_base: object | None = None,
        warp_index: object | None = None,
        pipeline_tile_iteration: object = 0,
        multistage_iterations: object = 0,
    ) -> ExecutionContext:
        """Create the immutable runtime context for this task."""
        return ExecutionContext(
            smem_base=smem_base,
            cluster_smem_base=cluster_smem_base,
            tmem_base=tmem_base,
            tasks_inputs=tasks_inputs,
            warp_index=warp_index,
            barrier_arena=barrier_arena,
            pipeline_tile_iteration=pipeline_tile_iteration,
            multistage_iterations=multistage_iterations,
        )

    def __call__(self, context: ExecutionContext) -> ExecutionContext:
        pipeline_states = tuple(
            PipelineState(
                index=(
                    context.pipeline_tile_iteration
                    * (
                        context.multistage_iterations
                        if self.pipeline_advances_per_domain[index]
                        else 1
                    )
                )
                % self.pipeline_stage_counts[index],
                phase=(
                    state.phase
                    ^ (
                        (
                            context.pipeline_tile_iteration
                            * (
                                context.multistage_iterations
                                if self.pipeline_advances_per_domain[index]
                                else 1
                            )
                        )
                        // self.pipeline_stage_counts[index]
                    )
                    & 1
                ),
            )
            for index, state in cl.static_iter(
                tuple(enumerate(self.initial_pipeline_states))
            )
        )
        initialized_context = replace(
            context, pipeline_states=pipeline_states
        )
        warp_index = context.warp_index
        if warp_index is None:
            warp_index = _task_warp_index()
        selected = self.warp_start <= warp_index < self.warp_end
        if self.run_only_on_cta_id >= 0:
            selected = selected and (
                cl.block_in_cluster_index(0) == self.run_only_on_cta_id
            )
        if selected:
            if self.num_registers > self.default_num_registers:
                cl.setmaxregister_increase(self.num_registers)
            else:
                cl.setmaxregister_decrease(self.num_registers)
            route_depth = len(initialized_context.route_values)
            result = _run_device_nodes(self.body, initialized_context)
            return result.pop_routes(len(result.route_values) - route_depth)
        return initialized_context


def _forward_tasks_inputs(values, warp_index):
    return values


def _no_resource_finalizer(values, warp_index) -> None:
    return None


def _forward_initialized_resources(values, warp_index):
    return values


def _no_barrier_arena(bindings, size, dynamic, warp_index):
    return None


def _no_smem_base(size_bytes, alignment, dynamic):
    return None


def _create_smem_base(size_bytes, alignment, dynamic):
    return cl.shared_array(
        size_bytes,
        cl.uint8,
        alignment=alignment,
        dynamic=dynamic,
    )


def smem_base_factory_for(size_bytes: int):
    """Select SMEM allocation outside compiler-visible device control flow."""
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("SMEM arena size must be a nonnegative integer")
    return _create_smem_base if size_bytes else _no_smem_base


@dataclass(frozen=True)
class _DeviceTmemState:
    storage: object


def _no_tmem_state(columns, dynamic, warp_index):
    return None


def _create_tmem_state_cta1(columns, dynamic, warp_index):
    storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
        dynamic=dynamic,
    )
    return _DeviceTmemState(storage)


def _create_tmem_state_cta2(columns, dynamic, warp_index):
    storage = cl.shared_array(
        1,
        cl.pointer_dtype(cl.float32, cl.MemorySpace.TENSOR),
        alignment=4,
        dynamic=dynamic,
    )
    return _DeviceTmemState(storage)


def tmem_state_factory_for(columns: int, uses_cluster: bool):
    """Select a manager-owned TMEM metadata layout before device lowering."""
    if type(columns) is not int or columns < 0:
        raise ValueError("TMEM columns must be a nonnegative integer")
    if not columns:
        return _no_tmem_state
    return _create_tmem_state_cta2 if uses_cluster else _create_tmem_state_cta1


def _warp_is_selected(warp_index, selected_warps):
    selected = False
    for selected_warp in cl.static_iter(selected_warps):
        selected = selected or warp_index == selected_warp
    return selected


def _initialize_tmem_state(
    state, columns, uses_cluster, sync_warps, sync_barrier, warp_index
):
    if state is None:
        return None
    cta_group = cl.CTAGroup.CTA_2 if uses_cluster else cl.CTAGroup.CTA_1
    if warp_index == 0:
        cl.tcgen05_allocate(
            state.storage.get_base_pointer(),
            columns,
            cta_group=cta_group,
        )
        cl.tcgen05_relinquish_allocation_permit(cta_group=cta_group)
    if _warp_is_selected(warp_index, sync_warps):
        cl.barrier_sync_block(
            number_of_threads=len(sync_warps) * cl.lane_count(),
            barrier_id=sync_barrier,
        )
    return state.storage[0]


def _finalize_tmem_state(
    state,
    columns,
    uses_cluster,
    barrier_arena,
    dealloc_barrier_offset,
    warp_index,
) -> None:
    if state is None:
        return
    cl.barrier_sync_block()
    if warp_index == 0:
        if uses_cluster:
            dealloc_barrier = barrier_arena.get_element_pointer(
                dealloc_barrier_offset
            )
            peer_rank = cl.block_in_cluster_index(0) ^ 1
            peer_barrier = cl.map_shared_to_cluster(
                dealloc_barrier,
                peer_rank,
            )
            cl.mbarrier_arrive(peer_barrier, scope=cl.MbarrierScope.BLOCK)
            cl.mbarrier_wait_parity(dealloc_barrier, 0)
        cl.tcgen05_deallocate(
            state.storage[0],
            columns,
            cta_group=(
                cl.CTAGroup.CTA_2 if uses_cluster else cl.CTAGroup.CTA_1
            ),
        )


def _initialize_barrier_runs(
    barrier_arena, initialization_runs, arena_size, warp_index
) -> None:
    """Use one warp to initialize the barrier arena in lane-sized chunks."""
    if warp_index == 0:
        lane = cl.lane_index()
        lane_count = cl.lane_count()
        if arena_size % lane_count != 0:
            for begin, end, arrive_count in cl.static_iter(initialization_runs):
                for offset in range(begin + lane, end, lane_count):
                    cl.mbarrier_initialize(
                        barrier_arena.get_element_pointer(offset),
                        arrive_count,
                    )
            return

        # Each instruction site initializes one padded lane-sized barrier
        # chunk. A boundary chunk selects the arrival count per lane; unused
        # padding receives the final run's count and is never addressed.
        for chunk_begin in cl.static_iter(range(0, arena_size, lane_count)):
            offset = chunk_begin + lane
            covering_arrive_count = -1
            for run_index in cl.static_iter(range(len(initialization_runs))):
                begin, end, run_arrive_count = initialization_runs[run_index]
                if begin <= chunk_begin and chunk_begin + lane_count <= end:
                    covering_arrive_count = run_arrive_count
            if covering_arrive_count >= 0:
                arrive_count = covering_arrive_count
            else:
                arrive_count = initialization_runs[-1][2]
                for run_index in cl.static_iter(
                    range(len(initialization_runs) - 2, -1, -1)
                ):
                    _, end, run_arrive_count = initialization_runs[run_index]
                    if offset < end:
                        arrive_count = run_arrive_count
            cl.mbarrier_initialize(
                barrier_arena.get_element_pointer(offset),
                arrive_count,
            )


def _create_barrier_arena(initialization_runs, size, dynamic, warp_index):
    barrier_arena = cl.shared_array(
        size,
        cl.mbarrier,
        alignment=8,
        dynamic=dynamic,
    )
    _initialize_barrier_runs(
        barrier_arena, initialization_runs, size, warp_index
    )
    cl.fence(
        cl.MemoryOrder.RELEASE,
        cl.MemoryScope.CLUSTER,
        restriction=cl.FenceRestriction.mbarrier_initialize(),
    )
    cl.barrier_sync_block()
    return barrier_arena


def _create_cluster_barrier_arena(initialization_runs, size, dynamic, warp_index):
    """Initialize manager barriers and publish them across a CTA cluster."""
    barrier_arena = cl.shared_array(
        size,
        cl.mbarrier,
        alignment=8,
        dynamic=dynamic,
    )
    _initialize_barrier_runs(
        barrier_arena, initialization_runs, size, warp_index
    )
    cl.fence(
        cl.MemoryOrder.RELEASE,
        cl.MemoryScope.CLUSTER,
        restriction=cl.FenceRestriction.mbarrier_initialize(),
    )
    cl.barrier_arrive_cluster(
        aligned=False,
        memory_order=cl.MemoryOrder.RELAXED,
    )
    cl.barrier_wait_cluster(aligned=False)
    return barrier_arena


def barrier_arena_factory_for(size: int, uses_cluster: bool = False):
    """Select zero/nonzero barrier allocation outside device lowering."""
    if type(size) is not int or size < 0:
        raise ValueError("barrier arena size must be a nonnegative integer")
    if not size:
        return _no_barrier_arena
    return _create_cluster_barrier_arena if uses_cluster else _create_barrier_arena


def _create_barrier_storage(size, dynamic):
    return cl.shared_array(
        size,
        cl.mbarrier,
        alignment=8,
        dynamic=dynamic,
    )


def _initialize_prepared_barriers(
    barrier_arena, initialization_runs, arena_size, warp_index, uses_cluster
):
    if barrier_arena is None:
        return
    _initialize_barrier_runs(
        barrier_arena, initialization_runs, arena_size, warp_index
    )
    cl.fence(
        cl.MemoryOrder.RELEASE,
        cl.MemoryScope.CLUSTER,
        restriction=cl.FenceRestriction.mbarrier_initialize(),
    )
    if uses_cluster:
        cl.barrier_arrive_cluster(
            aligned=False,
            memory_order=cl.MemoryOrder.RELAXED,
        )
        cl.barrier_wait_cluster(aligned=False)
    else:
        cl.barrier_sync_block()


def _named_device_offset(
    allocations: tuple[tuple[str, int], ...], name: str
) -> int:
    selected_offset = 0
    found = False
    for allocation_name, offset in cl.static_iter(allocations):
        if allocation_name == name:
            selected_offset = offset
            found = True
    cl.static_assert(found, "unknown named device allocation")
    return selected_offset


@dataclass(frozen=True)
class DeviceSmemAllocator:
    """Device view of one manager-owned, host-laid-out SMEM arena."""

    base: object
    allocations: tuple[tuple[str, int], ...] = ()

    def offset_of(self, name: str) -> int:
        return _named_device_offset(self.allocations, name)

    def get(self, name: str, dtype: object, shape: tuple[int, ...] = (1,)):
        return cl.reinterpret_pointer_as_array(
            self.base.get_base_pointer() + self.offset_of(name),
            dtype,
            shape,
        )


@dataclass(frozen=True)
class DeviceBarrierAllocator:
    """Device view of one manager-owned, host-laid-out mbarrier arena."""

    base: object
    allocations: tuple[tuple[str, int], ...] = ()

    def offset_of(self, name: str) -> int:
        return _named_device_offset(self.allocations, name)

    def get_ptr(self, name: str):
        return self.base.get_element_pointer(self.offset_of(name))


@dataclass(frozen=True)
class DeviceAllocators:
    """Device storage materialized once before task execution."""

    warp_index: object
    smem_allocator: DeviceSmemAllocator
    barrier_allocator: DeviceBarrierAllocator
    cluster_smem_base: object
    tmem_state: object
    tmem_base: object


@dataclass(frozen=True)
class DeviceTaskManager:
    """Frozen device executor preserving a validated manager's task order."""

    tasks: tuple[DeviceTask, ...]
    resource_factory: object = _forward_tasks_inputs
    resource_initializer: object = _forward_initialized_resources
    pipeline_bindings: tuple[DevicePipelineBinding, ...] = ()
    dynamic_pipeline_storage: bool = False
    resource_finalizer: object = _no_resource_finalizer
    barrier_initialization_runs: tuple[tuple[int, int, int], ...] = ()
    barrier_arena_size: int = 0
    barrier_allocation_offsets: tuple[tuple[str, int], ...] = ()
    barrier_arena_factory: object = _no_barrier_arena
    barrier_uses_cluster: bool = False
    smem_size_bytes: int = 0
    smem_alignment: int = 128
    smem_allocation_offsets: tuple[tuple[str, int], ...] = ()
    smem_base_factory: object = _no_smem_base
    tmem_columns: int = 0
    tmem_uses_cluster: bool = False
    tmem_sync_warps: tuple[int, ...] = ()
    tmem_sync_barrier: int = 1
    tmem_dealloc_barrier_offset: int = -1
    tmem_state_factory: object = _no_tmem_state

    def setup_resources_and_tasks(self) -> DeviceAllocators:
        """Materialize manager-owned device allocators before task execution."""
        # Cache the physical warp once. Task dispatch, resource setup, and work
        # callbacks all consume this warp-uniform value instead of emitting
        # independent shuffles at every use site.
        warp_index = _task_warp_index()
        tmem_state = self.tmem_state_factory(
            self.tmem_columns,
            self.dynamic_pipeline_storage,
            warp_index,
        )
        smem_base = self.smem_base_factory(
            self.smem_size_bytes,
            self.smem_alignment,
            self.dynamic_pipeline_storage,
        )
        cluster_smem_base = None
        if self.barrier_uses_cluster and smem_base is not None:
            cluster_smem_base = cl.shfl_sync(
                cl.bitcast(smem_base.get_base_pointer(), cl.uint32), 0
            )
        barrier_arena = (
            _create_barrier_storage(
                self.barrier_arena_size,
                self.dynamic_pipeline_storage,
            )
            if self.barrier_arena_size
            else None
        )
        _initialize_prepared_barriers(
            barrier_arena,
            self.barrier_initialization_runs,
            self.barrier_arena_size,
            warp_index,
            self.barrier_uses_cluster,
        )
        tmem_base = _initialize_tmem_state(
            tmem_state,
            self.tmem_columns,
            self.tmem_uses_cluster,
            self.tmem_sync_warps,
            self.tmem_sync_barrier,
            warp_index,
        )
        return DeviceAllocators(
            warp_index,
            DeviceSmemAllocator(smem_base, self.smem_allocation_offsets),
            DeviceBarrierAllocator(
                barrier_arena, self.barrier_allocation_offsets
            ),
            cluster_smem_base,
            tmem_state,
            tmem_base,
        )

    def _run_with_allocators(
        self,
        inputs: object,
        device_allocators: DeviceAllocators,
        pipeline_tile_iteration: object = 0,
        multistage_iterations: object = 0,
    ) -> None:
        tasks_inputs = self.resource_factory(inputs, device_allocators.warp_index)
        tasks_inputs = self.resource_initializer(
            tasks_inputs, device_allocators.warp_index
        )
        for task in cl.static_iter(self.tasks):
            task(
                task.make_context(
                    tasks_inputs,
                    device_allocators.barrier_allocator.base,
                    device_allocators.smem_allocator.base,
                    device_allocators.cluster_smem_base,
                    device_allocators.tmem_base,
                    device_allocators.warp_index,
                    pipeline_tile_iteration,
                    multistage_iterations,
                )
            )
        self.resource_finalizer(tasks_inputs, device_allocators.warp_index)

    def finalize_resources_and_tasks(
        self, device_allocators: DeviceAllocators
    ) -> None:
        """Release manager-owned state after the final work tile."""
        _finalize_tmem_state(
            device_allocators.tmem_state,
            self.tmem_columns,
            self.tmem_uses_cluster,
            device_allocators.barrier_allocator.base,
            self.tmem_dealloc_barrier_offset,
            device_allocators.warp_index,
        )

    def run(
        self,
        inputs: object,
        device_allocators: DeviceAllocators | None = None,
        pipeline_tile_iteration: object = 0,
        multistage_iterations: object = 0,
    ) -> None:
        """Run tasks with explicit allocators or a one-call managed lifecycle."""
        owns_allocators = device_allocators is None
        if owns_allocators:
            device_allocators = self.setup_resources_and_tasks()
        self._run_with_allocators(
            inputs,
            device_allocators,
            pipeline_tile_iteration,
            multistage_iterations,
        )
        if owns_allocators:
            self.finalize_resources_and_tasks(device_allocators)


_CONSUMER_RESOURCE_STAGES = frozenset(
    {
        ScheduleStage.ConsumerAuxWork,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerRelease,
        ScheduleStage.ConsumerWork,
    }
)
_PRODUCER_RESOURCE_STAGES = frozenset(
    {
        ScheduleStage.ProducerAuxWork,
        ScheduleStage.ProducerTryAcquire,
        ScheduleStage.ProducerAcquire,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ProducerWork,
    }
)


class Task:
    default_num_registers = 128

    def __init__(
        self,
        src_resources: list[MemoryResource] | int | None = None,
        dst_resources: list[MemoryResource] | int | None = None,
        warp_idx: int | None = None,
        num_warps: int | None = None,
        *,
        schedule: Schedule,
        num_registers: int | None = None,
        name: str = "",
        run_only_on_cta_id: int | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        positional_inference = (
            type(src_resources) is int and type(dst_resources) is int
        )
        keyword_inference = src_resources is None and dst_resources is None
        infer_resources = positional_inference or keyword_inference
        if positional_inference:
            if warp_idx is not None or num_warps is not None:
                raise TypeError(
                    "inferred Task construction takes warp_idx and num_warps once"
                )
            warp_idx = src_resources
            num_warps = dst_resources
            explicit_src_resources = None
            explicit_dst_resources = None
        elif keyword_inference:
            explicit_src_resources = None
            explicit_dst_resources = None
        else:
            explicit_src_resources = src_resources
            explicit_dst_resources = dst_resources
            if not isinstance(explicit_src_resources, list) or not isinstance(
                explicit_dst_resources, list
            ):
                raise TypeError("Task resources must be lists")
        if not isinstance(schedule, Schedule):
            raise ValueError("schedule must be a Schedule")
        if type(warp_idx) is not int or warp_idx < 0:
            raise ValueError("warp_idx must be a nonnegative integer")
        if type(num_warps) is not int or num_warps <= 0:
            raise ValueError("num_warps must be a positive integer")
        if num_registers is not None and (
            type(num_registers) is not int
            or not 8 <= num_registers <= 256
            or num_registers % 8
        ):
            raise ValueError("num_registers must be in [8, 256] and divisible by 8")
        self.name = name or schedule.name
        consumer_ids = set()
        producer_ids = set()
        used_ids = set()
        for node in _iter_nodes(schedule.body):
            if isinstance(node, Step):
                resource_id = id(node.memory_resource)
                used_ids.add(resource_id)
                if node.schedule_stage in _CONSUMER_RESOURCE_STAGES:
                    consumer_ids.add(resource_id)
                elif node.schedule_stage in _PRODUCER_RESOURCE_STAGES:
                    producer_ids.add(resource_id)
                else:
                    raise ValueError(
                        f"Task {self.name!r} cannot infer an access role for "
                        f"schedule stage {node.schedule_stage!r}"
                    )
            elif isinstance(node, WorkTileLoop):
                resource_id = id(node.work_queue)
                used_ids.add(resource_id)
                consumer_ids.add(resource_id)
        inferred_resources = [
            resource for resource in schedule.resources if id(resource) in used_ids
        ]
        inferred_consumers = [
            resource for resource in inferred_resources if id(resource) in consumer_ids
        ]
        inferred_producers = [
            resource for resource in inferred_resources if id(resource) in producer_ids
        ]
        if infer_resources:
            self.src_resources = inferred_consumers
            self.dst_resources = inferred_producers
            self.resources = inferred_resources
        else:
            self.src_resources = list(explicit_src_resources)
            self.dst_resources = list(explicit_dst_resources)
            self.resources = list(
                dict.fromkeys(explicit_src_resources + explicit_dst_resources)
            )
        self.consumer_resources = inferred_consumers
        self.producer_resources = inferred_producers
        self.warp_idx = warp_idx
        self.num_warps = num_warps
        self.schedule = schedule
        self.num_registers = num_registers
        self.run_only_on_cta_id = run_only_on_cta_id
        validate_queue_advance_placement(schedule.body)
        allowed = {id(resource) for resource in self.resources}
        for node in _iter_nodes(schedule.body):
            if isinstance(node, Step) and id(node.memory_resource) not in allowed:
                raise ValueError(
                    f"Task {self.name!r} schedule uses undeclared resource "
                    f"{node.memory_resource.name!r}"
                )

    @property
    def warp_start(self) -> int:
        return self.warp_idx

    @property
    def warp_end(self) -> int:
        return self.warp_idx + self.num_warps

    def __call__(self, context: ExecutionContext | None = None) -> ExecutionContext:
        """Lower the immutable tree when called from a ``cuda.lang`` kernel."""
        warp_index = _task_warp_index()
        selected = self.warp_start <= warp_index < self.warp_end
        if self.run_only_on_cta_id is not None:
            selected = (
                selected and cl.block_in_cluster_index(0) == self.run_only_on_cta_id
            )
        if selected:
            registers = self.num_registers or self.default_num_registers
            if registers > self.default_num_registers:
                cl.setmaxregister_increase(registers)
            else:
                cl.setmaxregister_decrease(registers)
            return self._run_nodes(self.schedule.body, context or ExecutionContext())
        return context or ExecutionContext()

    def to_device(
        self,
        callbacks: dict[object, object] | None = None,
        *,
        pipeline_bindings: dict[MemoryResource, DevicePipelineBinding] | None = None,
        route_values: bool = True,
        pass_stage_info: bool = True,
    ) -> DeviceTask:
        """Create the frozen CUDA Lang program for this validated tree.

        Directly decorated static work methods are inferred as callbacks.
        Explicit callbacks override them and are keyed by captured work label.
        A qualified ``"resource.label"`` key or ``(resource, label)`` pair
        disambiguates a label reused by two resources. Integer step IDs are
        accepted only as a compatibility fallback; they are not part of
        ordinary task authoring.
        """
        from .enums import Every, FIRST_ITER, LAST_ITER

        callbacks = callbacks or {}
        pipeline_bindings = pipeline_bindings or {}
        pipeline_resources = []
        for node in _iter_nodes(self.schedule.body):
            if (
                isinstance(node, Step)
                and node.memory_resource.pipeline_config is not None
                and node.memory_resource not in pipeline_resources
            ):
                pipeline_resources.append(node.memory_resource)
        pipeline_slots = {
            resource: slot for slot, resource in enumerate(pipeline_resources)
        }
        resolved_pipeline_bindings = {}
        next_barrier_offset = 0
        for resource in pipeline_resources:
            binding = pipeline_bindings.get(resource)
            if binding is not None and not binding.has_barrier_offsets:
                binding = binding.at_offsets(
                    next_barrier_offset,
                    next_barrier_offset + binding.num_stages,
                )
            if binding is not None:
                resolved_pipeline_bindings[resource] = binding
            next_barrier_offset += 2 * resource.pipeline_config.num_stages
        initial_pipeline_states = tuple(
            PipelineState(
                phase=1 if any(resource is item for item in self.dst_resources) else 0
            )
            for resource in pipeline_resources
        )
        used_callback_keys = set()

        def resolve_callback(step):
            label = step.label or step.schedule_stage.value
            candidates = (
                (step.memory_resource, label),
                f"{step.memory_resource.name}.{label}",
                label,
                step.unique_id,
            )
            for key in candidates:
                if key in callbacks:
                    used_callback_keys.add(key)
                    return callbacks[key]
            static_work_fn = _get_static_work_fn(step.memory_resource, label)
            if static_work_fn is not None:
                return static_work_fn
            raise ValueError(
                f"Task {self.name!r} has no device callback for "
                f"{step.memory_resource.name}.{label}; pass an explicit callback "
                "or define the work method as a directly decorated static method"
            )

        def anonymous_index(value, anonymous_routes):
            try:
                return anonymous_routes.index(value)
            except ValueError as error:
                raise ValueError(
                    f"Task {self.name!r} uses routed value %{value.value_id} "
                    "outside its defining scope"
                ) from error

        def lower_guard(guard, anonymous_routes):
            if guard is FIRST_ITER:
                return DeviceGuard("first", 1, 0, -1, False)
            if guard is LAST_ITER:
                return DeviceGuard("last", 1, 0, -1, False)
            if isinstance(guard, Every):
                return DeviceGuard("every", guard.period, guard.start, -1, False)
            if guard is SKIPPABLE:
                return DeviceGuard("skippable", 1, 0, -1, False)
            if isinstance(guard, OpaqueCondition):
                value = guard.route_token
                if not isinstance(value, ScheduleValue):
                    raise NotImplementedError(
                        "literal opaque guards need a device condition adapter"
                    )
                return DeviceGuard(
                    "opaque",
                    1,
                    0,
                    anonymous_index(value, anonymous_routes),
                    guard.negated,
                )
            raise TypeError(type(guard).__name__)

        def anonymous_uses(node):
            if isinstance(node, Step):
                return set(node.input_values.values())
            uses = set()
            if isinstance(node, DomainLoop):
                uses.update(node.initial_values.values())
                uses.update(node.yield_values.values())
            if isinstance(node, ConditionalBlock) and isinstance(
                node.condition, OpaqueCondition
            ):
                value = node.condition.route_token
                if isinstance(value, ScheduleValue):
                    uses.add(value)
            for child in node.body:
                uses.update(anonymous_uses(child))
            return uses

        def dead_trailing_count(routes, future_uses, frame_depth):
            count = 0
            while (
                len(routes) - count > frame_depth
                and routes[len(routes) - count - 1] not in future_uses
            ):
                count += 1
            return count

        def lower_nodes(
            nodes,
            anonymous_routes,
            *,
            in_domain_loop=False,
            frame_depth=None,
            terminal_uses=frozenset(),
        ):
            if frame_depth is None:
                frame_depth = len(anonymous_routes)
            suffix_uses = [set() for _ in range(len(nodes) + 1)]
            suffix_uses[-1].update(terminal_uses)
            for index in range(len(nodes) - 1, -1, -1):
                suffix_uses[index] = suffix_uses[index + 1] | anonymous_uses(
                    nodes[index]
                )

            lowered = []
            for index, child in enumerate(nodes):
                old_depth = len(anonymous_routes)
                device_node = lower(
                    child,
                    anonymous_routes,
                    in_domain_loop=in_domain_loop,
                )
                future_uses = suffix_uses[index + 1]
                if isinstance(device_node, DeviceStep):
                    appended_routes = anonymous_routes[old_depth:]
                    old_routes = anonymous_routes[:old_depth]
                    release_before = dead_trailing_count(
                        old_routes, future_uses, frame_depth
                    )
                    if release_before:
                        old_routes = old_routes[:-release_before]
                    release_after = dead_trailing_count(
                        appended_routes, future_uses, 0
                    )
                    if release_after:
                        appended_routes = appended_routes[:-release_after]
                    anonymous_routes[:] = old_routes + appended_routes
                    device_node = replace(
                        device_node,
                        release_before_append=release_before,
                        release_after_append=release_after,
                    )
                elif isinstance(
                    device_node, (DeviceConditional, DeviceDomainLoop)
                ):
                    release_count = dead_trailing_count(
                        anonymous_routes, future_uses, frame_depth
                    )
                    if release_count:
                        del anonymous_routes[-release_count:]
                    device_node = replace(
                        device_node, release_count=release_count
                    )
                lowered.append(device_node)
            return tuple(lowered)

        def lower(node, anonymous_routes, *, in_domain_loop=False):
            if isinstance(node, Step):
                stage_actions = {
                    ScheduleStage.ProducerTryAcquire: DevicePipelineStep.TRY_ACQUIRE,
                    ScheduleStage.ProducerAcquire: DevicePipelineStep.ACQUIRE,
                    ScheduleStage.ProducerCommit: DevicePipelineStep.COMMIT,
                    ScheduleStage.ConsumerTryWait: DevicePipelineStep.TRY_WAIT,
                    ScheduleStage.ConsumerWait: DevicePipelineStep.WAIT,
                    ScheduleStage.ConsumerRelease: DevicePipelineStep.RELEASE,
                }
                if node.schedule_stage in stage_actions:
                    binding = resolved_pipeline_bindings.get(node.memory_resource)
                    if binding is None:
                        raise ValueError(
                            f"Task {self.name!r} needs a DevicePipelineBinding "
                            f"for resource {node.memory_resource.name!r}"
                        )
                    return DevicePipelineStep(
                        stage_actions[node.schedule_stage],
                        binding,
                        pipeline_slots[node.memory_resource],
                        node.unique_id,
                    )
                callback = resolve_callback(node)
                input_slots = tuple(
                    anonymous_index(value, anonymous_routes)
                    for value in node.input_values.values()
                )
                static_args = tuple(node.constexpr_kwargs.values())
                input_indices = {
                    name: index for index, name in enumerate(node.input_values)
                }
                static_indices = {
                    name: len(input_slots) + index
                    for index, name in enumerate(node.constexpr_kwargs)
                }
                argument_order = tuple(
                    input_indices[name]
                    if name in input_indices
                    else static_indices[name]
                    for name in node.argument_order
                )
                outputs = node.output_values
                if outputs and not route_values:
                    raise ValueError(
                        f"Task {self.name!r} cannot disable routing with "
                        "work outputs"
                    )
                stage_resource = node.stage_resource or node.memory_resource
                if stage_resource not in pipeline_slots:
                    input_stage_resources = []
                    for value in node.input_values.values():
                        owner = value.stage_resource
                        if (
                            owner is not None
                            and owner in pipeline_slots
                            and owner not in input_stage_resources
                        ):
                            input_stage_resources.append(owner)
                    if len(input_stage_resources) > 1:
                        names = ", ".join(
                            resource.name for resource in input_stage_resources
                        )
                        step_label = node.label or node.schedule_stage.value
                        raise ValueError(
                            f"Task {self.name!r} cannot infer one pipeline stage for "
                            f"{node.memory_resource.name}.{step_label} "
                            f"from inputs owned by {names}"
                        )
                    if len(input_stage_resources) == 1:
                        stage_resource = input_stage_resources[0]
                device_step = DeviceStep(
                    callback=callback,
                    unique_id=node.unique_id,
                    static_args=static_args,
                    argument_order=argument_order,
                    pipeline_slot=pipeline_slots.get(stage_resource, -1),
                    input_slots=input_slots,
                    append_output_count=len(outputs),
                    automatic_routing=route_values,
                    pipeline_binding=resolved_pipeline_bindings.get(stage_resource),
                    pass_stage_info=pass_stage_info,
                    label=node.label,
                    in_domain_loop=in_domain_loop,
                )
                anonymous_routes.extend(outputs)
                return device_step
            if isinstance(node, ConditionalBlock):
                body_routes = list(anonymous_routes)
                body_frame_depth = len(body_routes)
                return DeviceConditional(
                    lower_nodes(
                        node.body,
                        body_routes,
                        in_domain_loop=in_domain_loop,
                        frame_depth=body_frame_depth,
                    ),
                    lower_guard(node.condition, anonymous_routes),
                )
            if isinstance(node, DomainLoop):
                if not all(
                    type(value) is int or isinstance(value, DynamicDomainBound)
                    for value in (node.start, node.end)
                ):
                    raise NotImplementedError(
                        "dynamic domain bounds must use dynamic_domain_bound()"
                    )
                if type(node.step) is not int or node.step <= 0:
                    raise NotImplementedError(
                        "device domain loops require a positive static step"
                    )
                static_bounds = all(
                    type(value) is int for value in (node.start, node.end)
                )
                dynamic_start = isinstance(node.start, DynamicDomainBound)
                dynamic_end = isinstance(node.end, DynamicDomainBound)
                initial_positions = tuple(
                    anonymous_index(variable, anonymous_routes)
                    for variable in node.initial_values.values()
                )
                body_routes = list(anonymous_routes)
                body_routes.extend(node.iter_values.values())
                body_frame_depth = len(body_routes)
                body = lower_nodes(
                    node.body,
                    body_routes,
                    in_domain_loop=True,
                    frame_depth=body_frame_depth,
                    terminal_uses=frozenset(node.yield_values.values()),
                )
                yield_positions = tuple(
                    anonymous_index(variable, body_routes)
                    for variable in node.yield_values.values()
                )
                result_values = tuple(node.result_values.values())
                anonymous_routes.extend(result_values)
                return DeviceDomainLoop(
                    node.start if not dynamic_start else 0,
                    node.end if not dynamic_end else 0,
                    node.step,
                    (
                        len(range(node.start, node.end, node.step))
                        if static_bounds
                        else -1
                    ),
                    body,
                    dynamic_start,
                    dynamic_end,
                    node.start.resolver if dynamic_start else None,
                    node.end.resolver if dynamic_end else None,
                    initial_route_positions=initial_positions,
                    yield_route_positions=yield_positions,
                    result_indices=tuple(range(len(result_values))),
                )
            raise NotImplementedError(
                "WorkTileLoop needs a scheduler-specific device adapter"
            )

        def advances_per_domain(nodes, resource, in_domain=False):
            for node in nodes:
                if isinstance(node, Step):
                    if (
                        node.memory_resource is resource
                        and node.schedule_stage
                        in (
                            ScheduleStage.ProducerCommit,
                            ScheduleStage.ConsumerRelease,
                        )
                    ):
                        return in_domain
                elif isinstance(node, DomainLoop):
                    if advances_per_domain(node.body, resource, True):
                        return True
                elif isinstance(node, ConditionalBlock):
                    if advances_per_domain(node.body, resource, in_domain):
                        return True
            return False

        device_task = DeviceTask(
            body=lower_nodes(self.schedule.body, []),
            warp_start=self.warp_start,
            warp_end=self.warp_end,
            num_registers=self.num_registers or self.default_num_registers,
            run_only_on_cta_id=(
                self.run_only_on_cta_id
                if self.run_only_on_cta_id is not None
                else -1
            ),
            initial_pipeline_states=initial_pipeline_states,
            default_num_registers=self.default_num_registers,
            pipeline_stage_counts=tuple(
                resource.pipeline_config.num_stages
                for resource in pipeline_resources
            ),
            pipeline_advances_per_domain=tuple(
                advances_per_domain(self.schedule.body, resource)
                for resource in pipeline_resources
            ),
        )
        unused_callback_keys = set(callbacks) - used_callback_keys
        if unused_callback_keys:
            raise ValueError(
                f"Task {self.name!r} has unused device callback keys: "
                f"{sorted(map(str, unused_callback_keys))}"
            )
        return device_task

    def _run_nodes(self, nodes, context: ExecutionContext) -> ExecutionContext:
        for node in cl.static_iter(nodes):
            context = self._run_node(node, context)
        return context

    def _run_node(self, node: Node, context: ExecutionContext) -> ExecutionContext:
        if isinstance(node, Step):
            return self._run_step(node, context)
        if isinstance(node, DomainLoop):
            start = node.start
            if isinstance(start, DynamicDomainBound):
                start = start.resolve(context.tasks_inputs)
            end = node.end
            if isinstance(end, DynamicDomainBound):
                end = end.resolve(context.tasks_inputs)
            stride = node.step
            distance = end - start
            num_iterations = (distance + stride - 1) // stride if distance > 0 else 0
            outer_loop_state = (
                context.loop_offset,
                context.iteration_index,
                context.num_iterations,
                context.loop_start,
                context.loop_end,
                context.loop_step,
                context.in_domain_loop,
            )
            for loop_offset in range(start, end, stride):
                iteration_index = (loop_offset - start) // stride
                context = replace(
                    context,
                    loop_offset=loop_offset,
                    iteration_index=iteration_index,
                    num_iterations=num_iterations,
                    loop_start=start,
                    loop_end=end,
                    loop_step=stride,
                    in_domain_loop=True,
                )
                context = self._run_nodes(node.body, context)
            return replace(
                context,
                loop_offset=outer_loop_state[0],
                iteration_index=outer_loop_state[1],
                num_iterations=outer_loop_state[2],
                loop_start=outer_loop_state[3],
                loop_end=outer_loop_state[4],
                loop_step=outer_loop_state[5],
                in_domain_loop=outer_loop_state[6],
            )
        if isinstance(node, ConditionalBlock):
            active = self._guard_active(node.condition, context)
            if active:
                return self._run_nodes(node.body, context)
            return context
        if isinstance(node, WorkTileLoop):
            raise NotImplementedError(
                "dynamic WorkQueue device lowering requires a scheduler-specific "
                "work-tile adapter"
            )
        raise TypeError(type(node).__name__)

    def _guard_active(self, guard: object, context: ExecutionContext) -> object:
        from .enums import IterationPredicate

        if guard is SKIPPABLE:
            return not context.skipped
        if isinstance(guard, OpaqueCondition):
            raise NotImplementedError(
                "opaque guards require frozen device lowering"
            )
        if isinstance(guard, IterationPredicate):
            # Predicates contain only host constants; CUDA Lang specializes them.
            from .enums import Every, FIRST_ITER, LAST_ITER

            if guard is FIRST_ITER:
                return context.iteration_index == 0
            if guard is LAST_ITER:
                return context.iteration_index == context.num_iterations - 1
            if isinstance(guard, Every):
                return (
                    context.iteration_index >= guard.start
                    and (context.iteration_index - guard.start) % guard.period == 0
                )
        raise TypeError(type(guard).__name__)

    def _run_step(self, step: Step, context: ExecutionContext) -> ExecutionContext:
        resource = step.memory_resource
        config = resource.pipeline_config
        if config is not None:
            require_device_support(config)
        cl.ptx_comment(
            f"task_scheduling #{step.unique_id} {resource.name}."
            f"{step.label or step.schedule_stage.value}"
        )
        stage_info = StageInfo(
            count=context.iteration_index,
            loop_offset=context.loop_offset if context.in_domain_loop else None,
            loop_start=context.loop_start if context.in_domain_loop else None,
            loop_end=context.loop_end if context.in_domain_loop else None,
            loop_step=context.loop_step if context.in_domain_loop else None,
            label=step.label,
            work_tile=context.work_tile,
            context=context,
        )
        if step.schedule_stage not in (
            ScheduleStage.ConsumerWork,
            ScheduleStage.ConsumerAuxWork,
            ScheduleStage.ProducerWork,
            ScheduleStage.ProducerAuxWork,
        ):
            lower = getattr(resource, "lower_pipeline_stage", None)
            if lower is None:
                return context
            lower(step.schedule_stage, stage_info)
            return context
        if step.label is None:
            method = getattr(resource, step.schedule_stage.value)
        else:
            method = getattr(resource, step.label)
        if step.input_values or step.output_values:
            raise NotImplementedError(
                "routed work values require frozen device lowering"
            )
        method(stage_info, **step.constexpr_kwargs)
        return context
