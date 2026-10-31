# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TMA copy whose store task exits through ``while_true_loop`` (Tutorial 04).

LoadTask keeps the ordinary ``domain_loop`` TMA producer from kernel 02/03.
StoreTask consumes rows through a ``while_true_loop`` whose cursor comes from
``retry_init`` / ``retry_advance``.  The exit formula is **only** the
dominating guard of ``break_loop()``::

    rows_done = smem.rows_done()
    with ts.when_true(rows_done):
        trace.mark_exit()
        loop.break_loop()
        trace.mark_after_break()
    with ts.when_false(rows_done, verify_num_fires=1):
        smem.try_wait() / wait / read / release / store
        trace.mark_iteration()

The verifier is deliberately offered ``verify_num_iters=(2, 3, 4)``.  Because
``num_rows`` is a runtime value, the load task expands to one representative
pipeline round.  ``verify_num_fires=1`` on the guarded wait/read/release path
therefore retains only the matching one-retry-plus-break history.  Without the
fire-count assertion, the extra modeled histories consume pipeline stages that
the representative producer never commits and report false deadlocks.

Host checks the copied matrix and a three-column trace: one iteration marker
per row, one exit marker on the last row, and an all-zero sentinel column that
proves work recorded after ``break_loop()`` did not execute.

CLI extras (host-only schedule capture, no extra TaskManager over load+store):

* ``--verify-num-iters {1,2,3}`` prints retry vs break assignment sequences
* ``--inject-nested-break`` wraps ``break_loop()`` in ``when_true(is_highlight)``
  so the printed exit formula becomes ``rows_done and highlight``
* ``--inject-unconditional-break`` with modeled count 2 shows the
  "no retry assignment" builder error
"""

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch

import task_scheduling as ts
from task_scheduling.exhaustive_checker import (
    collect_opaque_keys,
    format_opaque_assignment,
)


WARP_SIZE = 32
BLOCK_THREADS = 8 * WARP_SIZE
num_stages = 1
tile_size = 128
TILE_BYTES = tile_size * 2
FP16_BYTES = cl.float16.bitwidth // 8
STORE_TASK_WARP_IDX = 0
LOAD_TASK_WARP_IDX = 4
PADDING_TASK_WARP_IDX = 5
NUM_WARPS_EPILOGUE = 4

ITERATION_MARKER = 1000.0
EXIT_MARKER = 4000.0
AFTER_BREAK_MARKER = 8000.0

# This program owns one SMEM allocation, so its view begins at the arena base.
SMEM_VIEW_OFFSET_BYTES = 0
SMEM_VIEW_ELEMENTS = num_stages * tile_size


def _expect_trace_column(
    trace: torch.Tensor,
    *,
    column: int,
    marker: float,
    expected_rows: list[int],
    label: str,
) -> None:
    """Require a trace marker column to match exactly the expected rows."""
    actual = trace[:, column]
    expected = torch.zeros_like(actual)
    if expected_rows:
        expected[expected_rows] = marker

    if torch.equal(actual, expected):
        return

    mismatch_rows = torch.nonzero(actual != expected, as_tuple=False).flatten()
    row_details = []
    for row in mismatch_rows[:8].cpu().tolist():
        row_details.append(
            f"row {row}: expected {expected[row].item()}, got {actual[row].item()}"
        )
    extra_count = max(0, mismatch_rows.numel() - len(row_details))
    suffix = f"; plus {extra_count} more mismatch(es)" if extra_count else ""
    raise RuntimeError(
        f"{label} trace column {column} does not match expected rows "
        f"{expected_rows}: {', '.join(row_details)}{suffix}"
    )


# -----------------------------------------------------------------------------
# Resource definitions
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class InputGmemResource(ts.MemoryResource):
    """Coordinate-only GMEM source: emits the TMA load column coordinate."""

    @ts.consumer_work(outputs=1)
    @staticmethod
    def compute_coords(stage_info):
        return cl.block_index(0) * tile_size


def _is_store_trace_writer():
    """Select the store task's first warp in CTA 0; callers then elect a lane.

    Broadcasting lane 0's warp index makes the predicate warp-uniform, so the
    following ``elect_sync`` needs no divergence handling.
    """
    warp_idx, _ = cl.shuffle_sync(
        cl.ShuffleKind.INDEX, cl.thread_index(0) // WARP_SIZE, 0
    )
    return cl.block_index(0) == 0 and warp_idx == STORE_TASK_WARP_IDX


@dataclass(kw_only=True, eq=False)
class TraceGmemResource(ts.MemoryResource):
    """Side-channel trace including an after-break sentinel column."""

    @ts.producer_work
    @staticmethod
    def mark_iteration(stage_info):
        if _is_store_trace_writer():
            if cl.elect_sync():
                trace = stage_info.context.tasks_inputs.trace
                trace[stage_info.loop_offset, 0] = ITERATION_MARKER

    @ts.producer_work
    @staticmethod
    def mark_exit(stage_info):
        if _is_store_trace_writer():
            if cl.elect_sync():
                values = stage_info.context.tasks_inputs
                values.trace[values.num_rows - 1, 1] = EXIT_MARKER

    @ts.producer_work
    @staticmethod
    def mark_after_break(stage_info):
        """Mark forbidden fallthrough after ``break_loop()`` for the test."""
        if _is_store_trace_writer():
            if cl.elect_sync():
                values = stage_info.context.tasks_inputs
                values.trace[values.num_rows - 1, 2] = AFTER_BREAK_MARKER


@dataclass(kw_only=True, eq=False)
class SmemResource(ts.MemoryResource):
    """Pipelined SMEM tile filled by TMA, consumed by the epilogue warps."""

    @staticmethod
    def _init_smem_state(stage_info):
        """Create the typed view into the manager-owned SMEM allocation."""
        smem_base = stage_info.context.smem_base
        smem_pointer = smem_base.pointer() + SMEM_VIEW_OFFSET_BYTES
        typed_pointer = cl.bitcast(
            smem_pointer,
            cl.pointer_dtype(cl.float16, smem_pointer.memory_space),
        )
        return cl.Array.from_parts(typed_pointer, SMEM_VIEW_ELEMENTS)

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_load_state(stage_info):
        return SmemResource._init_smem_state(stage_info)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_read_state(stage_info):
        return SmemResource._init_smem_state(stage_info)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def retry_init(stage_info):
        return cl.int32(0)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def retry_advance(stage_info):
        return stage_info.loop_offset + 1

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def rows_done(stage_info):
        return stage_info.loop_offset >= stage_info.context.tasks_inputs.num_rows

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def is_highlight_tile(stage_info):
        # Used only by the host-only --inject-nested-break schedule capture.
        return stage_info.loop_offset == stage_info.context.tasks_inputs.highlight_row

    @ts.consumer_work(outputs=1)
    @staticmethod
    def read_smem(stage_info, smem_view):
        return smem_view[stage_info.stage_idx * tile_size + cl.thread_index(0)]

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, smem_view, gmem_idx):
        if cl.elect_sync():
            cl.copy_async_bulk_tensor_global_to_shared(
                stage_info.context.tasks_inputs.tensor_map,
                (gmem_idx, stage_info.loop_offset),
                smem_view.pointer(stage_info.stage_idx * tile_size),
                stage_info.barrier,
            )


@dataclass(kw_only=True, eq=False)
class OutputGmemResource(ts.MemoryResource):
    """GMEM sink: stores the SMEM-loaded value into matrix B."""

    @ts.producer_work
    @staticmethod
    def store(stage_info, smem_val):
        m_idx = cl.thread_index(0) + cl.block_index(0) * tile_size
        stage_info.context.tasks_inputs.output[stage_info.loop_offset, m_idx] = smem_val


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------


@ts.schedule
def load_schedule(stage_info, input_gmem, smem):
    smem_view = smem.init_load_state()

    def loop_body():
        gmem_idx = input_gmem.compute_coords()
        smem.try_acquire()
        smem.acquire()
        smem.tma_load(smem_view, gmem_idx)
        smem.commit()

    ts.domain_loop(0, stage_info.context.tasks_inputs.num_rows, 1, loop_body)


def _store_schedule(
    *,
    verify_num_iters=(2, 3, 4),
    verify_num_fires=1,
    inject_nested_break=False,
    inject_unconditional_break=False,
):
    """Build the store schedule; the keyword options feed the CLI showcase."""

    @ts.schedule
    def store_schedule(smem, output_gmem, trace):
        smem_view = smem.init_read_state()
        with ts.while_true_loop(
            init=smem.retry_init,
            advance=smem.retry_advance,
            verify_num_iters=verify_num_iters,
        ) as loop:
            if inject_unconditional_break:
                loop.break_loop()
                return
            rows_done = smem.rows_done()
            with ts.when_true(rows_done):
                trace.mark_exit()
                if inject_nested_break:
                    highlight = smem.is_highlight_tile()
                    with ts.when_true(highlight):
                        loop.break_loop()
                else:
                    loop.break_loop()
                trace.mark_after_break()
            with ts.when_false(rows_done, verify_num_fires=verify_num_fires):
                smem.try_wait()
                smem.wait()
                smem_val = smem.read_smem(smem_view)
                smem.release()
                output_gmem.store(smem_val)
                trace.mark_iteration()

    return store_schedule


@ts.schedule
def padding_schedule(stage_info):
    ts.domain_loop(0, stage_info.context.tasks_inputs.num_rows, 1, lambda: None)


# -----------------------------------------------------------------------------
# Host-only sequence showcase
# -----------------------------------------------------------------------------


def print_while_assignment_sequences(
    *,
    verify_num_iters=2,
    verify_num_fires=None,
    inject_nested_break=False,
    inject_unconditional_break=False,
):
    """Capture the store while-loop on the host and print retry/break sequences."""
    smem = SmemResource(name="smem")
    output_gmem = OutputGmemResource(name="output")
    trace = TraceGmemResource(name="trace")
    try:
        built = _store_schedule(
            verify_num_iters=verify_num_iters,
            verify_num_fires=verify_num_fires,
            inject_nested_break=inject_nested_break,
            inject_unconditional_break=inject_unconditional_break,
        )(smem, output_gmem, trace)
    except ts.ScheduleError as exc:
        print(f"ScheduleError: {exc}")
        return
    print(built)
    store_task = ts.Task(
        src_resources=[smem],
        dst_resources=[output_gmem, trace],
        warp_idx=STORE_TASK_WARP_IDX,
        num_warps=1,
        schedule=built,
        name="StoreTask",
    )
    keys = collect_opaque_keys([store_task])
    scenarios = ts.opaque_assignment_sequences([store_task], keys)
    print(f"opaque keys: {keys}")
    print(f"{len(scenarios)} modeled assignment sequence(s):")
    for index, scenario in enumerate(scenarios):
        print(f"  [{index}] {format_opaque_assignment(scenario.for_task(store_task))}")


# -----------------------------------------------------------------------------
# Program and kernel
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TasksInputs:
    tensor_map: object
    output: object
    trace: object
    num_rows: object


@dataclass(frozen=True)
class TmaCopyWhileExitProgram:
    """Host model and frozen device manager for one kernel specialization."""

    manager: object
    device_manager: object


def make_tma_copy_while_exit_program(*, verbose=True):
    """Construct and validate the complete task-scheduling program."""
    pipeline_config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        num_stages=num_stages,
        num_bytes=TILE_BYTES,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(NUM_WARPS_EPILOGUE * WARP_SIZE),
    )
    input_gmem_resource = InputGmemResource(name="inputGmemResource")
    trace_gmem_resource = TraceGmemResource(name="traceGmemResource")
    smem_allocation = ts.SmemAllocation(
        "smem_data", num_stages * TILE_BYTES, alignment=128
    )
    smem_resource = SmemResource(
        name="smemResource",
        pipeline_config=pipeline_config,
        smem_requirements=[smem_allocation],
    )
    output_gmem_resource = OutputGmemResource(name="outputGmemResource")

    allocator = ts.SmemAllocator()
    allocator.add_resource(smem_resource)
    allocator.compute_layout()
    if smem_allocation.offset != SMEM_VIEW_OFFSET_BYTES:
        raise ValueError("SMEM view offset does not match its allocator layout")
    if smem_allocation.size_bytes // FP16_BYTES != SMEM_VIEW_ELEMENTS:
        raise ValueError("SMEM view size does not match its allocator layout")

    load_task = ts.Task(
        src_resources=[input_gmem_resource],
        dst_resources=[smem_resource],
        warp_idx=LOAD_TASK_WARP_IDX,
        num_warps=1,
        schedule=load_schedule(input_gmem_resource, smem_resource),
        num_registers=40,
        name="LoadTask",
    )
    store_task = ts.Task(
        src_resources=[smem_resource],
        dst_resources=[output_gmem_resource, trace_gmem_resource],
        warp_idx=STORE_TASK_WARP_IDX,
        num_warps=NUM_WARPS_EPILOGUE,
        schedule=_store_schedule()(
            smem_resource, output_gmem_resource, trace_gmem_resource
        ),
        num_registers=160,
        name="StoreTask",
    )
    padding_task = ts.Task(
        src_resources=[],
        dst_resources=[],
        warp_idx=PADDING_TASK_WARP_IDX,
        num_warps=3,
        schedule=padding_schedule(),
        num_registers=40,
        name="PaddingTask",
    )

    resource_dependency_graph = {
        smem_resource: [input_gmem_resource],
        output_gmem_resource: [smem_resource],
        trace_gmem_resource: [smem_resource],
    }
    manager = ts.TaskManager(
        tasks=[load_task, store_task, padding_task],
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=allocator,
        exhaustive_deadlock_race_check=True,
        exhaustive_representative_domain=True,
        verbose=verbose,
    )
    return TmaCopyWhileExitProgram(manager, manager.to_device())


def make_tma_copy_while_exit_kernel(device_manager):
    """Specialize the kernel for one frozen task manager."""

    @cl.kernel
    def tma_copy_while_exit_kernel(tensor_map, output, trace, num_rows):
        if cl.elect_sync():
            cl.prefetch_tensor_map(tensor_map)
        device_allocators = device_manager.setup_resources_and_tasks()
        device_manager.run(
            TasksInputs(tensor_map, output, trace, num_rows),
            device_allocators,
        )

    return tma_copy_while_exit_kernel


# -----------------------------------------------------------------------------
# Host runners and CLI
# -----------------------------------------------------------------------------


def _verify_trace_markers(trace: torch.Tensor, *, num_rows: int) -> None:
    """Check iteration/exit markers and prove break has no fallthrough."""
    if trace.shape != (num_rows, 3):
        raise RuntimeError(
            f"expected trace shape {(num_rows, 3)}, got {tuple(trace.shape)}"
        )
    _expect_trace_column(
        trace,
        column=0,
        marker=ITERATION_MARKER,
        expected_rows=list(range(num_rows)),
        label="iteration",
    )
    _expect_trace_column(
        trace,
        column=1,
        marker=EXIT_MARKER,
        expected_rows=[num_rows - 1],
        label="exit",
    )
    _expect_trace_column(
        trace,
        column=2,
        marker=AFTER_BREAK_MARKER,
        expected_rows=[],
        label="after-break",
    )


def run_tma_copy_while_exit_kernel_prim(rows_cols, highlight_row=0, *, verbose=True):
    if verbose:
        print("===================================================================")
        print("Running while-exit TMA copy kernel with:")
        print(f"  rows_cols: {rows_cols}")
        print(f"  highlight_row: {highlight_row}")
        print("===================================================================")
        print()

    rows, cols = rows_cols
    if rows <= 0:
        raise ValueError(f"rows={rows} must be positive")
    if cols <= 0:
        raise ValueError(f"cols={cols} must be positive")
    if cols % tile_size != 0:
        raise ValueError(f"cols must be a multiple of tile_size={tile_size}")
    if not (0 <= highlight_row < rows):
        raise ValueError(
            f"highlight_row={highlight_row} must satisfy 0 <= highlight_row < rows"
        )

    program = make_tma_copy_while_exit_program(verbose=verbose)
    if verbose:
        program.manager.print_verbose_report()
    kernel = make_tma_copy_while_exit_kernel(program.device_manager)

    a = torch.randn(rows, cols).to(torch.float16).cuda()
    b = torch.zeros_like(a)
    trace = torch.zeros(rows, 3, dtype=torch.float16, device="cuda")
    tensor_map = cl.tensor_map_tiled(a, (tile_size, 1), order="F")
    cl.launch(
        torch.cuda.current_stream(),
        (cols // tile_size,),
        (BLOCK_THREADS,),
        kernel,
        (tensor_map, b, trace, rows),
    )

    torch.testing.assert_close(b, a)
    _verify_trace_markers(trace, num_rows=rows)
    if verbose:
        print("PASS")
    return b, trace


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="TMA copy kernel with while_true_loop exit inferred from break_loop()"
    )
    parser.add_argument(
        "--rows_cols",
        type=parse_comma_separated_ints,
        default=[256, 512],
        help="Num rows and cols (comma-separated)",
    )
    parser.add_argument(
        "--highlight-row",
        type=int,
        default=128,
        help="Row index used by --inject-nested-break",
    )
    parser.add_argument(
        "--verify-num-iters",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Print retry/break assignment sequences for this modeled count",
    )
    parser.add_argument(
        "--inject-nested-break",
        action="store_true",
        help="Wrap break_loop() in when_true(is_highlight); print the exit formula",
    )
    parser.add_argument(
        "--inject-unconditional-break",
        action="store_true",
        help="Show the unconditional break + verify_num_iters=2 builder error",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Only print sequences / builder errors; do not compile the kernel",
    )
    args = parser.parse_args()
    print(args)
    if len(args.rows_cols) != 2:
        parser.error("--rows_cols must contain exactly 2 values")

    if (
        args.verify_num_iters is not None
        or args.inject_nested_break
        or args.inject_unconditional_break
    ):
        modeled = args.verify_num_iters if args.verify_num_iters is not None else 2
        print_while_assignment_sequences(
            verify_num_iters=modeled,
            inject_nested_break=args.inject_nested_break,
            inject_unconditional_break=args.inject_unconditional_break,
        )

    if args.inject_nested_break or args.inject_unconditional_break or args.skip_launch:
        raise SystemExit(0)

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required to run this example")
    run_tma_copy_while_exit_kernel_prim(
        tuple(args.rows_cols), highlight_row=args.highlight_row
    )
