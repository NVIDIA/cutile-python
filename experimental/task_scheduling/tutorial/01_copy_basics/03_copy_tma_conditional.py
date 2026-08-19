# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional, warp-specialized TMA copy.

The schedule is built and validated once on the host, then frozen into three
device tasks. Four warps consume shared-memory tiles, one warp issues TMA
loads, and three padding warps complete the register-allocation warp group.
Resource/task flow:

                   +-------------------+
                   | InputGmemResource |
                   +---------+---------+
                             |
             LoadTask         v
                  +---------+---------+
                  |   SmemResource    |
                  +---------+---------+
            StoreTask       |
                            v
                  +---------+---------+
                  | OutputGmemResource|
                  +-------------------+
                            |
                            v
                  +-------------------+
                  | TraceGmemResource |
                  |      markers      |
                  +-------------------+
"""

import argparse
from dataclasses import dataclass

import cuda.lang as cl
import torch

import task_scheduling as ts


WARP_SIZE = 32
BLOCK_THREADS = 8 * WARP_SIZE
NUM_ROWS = 256
NUM_STAGES = 1
TILE_SIZE = 128
TILE_BYTES = TILE_SIZE * 2
FP16_BYTES = cl.float16.bitwidth // 8
STORE_WARPS = 4
LOAD_WARPS = 1
PADDING_WARPS = 3
STORE_TASK_WARP_IDX = 0
LOAD_TASK_WARP_IDX = 4
PADDING_TASK_WARP_IDX = 5
HIGHLIGHT_ROW = 128
HEARTBEAT_PERIOD = 4
TRACE_COLUMNS = 4

TRACE_BEGIN_COLUMN = 0
TRACE_HEARTBEAT_COLUMN = 1
TRACE_HIGHLIGHT_COLUMN = 2
TRACE_END_COLUMN = 3

BEGIN_MARKER = 1000.0
HEARTBEAT_MARKER = 2000.0
HIGHLIGHT_MARKER = 3000.0
END_MARKER = 4000.0

# This program owns one SMEM allocation, so its view begins at the arena base.
SMEM_VIEW_OFFSET_BYTES = 0
SMEM_VIEW_ELEMENTS = NUM_STAGES * TILE_SIZE


# -----------------------------------------------------------------------------
# Host resource model and schedule tree
# -----------------------------------------------------------------------------


@dataclass(kw_only=True, eq=False)
class InputGmemResource(ts.MemoryResource):
    """Produce the TMA column coordinate consumed by the SMEM resource."""

    @ts.consumer_work(outputs=1)
    @staticmethod
    def compute_coords(stage_info):
        return cl.block_index(0) * TILE_SIZE


@dataclass(kw_only=True, eq=False)
class SmemResource(ts.MemoryResource):
    """Own the routed SMEM view and values derived from each pipeline stage."""

    highlight_row: int = HIGHLIGHT_ROW

    @staticmethod
    def _init_smem_state(stage_info):
        """Create the real typed view into the manager-owned SMEM allocation."""
        smem_base = stage_info.context.smem_base
        smem_pointer = smem_base.get_base_pointer() + SMEM_VIEW_OFFSET_BYTES
        return cl.reinterpret_pointer_as_array(
            smem_pointer,
            cl.float16,
            SMEM_VIEW_ELEMENTS,
        )

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info):
        return SmemResource._init_smem_state(stage_info)

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_read_state(stage_info):
        return SmemResource._init_smem_state(stage_info)

    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, smem_view, gmem_idx):
        values = stage_info.context.tasks_inputs
        tensor_map = values.tensor_map
        smem_stage = smem_view.get_element_pointer(
            stage_info.stage_idx * TILE_SIZE
        )
        if cl.lane_index() == 0:
            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (gmem_idx, stage_info.loop_offset),
                smem_stage,
                stage_info.barrier,
            )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def read_smem(stage_info, smem_view):
        return smem_view[
            stage_info.stage_idx * TILE_SIZE + cl.thread_index(0)
        ]

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def is_highlight_tile(stage_info):
        highlight_row = stage_info.context.tasks_inputs.highlight_row
        return stage_info.loop_offset == highlight_row


@dataclass(kw_only=True, eq=False)
class OutputGmemResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def store(stage_info, smem_val):
        output = stage_info.context.tasks_inputs.output
        column = cl.block_index(0) * TILE_SIZE + cl.thread_index(0)
        output[stage_info.loop_offset, column] = smem_val


@dataclass(kw_only=True, eq=False)
class TraceGmemResource(ts.MemoryResource):
    @staticmethod
    def _record_trace(stage_info, column, marker, warp_idx):
        context = stage_info.context
        trace = context.tasks_inputs.trace
        if (
            cl.block_index(0) == 0
            and cl.thread_index(0) // WARP_SIZE == warp_idx
            and cl.lane_index() == 0
        ):
            trace[stage_info.loop_offset, column] = marker

    @ts.producer_work
    @staticmethod
    def mark_begin(stage_info):
        return TraceGmemResource._record_trace(
            stage_info, TRACE_BEGIN_COLUMN, BEGIN_MARKER, LOAD_TASK_WARP_IDX
        )

    @ts.producer_work
    @staticmethod
    def record_heartbeat(stage_info):
        return TraceGmemResource._record_trace(
            stage_info,
            TRACE_HEARTBEAT_COLUMN,
            HEARTBEAT_MARKER,
            LOAD_TASK_WARP_IDX,
        )

    @ts.producer_work
    @staticmethod
    def mark_highlight(stage_info):
        return TraceGmemResource._record_trace(
            stage_info,
            TRACE_HIGHLIGHT_COLUMN,
            HIGHLIGHT_MARKER,
            STORE_TASK_WARP_IDX,
        )

    @ts.producer_work
    @staticmethod
    def mark_end(stage_info):
        return TraceGmemResource._record_trace(
            stage_info, TRACE_END_COLUMN, END_MARKER, STORE_TASK_WARP_IDX
        )


@ts.schedule
def load_schedule(stage_info, input_gmem, smem, trace):
    # Initialize once per task and route the same typed view through every row.
    smem_view = smem.init_load_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_rows) as domain:
        with domain.first_iter():
            trace.mark_begin()
        with domain.every(HEARTBEAT_PERIOD):
            trace.record_heartbeat()
        coordinate = input_gmem.compute_coords()
        smem.try_acquire()
        smem.acquire()
        smem.tma_load(smem_view, coordinate)
        smem.commit()


@ts.schedule
def store_schedule(stage_info, smem, output_gmem, trace):
    # The consumer task has a private context, so it initializes its own route.
    smem_view = smem.init_read_state()
    with ts.domain_loop(stage_info.context.tasks_inputs.num_rows) as domain:
        smem.try_wait()
        smem.wait()
        value = smem.read_smem(smem_view)
        smem.release()
        highlight = smem.is_highlight_tile()
        with ts.when_true(highlight):
            trace.mark_highlight()
        output_gmem.store(value)
        with domain.last_iter():
            trace.mark_end()


@ts.schedule
def padding_schedule(stage_info):
    with ts.domain_loop(stage_info.context.tasks_inputs.num_rows):
        pass


@dataclass(frozen=True)
class TasksInputs:
    tensor_map: object
    output: object
    trace: object
    num_rows: object
    highlight_row: object


@dataclass(frozen=True)
class TmaCopyProgram:
    """Host model and frozen device manager for one kernel specialization."""

    manager: object
    device_manager: object
    smem: SmemResource
    smem_allocation: object


def make_tma_copy_program():
    """Construct and validate the complete task-scheduling program."""
    pipeline_config = ts.PipelineConfig.create_tma_async_pipeline_cfg(
        num_stages=NUM_STAGES,
        num_bytes=TILE_BYTES,
        producer_group=ts.CooperativeGroup(1),
        consumer_group=ts.CooperativeGroup(STORE_WARPS * WARP_SIZE),
    )
    input_gmem = InputGmemResource(name="inputGmemResource")
    smem_allocation = ts.SmemAllocation(
        "smem_data", NUM_STAGES * TILE_BYTES, alignment=128
    )
    smem = SmemResource(
        name="smemResource",
        pipeline_config=pipeline_config,
        smem_requirements=[smem_allocation],
    )
    output_gmem = OutputGmemResource(name="outputGmemResource")
    trace_gmem = TraceGmemResource(name="traceGmemResource")

    load_task = ts.Task(
        LOAD_TASK_WARP_IDX,
        LOAD_WARPS,
        schedule=load_schedule(input_gmem, smem, trace_gmem),
        num_registers=40,
        name="LoadTask",
    )
    store_task = ts.Task(
        STORE_TASK_WARP_IDX,
        STORE_WARPS,
        schedule=store_schedule(smem, output_gmem, trace_gmem),
        num_registers=160,
        name="StoreTask",
    )
    padding_task = ts.Task(
        PADDING_TASK_WARP_IDX,
        PADDING_WARPS,
        schedule=padding_schedule(),
        num_registers=40,
        name="PaddingTask",
    )

    allocator = ts.SmemAllocator()
    allocator.add_resource(smem)
    allocator.compute_layout()
    if smem_allocation.offset != SMEM_VIEW_OFFSET_BYTES:
        raise ValueError("SMEM view offset does not match its allocator layout")
    if smem_allocation.offset % FP16_BYTES:
        raise ValueError("FP16 SMEM allocation offset must be element-aligned")
    if smem_allocation.size_bytes // FP16_BYTES != SMEM_VIEW_ELEMENTS:
        raise ValueError("SMEM view size does not match its allocator layout")

    manager = ts.TaskManager(
        tasks=[load_task, store_task, padding_task],
        smem_allocator=allocator,
        exhaustive_deadlock_race_check=True,
        exhaustive_representative_domain=True,
        verbose=True,
    )
    device_manager = manager.to_device()
    return TmaCopyProgram(manager, device_manager, smem, smem_allocation)


# -----------------------------------------------------------------------------
# Kernel launch and validation
# -----------------------------------------------------------------------------


def make_tma_copy_conditional_kernel(device_manager):
    """Specialize the kernel for one frozen task manager."""

    @cl.kernel
    def tma_copy_conditional_kernel(
        input_, output, trace, num_rows, highlight_row: cl.Constant[int]
    ):
        tensor_map = cl.tensor_map_tiled(input_, (TILE_SIZE, 1), order="F")
        if cl.lane_index() == 0:
            cl.prefetch_tensor_map(tensor_map)
        device_allocators = device_manager.setup_resources_and_tasks()
        device_manager.run(
            TasksInputs(
                tensor_map,
                output,
                trace,
                num_rows,
                highlight_row,
            ),
            device_allocators,
        )

    return tma_copy_conditional_kernel


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


def _verify_trace_markers(
    trace: torch.Tensor,
    *,
    num_rows: int,
    highlight_row: int,
) -> None:
    """Check guarded schedule markers written by ``TraceGmemResource``."""
    if not 0 <= highlight_row < num_rows:
        raise ValueError(
            f"highlight_row={highlight_row} must satisfy 0 <= highlight_row < num_rows"
        )
    if tuple(trace.shape) != (num_rows, TRACE_COLUMNS):
        raise RuntimeError(
            f"expected trace shape {(num_rows, TRACE_COLUMNS)}, "
            f"got {tuple(trace.shape)}"
        )
    _expect_trace_column(
        trace=trace,
        column=TRACE_BEGIN_COLUMN,
        marker=BEGIN_MARKER,
        expected_rows=[0],
        label="begin",
    )
    _expect_trace_column(
        trace=trace,
        column=TRACE_HEARTBEAT_COLUMN,
        marker=HEARTBEAT_MARKER,
        expected_rows=list(range(0, num_rows, HEARTBEAT_PERIOD)),
        label="heartbeat",
    )
    _expect_trace_column(
        trace=trace,
        column=TRACE_HIGHLIGHT_COLUMN,
        marker=HIGHLIGHT_MARKER,
        expected_rows=[highlight_row],
        label="highlight",
    )
    _expect_trace_column(
        trace=trace,
        column=TRACE_END_COLUMN,
        marker=END_MARKER,
        expected_rows=[num_rows - 1],
        label="end",
    )


def run_tma_copy_conditional_kernel_prim(
    rows_cols=(NUM_ROWS, 512), highlight_row=HIGHLIGHT_ROW, *, verbose=True
):
    if verbose:
        print("===================================================================")
        print("Running conditional TMA copy kernel with:")
        print(f"  rows_cols: {rows_cols}")
        print(f"  highlight_row: {highlight_row}")
        print("===================================================================")
        print()

    rows, columns = rows_cols
    if rows <= 0:
        raise ValueError("rows must be positive")
    if columns <= 0 or columns % TILE_SIZE:
        raise ValueError(f"columns must be a positive multiple of {TILE_SIZE}")
    if not 0 <= highlight_row < rows:
        raise ValueError("highlight row is outside the input")

    program = make_tma_copy_program()
    # The default tree captures 128. Preserve explicit rejection rather than
    # silently lowering a different opaque guard than host analysis saw.
    if highlight_row != program.smem.highlight_row:
        raise ValueError(
            f"this specialization requires highlight_row={program.smem.highlight_row}"
        )
    if verbose:
        program.manager.print_verbose_report()

    tma_copy_conditional_kernel = make_tma_copy_conditional_kernel(
        program.device_manager
    )
    input_ = torch.randn(rows_cols, device="cuda", dtype=torch.float16)
    output = torch.zeros_like(input_)
    trace = torch.zeros((rows, TRACE_COLUMNS), device="cuda", dtype=torch.float16)
    cl.launch(
        torch.cuda.current_stream(),
        (columns // TILE_SIZE,),
        (BLOCK_THREADS,),
        tma_copy_conditional_kernel,
        (input_, output, trace, rows, highlight_row),
    )
    torch.testing.assert_close(output, input_, rtol=0, atol=0)
    _verify_trace_markers(
        trace,
        num_rows=rows,
        highlight_row=highlight_row,
    )
    if verbose:
        print("PASS")
    return output, trace


def _parse_rows_cols(value):
    try:
        shape = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected ROWS,COLUMNS") from error
    if len(shape) != 2:
        raise argparse.ArgumentTypeError("expected exactly two dimensions")
    return shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-cols", type=_parse_rows_cols, default=(NUM_ROWS, 512)
    )
    parser.add_argument("--highlight-row", type=int, default=HIGHLIGHT_ROW)
    arguments = parser.parse_args()
    run_tma_copy_conditional_kernel_prim(
        arguments.rows_cols, arguments.highlight_row
    )
