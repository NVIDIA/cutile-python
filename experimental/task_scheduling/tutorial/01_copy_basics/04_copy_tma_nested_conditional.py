# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TMA copy with nested runtime conditions and an exact branch trace.

For each row, the consumer selects one of four leaves using row % 4 < 2
and row % 2 == 0. The inner predicate is produced inside the outer branch.
Columns 0..3 record TT, TF, FT, FF respectively; exactly one marker is
written per row. A fifth column exercises a third nested condition.
The copied value remains live across the conditional tree and is stored
unconditionally. Pipeline acquire/wait/commit/release follow the basic copy.
With --stop-row, both tasks break before processing that row; the remaining
output and trace stay zero. The break itself is inside two runtime conditions
and returns the loop-carried copied-row count with break_loop(copied_rows).
Both tasks write their final count after the loop, for validation on the host.
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
TRACE_COLUMNS = 5
NESTED_COLUMN = 4

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

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def has_copy_limit(stage_info):
        return stage_info.context.tasks_inputs.stop_row >= 0

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def limit_reached(stage_info):
        return stage_info.loop_offset >= stage_info.context.tasks_inputs.stop_row

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_copied_rows(stage_info):
        return cl.int32(0)

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def advance_copied_rows(stage_info, copied_rows):
        return copied_rows + 1


@dataclass(kw_only=True, eq=False)
class SmemResource(ts.MemoryResource):
    """Own the routed SMEM view and values derived from each pipeline stage."""

    @staticmethod
    def _init_smem_state(stage_info):
        """Create the real typed view into the manager-owned SMEM allocation."""
        smem_base = stage_info.context.smem_base
        smem_pointer = smem_base.pointer() + SMEM_VIEW_OFFSET_BYTES
        typed_pointer = cl.bitcast(
            smem_pointer,
            cl.pointer_dtype(cl.float16, smem_pointer.memory_space),
        )
        return cl.Array.from_parts(typed_pointer, SMEM_VIEW_ELEMENTS)

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
        smem_stage = smem_view.pointer(
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

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def is_first_pair(stage_info):
        return stage_info.loop_offset % 4 < 2

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def is_even_row(stage_info):
        return stage_info.loop_offset % 2 == 0

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def is_first_octet_half(stage_info):
        return stage_info.loop_offset % 8 < 4


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
    @ts.producer_work
    @staticmethod
    def record_copied_rows(stage_info, copied_rows, slot, warp_idx):
        if cl.block_index(0) == 0 and cl.thread_index(0) == warp_idx * WARP_SIZE:
            stage_info.context.tasks_inputs.copied_rows[slot] = copied_rows

    @ts.producer_work
    @staticmethod
    def record_branch(stage_info, column):
        # One writer per row; all consumer threads take the same branch.
        if cl.block_index(0) == 0 and cl.thread_index(0) == 0:
            trace = stage_info.context.tasks_inputs.trace
            trace[stage_info.loop_offset, column] = column + 1


@ts.schedule
def load_schedule(stage_info, input_gmem, smem, trace):
    # Initialize once per task and route the same typed view through every row.
    smem_view = smem.init_load_state()
    copied_rows = input_gmem.init_copied_rows()

    def loop_body(copied_rows):
        limited = input_gmem.has_copy_limit()
        with ts.when_true(limited, key="copy_limited"):
            done = input_gmem.limit_reached()
            with ts.when_true(done, key="copy_limit_reached"):
                ts.break_loop(copied_rows)
        coordinate = input_gmem.compute_coords()
        smem.try_acquire()
        smem.acquire()
        smem.tma_load(smem_view, coordinate)
        smem.commit()
        return input_gmem.advance_copied_rows(copied_rows)

    copied_rows = ts.domain_loop(
        0,
        stage_info.context.tasks_inputs.num_rows,
        1,
        loop_body,
        copied_rows,
    )
    trace.record_copied_rows(copied_rows, 0, LOAD_TASK_WARP_IDX)


@ts.schedule
def store_schedule(stage_info, input_gmem, smem, output_gmem, trace):
    # The consumer task has a private context, so it initializes its own route.
    smem_view = smem.init_read_state()
    copied_rows = input_gmem.init_copied_rows()

    def loop_body(copied_rows):
        limited = input_gmem.has_copy_limit()
        with ts.when_true(limited, key="copy_limited"):
            done = input_gmem.limit_reached()
            with ts.when_true(done, key="copy_limit_reached"):
                ts.break_loop(copied_rows)
        smem.try_wait()
        smem.wait()
        value = smem.read_smem(smem_view)
        smem.release()
        first_pair = smem.is_first_pair()
        with ts.when_true(first_pair):
            even = smem.is_even_row()
            with ts.when_true(even):
                trace.record_branch(0)
                first_half = smem.is_first_octet_half()
                with ts.when_false(first_half):
                    trace.record_branch(NESTED_COLUMN)
            with ts.when_false(even):
                trace.record_branch(1)
        with ts.when_false(first_pair):
            even = smem.is_even_row()
            with ts.when_true(even):
                trace.record_branch(2)
            with ts.when_false(even):
                trace.record_branch(3)
        output_gmem.store(value)
        return input_gmem.advance_copied_rows(copied_rows)

    copied_rows = ts.domain_loop(
        0,
        stage_info.context.tasks_inputs.num_rows,
        1,
        loop_body,
        copied_rows,
    )
    trace.record_copied_rows(copied_rows, 1, STORE_TASK_WARP_IDX)


@ts.schedule
def padding_schedule(stage_info):
    def loop_body():
        pass

    ts.domain_loop(
        0,
        stage_info.context.tasks_inputs.num_rows,
        1,
        loop_body,
    )


@dataclass(frozen=True)
class TasksInputs:
    tensor_map: object
    output: object
    trace: object
    num_rows: object
    stop_row: object
    copied_rows: object


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
        schedule=store_schedule(input_gmem, smem, output_gmem, trace_gmem),
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
        resource_dependency_graph={},
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


def make_tma_copy_nested_conditional_kernel(device_manager):
    """Specialize the kernel for one frozen task manager."""

    @cl.kernel
    def tma_copy_nested_conditional_kernel(
        input_, output, trace, num_rows, stop_row, copied_rows
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
                stop_row,
                copied_rows,
            ),
            device_allocators,
        )

    return tma_copy_nested_conditional_kernel


def _verify_trace_markers(
    trace: torch.Tensor, *, num_rows: int, stop_row: int | None = None
) -> None:
    if tuple(trace.shape) != (num_rows, TRACE_COLUMNS):
        raise RuntimeError(f"unexpected trace shape: {tuple(trace.shape)}")
    expected = torch.zeros_like(trace)
    copied_rows = num_rows if stop_row is None else stop_row
    for row in range(copied_rows):
        column = row % 4
        expected[row, column] = column + 1
        if row % 8 == 4:
            expected[row, NESTED_COLUMN] = NESTED_COLUMN + 1
    torch.testing.assert_close(trace, expected, rtol=0, atol=0)


def run_tma_copy_nested_conditional_kernel_prim(
    rows_cols=(NUM_ROWS, 512), *, stop_row=None, verbose=True
):
    rows, columns = rows_cols
    if rows <= 0:
        raise ValueError("rows must be positive")
    if columns <= 0 or columns % TILE_SIZE:
        raise ValueError(f"columns must be a positive multiple of {TILE_SIZE}")
    if stop_row is not None and (type(stop_row) is not int or not 0 <= stop_row <= rows):
        raise ValueError("stop_row must be an integer between zero and rows")

    program = make_tma_copy_program()
    if verbose:
        program.manager.print_verbose_report()
    kernel = make_tma_copy_nested_conditional_kernel(program.device_manager)
    input_ = torch.randn(rows_cols, device="cuda:0", dtype=torch.float16)
    output = torch.zeros_like(input_)
    trace = torch.zeros((rows, TRACE_COLUMNS), device="cuda:0", dtype=torch.float16)
    # A sentinel catches missing post-loop writes even when stop_row is zero.
    copied_rows = torch.full((2,), -1, device="cuda:0", dtype=torch.int32)
    cl.launch(
        torch.cuda.current_stream(),
        (columns // TILE_SIZE,),
        (BLOCK_THREADS,),
        kernel,
        (input_, output, trace, rows, -1 if stop_row is None else stop_row, copied_rows),
    )
    expected = input_.clone()
    if stop_row is not None:
        expected[stop_row:] = 0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    _verify_trace_markers(trace, num_rows=rows, stop_row=stop_row)
    expected_count = rows if stop_row is None else stop_row
    torch.testing.assert_close(
        copied_rows, torch.full_like(copied_rows, expected_count), rtol=0, atol=0
    )
    if verbose:
        print("PASS: exact copy, nested branch trace, and loop-carried copied-row counts")
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
    parser.add_argument("--stop-row", type=int, default=None)
    arguments = parser.parse_args()
    run_tma_copy_nested_conditional_kernel_prim(
        arguments.rows_cols, stop_row=arguments.stop_row
    )
