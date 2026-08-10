# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any, Callable
from enum import Enum, auto

import cuda.lang as cl
import pytest
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).parents[3]))
    __package__ = "test.examples.program_builder"

from test.util import require_hopper_or_newer

from .program_builder import (
    ProgN,
    ProgramFragment,
    VisitorIterate,
)


NUM_STAGES = 1
TILE_SIZE = 128
WARP_SIZE = 32
DEFAULT_REGISTER_COUNT = 128
STORE_WARP_START = 0
STORE_WARP_COUNT = 4
LOAD_WARP_START = 4
PADDING_WARP_START = 5


class ScheduleStage(Enum):
    ProducerAuxWork = auto()
    ProducerTryAcquire = auto()
    ProducerAcquire = auto()
    ProducerWork = auto()
    ProducerCommit = auto()

    ConsumerAuxWork = auto()
    ConsumerTryWait = auto()
    ConsumerWait = auto()
    ConsumerWork = auto()
    ConsumerRelease = auto()


@dataclass(frozen=True)
class ResourceReference:
    label: str


@dataclass(frozen=True)
class ValueReference:
    label: str
    dtype: Any


@dataclass(frozen=True)
class ScheduleStep(ProgramFragment):
    resource: ResourceReference
    stage: ScheduleStage
    label: str
    action: Callable
    inputs: tuple[ValueReference, ...] = ()
    outputs: tuple[ValueReference, ...] = ()
    reads: tuple[ResourceReference, ...] = ()
    writes: tuple[ResourceReference, ...] = ()

    def __call__(self, context):
        return self.action(context)


@dataclass(frozen=True)
class DomainLoop(ProgramFragment):
    """Run a schedule body once for each row to be copied"""

    body: ProgramFragment

    def __call__(self, context):
        for loop_offset in range(context.num_rows):
            context = replace(context, loop_offset=loop_offset)
            context = self.body(context)
        return context


@dataclass(frozen=True)
class Task(ProgramFragment):
    """A subschedule assigned to contiguous warps."""

    name: str
    warp_idx: int
    num_warps: int
    num_registers: int
    body: ProgramFragment

    def __call__(self, context):
        if cl.ensure_constant(self.num_registers > DEFAULT_REGISTER_COUNT):
            cl.setmaxregister_increase(self.num_registers)
        else:
            cl.setmaxregister_decrease(self.num_registers)
        return self.body(context)


@dataclass(frozen=True)
class TaskProgram(ProgramFragment):
    """Collection of tasks and resources"""

    resources: tuple[ResourceReference, ...]
    tasks: tuple[Task, ...]

    def __call__(self, context):
        warp_idx = cl.shfl_sync(cl.warp_index(), 0)
        for task in cl.static_iter(self.tasks):
            selected = (
                warp_idx >= task.warp_idx and warp_idx < task.warp_idx + task.num_warps
            )
            if selected:
                context = task(context)
        return context


@dataclass(frozen=True)
class CopyContext:
    """
    Runtime context needed for tma copy kernel

    Drawback of this approach is that fields cannot change types, so you can't
    have a nullable context field for example.
    """

    tensor_map: Any
    output: Any
    shared_memory: Any
    full_mbarrier: Any
    empty_mbarrier: Any
    num_rows: int
    loop_offset: int
    gmem_idx: int
    producer_ready: bool
    consumer_ready: bool
    smem_value: Any

    def stage_index(self):
        return self.loop_offset % NUM_STAGES

    def full_phase(self):
        return (self.loop_offset // NUM_STAGES) & 1

    def empty_phase(self):
        return self.full_phase() ^ 1

    def full_barrier(self):
        return self.full_mbarrier + self.stage_index()

    def empty_barrier(self):
        return self.empty_mbarrier + self.stage_index()


INPUT = ResourceReference("input")
SMEM = ResourceReference("smem")
OUTPUT = ResourceReference("output")

GMEM_IDX = ValueReference("gmem_idx", cl.int32)
PRODUCER_READY = ValueReference("producer_ready", cl.bool_)
CONSUMER_READY = ValueReference("consumer_ready", cl.bool_)
SMEM_VALUE = ValueReference("smem_value", cl.float16)


def compute_coordinates(context):
    gmem_idx = cl.block_index(0) * TILE_SIZE
    return replace(context, gmem_idx=gmem_idx)


def try_acquire(context):
    ready = context.producer_ready
    if cl.elect_sync():
        ready = cl.mbarrier_try_wait_parity(
            context.empty_barrier(),
            context.empty_phase(),
        )
    return replace(context, producer_ready=ready)


def acquire(context):
    if cl.elect_sync():
        if not context.producer_ready:
            cl.mbarrier_wait_parity(
                context.empty_barrier(),
                context.empty_phase(),
            )
    return context


def tma_load(context):
    if cl.elect_sync():
        full_mbarrier = context.full_barrier()
        cl.mbarrier_arrive_expect_transaction(
            full_mbarrier,
            TILE_SIZE * cast(int, cl.float16.bitwidth) // 8,
        )
        cl.copy_async_bulk_tensor_global_to_shared(
            context.tensor_map,
            (context.gmem_idx, context.loop_offset),
            context.shared_memory.get_element_pointer((context.stage_index(), 0)),
            full_mbarrier,
        )
    return context


def try_wait(context):
    ready = cl.mbarrier_try_wait_parity(
        context.full_barrier(),
        context.full_phase(),
    )
    return replace(context, consumer_ready=ready)


def wait(context):
    if not context.consumer_ready:
        cl.mbarrier_wait_parity(
            context.full_barrier(),
            context.full_phase(),
        )
    return context


def read_shared_memory(context):
    smem_value = context.shared_memory[context.stage_index(), cl.thread_index(0)]
    return replace(context, smem_value=smem_value)


def release(context):
    cl.mbarrier_arrive(context.empty_barrier())
    return context


def store_output(context):
    output_column = cl.block_index(0) * TILE_SIZE + cl.thread_index(0)
    context.output[context.loop_offset, output_column] = context.smem_value
    return context


LOAD_SCHEDULE = ProgN(
    (
        DomainLoop(
            ProgN(
                (
                    ScheduleStep(
                        resource=INPUT,
                        stage=ScheduleStage.ConsumerWork,
                        label="compute coordinates",
                        action=compute_coordinates,
                        outputs=(GMEM_IDX,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ProducerTryAcquire,
                        label="try acquire shared-memory stage",
                        action=try_acquire,
                        outputs=(PRODUCER_READY,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ProducerAcquire,
                        label="acquire shared-memory stage",
                        action=acquire,
                        inputs=(PRODUCER_READY,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ProducerWork,
                        label="load input tile with TMA",
                        action=tma_load,
                        reads=(INPUT,),
                        writes=(SMEM,),
                        inputs=(GMEM_IDX,),
                    ),
                )
            )
        ),
    )
)

STORE_SCHEDULE = ProgN(
    (
        DomainLoop(
            ProgN(
                (
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ConsumerTryWait,
                        label="try wait for shared-memory stage",
                        action=try_wait,
                        outputs=(CONSUMER_READY,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ConsumerWait,
                        label="wait for shared-memory stage",
                        action=wait,
                        inputs=(CONSUMER_READY,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ConsumerWork,
                        label="read shared-memory value",
                        action=read_shared_memory,
                        reads=(SMEM,),
                        outputs=(SMEM_VALUE,),
                    ),
                    ScheduleStep(
                        resource=SMEM,
                        stage=ScheduleStage.ConsumerRelease,
                        label="release shared-memory stage",
                        action=release,
                    ),
                    ScheduleStep(
                        resource=OUTPUT,
                        stage=ScheduleStage.ProducerWork,
                        label="store output value",
                        action=store_output,
                        reads=(SMEM,),
                        writes=(OUTPUT,),
                        inputs=(SMEM_VALUE,),
                    ),
                )
            )
        ),
    )
)


PADDING_SCHEDULE = DomainLoop(ProgN(()))


PROGRAM = TaskProgram(
    resources=(INPUT, SMEM, OUTPUT),
    tasks=(
        Task(
            name="StoreTask",
            warp_idx=STORE_WARP_START,
            num_warps=STORE_WARP_COUNT,
            num_registers=160,
            body=STORE_SCHEDULE,
        ),
        Task(
            name="LoadTask",
            warp_idx=LOAD_WARP_START,
            num_warps=1,
            num_registers=40,
            body=LOAD_SCHEDULE,
        ),
        Task(
            name="PaddingTask",
            warp_idx=PADDING_WARP_START,
            num_warps=3,
            num_registers=40,
            body=PADDING_SCHEDULE,
        ),
    ),
)


def collect_steps(task):
    steps = []

    def visitor(node):
        if isinstance(node, ScheduleStep):
            steps.append(node)
        return VisitorIterate.CONTINUE

    task.body.visit(visitor)
    return steps


def validate_program(program):
    declared_resources = set(program.resources)
    occupied_warps = set()
    graph_edges = set()

    for task in program.tasks:
        assert 8 <= task.num_registers <= 256
        assert task.num_registers % 8 == 0

        task_warps = set(range(task.warp_idx, task.warp_idx + task.num_warps))
        assert occupied_warps.isdisjoint(task_warps)
        occupied_warps.update(task_warps)

        for step in collect_steps(task):
            assert step.resource in declared_resources
            assert set(step.reads) <= declared_resources
            assert set(step.writes) <= declared_resources
            graph_edges.update(
                (read, written) for read in step.reads for written in step.writes
            )

    assert occupied_warps == set(range(8))
    assert graph_edges == {(INPUT, SMEM), (SMEM, OUTPUT)}


@cl.kernel
def tma_copy_kernel(input_tensor, output_tensor):
    tensor_map = cl.tensor_map_tiled(
        input_tensor,
        (TILE_SIZE, 1),
        order="F",
    )
    shared_memory = cl.shared_array(
        (NUM_STAGES, TILE_SIZE),
        cl.float16,
        alignment=128,
    )
    full_mbarriers = cl.shared_array(NUM_STAGES, cl.mbarrier, alignment=8)
    empty_mbarriers = cl.shared_array(NUM_STAGES, cl.mbarrier, alignment=8)

    if cl.thread_index(0) == 0:
        cl.prefetch_tensor_map(tensor_map)
        for stage_idx in cl.static_iter(range(NUM_STAGES)):
            cl.mbarrier_initialize(
                full_mbarriers.get_element_pointer(stage_idx),
                1,
            )
            cl.mbarrier_initialize(
                empty_mbarriers.get_element_pointer(stage_idx),
                STORE_WARP_COUNT * WARP_SIZE,
            )
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=cl.FenceRestriction.mbarrier_initialize(),
        )

    cl.barrier_sync_block()

    context = CopyContext(
        tensor_map=tensor_map,
        output=output_tensor,
        shared_memory=shared_memory,
        full_mbarrier=full_mbarriers.get_base_pointer(),
        empty_mbarrier=empty_mbarriers.get_base_pointer(),
        num_rows=input_tensor.shape[0],
        loop_offset=cl.int32(0),
        gmem_idx=cl.int32(0),
        producer_ready=cl.bool_(False),
        consumer_ready=cl.bool_(False),
        smem_value=cl.float16(0.0),
    )
    PROGRAM(context)


def test_tma_copy_program_structure():
    validate_program(PROGRAM)


CONFIGS = ((3, 128), (5, 256))


@require_hopper_or_newer()
@pytest.mark.parametrize("rows,columns", CONFIGS)
def test_tma_copy_program_builder(rows, columns):
    torch.manual_seed(0)
    input_tensor = torch.randn(
        (rows, columns),
        dtype=torch.float16,
        device="cuda",
    )
    output_tensor = torch.zeros_like(input_tensor)

    cl.launch(
        torch.cuda.current_stream(),
        (columns // TILE_SIZE,),
        (256,),
        tma_copy_kernel,
        (input_tensor, output_tensor),
    )

    torch.testing.assert_close(output_tensor, input_tensor, rtol=0, atol=0)


if __name__ == "__main__":
    print("validating...")
    validate_program(PROGRAM)
    print(str(PROGRAM))
    print("running...")
    test_tma_copy_program_builder(*CONFIGS[0])
    print("success")
