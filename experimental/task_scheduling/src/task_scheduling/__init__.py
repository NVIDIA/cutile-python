# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task scheduling for CUDA Lang.

Schedules and analyses are pure Python and import on hosts without a CUDA
device. Calling a frozen :class:`DeviceTask` from a ``cuda.lang`` kernel lowers
the same immutable tree. Unsupported pipeline kinds retain analysis metadata
but raise ``NotImplementedError`` at their device-lowering boundary.
"""

from . import pipeline
from .enums import (
    Every,
    IterationPredicate,
    OpaqueCondition,
    PipelineGroupMode,
    PipelineType,
    ScheduleStage,
    SignalingThreads,
    WorkAttr,
)
from .exhaustive_checker import check_all_interleavings, expand_task
from .ir import (
    ConditionalIR,
    DependencyEdgeIR,
    DomainLoopIR,
    GuardIR,
    ProgramIR,
    ResourceIR,
    ScheduleValue,
    StepIR,
    TaskIR,
    WorkTileLoopIR,
)
from .memory import (
    BarrierAllocation,
    BarrierAllocator,
    ResourceContext,
    SmemAllocation,
    SmemAllocator,
    TmemAllocation,
    TmemAllocator,
)
from .pipeline_group import PipelineGroup
from .pipeline import DevicePipelineBinding, PipelineState
from .resources import (
    CooperativeGroup,
    MemoryResource,
    PipelineConfig,
    StageInfo,
    WorkQueue,
    consumer_work,
    producer_work,
)
from .schedule_builder import (
    ConditionalBlock,
    DomainLoop,
    DynamicDomainBound,
    Schedule,
    ScheduleError,
    ScheduleStageInfo,
    Step,
    WorkTileLoop,
    domain_loop,
    dynamic_domain_bound,
    schedule,
    when_false,
    when_true,
    work_tile_loop,
)
from .task import (
    DeviceAllocators,
    DeviceBarrierAllocator,
    DeviceSmemAllocator,
    DeviceTask,
    DeviceTaskManager,
    ExecutionContext,
    Task,
)
from .task_manager import TaskManager


__all__ = [
    "BarrierAllocation",
    "BarrierAllocator",
    "check_all_interleavings",
    "ConditionalBlock",
    "ConditionalIR",
    "consumer_work",
    "CooperativeGroup",
    "DependencyEdgeIR",
    "domain_loop",
    "DomainLoop",
    "DomainLoopIR",
    "dynamic_domain_bound",
    "DynamicDomainBound",
    "DeviceAllocators",
    "DeviceBarrierAllocator",
    "DeviceSmemAllocator",
    "DeviceTask",
    "DeviceTaskManager",
    "DevicePipelineBinding",
    "Every",
    "ExecutionContext",
    "expand_task",
    "IterationPredicate",
    "GuardIR",
    "MemoryResource",
    "OpaqueCondition",
    "pipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineGroup",
    "PipelineGroupMode",
    "PipelineType",
    "ProgramIR",
    "producer_work",
    "ResourceContext",
    "ResourceIR",
    "schedule",
    "Schedule",
    "ScheduleError",
    "ScheduleStageInfo",
    "ScheduleValue",
    "ScheduleStage",
    "SignalingThreads",
    "SmemAllocation",
    "SmemAllocator",
    "StageInfo",
    "Step",
    "StepIR",
    "Task",
    "TaskIR",
    "TaskManager",
    "TmemAllocation",
    "TmemAllocator",
    "when_false",
    "when_true",
    "WorkAttr",
    "WorkQueue",
    "work_tile_loop",
    "WorkTileLoop",
    "WorkTileLoopIR",
]
