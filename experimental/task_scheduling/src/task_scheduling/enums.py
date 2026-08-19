# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-visible task-scheduling enums and conditional guards."""

from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Hashable


_INT32_MAX = (1 << 31) - 1


class BlockGuard:
    """Base class for a captured conditional block guard."""


class IterationPredicate(BlockGuard):
    """A guard decidable from a zero-based loop iteration."""

    def fires(self, iter_idx: int, num_iters: int) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class _FirstIterPredicate(IterationPredicate):
    def fires(self, iter_idx: int, num_iters: int) -> bool:
        del num_iters
        return iter_idx == 0


@dataclass(frozen=True)
class _LastIterPredicate(IterationPredicate):
    def fires(self, iter_idx: int, num_iters: int) -> bool:
        return num_iters > 0 and iter_idx == num_iters - 1


FIRST_ITER = _FirstIterPredicate()
LAST_ITER = _LastIterPredicate()


@dataclass(frozen=True)
class Every(IterationPredicate):
    """Fire at ``start`` and every ``period`` iterations thereafter."""

    period: int
    start: int = 0

    def __post_init__(self) -> None:
        if type(self.period) is not int:
            raise TypeError(
                f"Every period must be an int, got {type(self.period).__name__}."
            )
        if not 0 < self.period <= _INT32_MAX:
            raise ValueError(
                "Every period must be positive and fit in a signed 32-bit "
                f"integer, got {self.period}."
            )
        if type(self.start) is not int:
            raise TypeError(
                f"Every start must be an int, got {type(self.start).__name__}."
            )
        if not 0 <= self.start <= _INT32_MAX:
            raise ValueError(
                "Every start must be nonnegative and fit in a signed 32-bit "
                f"integer, got {self.start}."
            )

    def fires(self, iter_idx: int, num_iters: int) -> bool:
        del num_iters
        return iter_idx >= self.start and (iter_idx - self.start) % self.period == 0


@dataclass(frozen=True)
class OpaqueCondition(BlockGuard):
    """Runtime condition correlated by a stable, host-visible key."""

    key: Hashable
    negated: bool = False
    resource: object | None = None
    method_label: str | None = None
    route_token: object | None = None


@dataclass(frozen=True)
class _SkippableGuard(BlockGuard):
    pass


SKIPPABLE = _SkippableGuard()


def is_skippable_guard(guard: object) -> bool:
    return guard is SKIPPABLE


class PipelineType(Enum):
    AsyncAsync = "AsyncAsync"
    TmaAsync = "TmaAsync"
    TmaUmma = "TmaUmma"
    UmmaAsync = "UmmaAsync"
    AsyncUmma = "AsyncUmma"
    UmmaUmma = "UmmaUmma"
    ClcFetchAsync = "ClcFetchAsync"
    DpcTmaDlc = "DpcTmaDlc"
    DlcAsync = "DlcAsync"
    AsyncDlc = "AsyncDlc"
    DlcDlc = "DlcDlc"


class TileSchedulerType(Enum):
    StaticPersistent = "StaticPersistent"
    ClcDynamicPersistent = "ClcDynamicPersistent"


class PipelineGroupMode(Enum):
    Merge = "Merge"
    Fork = "Fork"
    FusedMerge = "FusedMerge"


class SignalingThreads(IntFlag):
    All = 1
    CtaLeader = 2
    TaskWarpLeader = 4

    @classmethod
    def validate(cls, value: "SignalingThreads", field_name: str) -> None:
        valid = (
            cls.All,
            cls.CtaLeader,
            cls.TaskWarpLeader,
            cls.CtaLeader | cls.TaskWarpLeader,
        )
        if not isinstance(value, cls):
            raise TypeError(
                f"{field_name} must be a SignalingThreads value, got "
                f"{type(value).__name__}."
            )
        if value not in valid:
            raise ValueError(f"{field_name} has unsupported combination {value}.")

    def has_cta_leader(self) -> bool:
        return bool(self & SignalingThreads.CtaLeader)

    def has_task_warp_leader(self) -> bool:
        return bool(self & SignalingThreads.TaskWarpLeader)


class WorkAttr(IntFlag):
    NONE = 0
    AUXILIARY = 1

    @classmethod
    def validate(cls, value: "WorkAttr", field_name: str) -> None:
        if not isinstance(value, cls):
            raise TypeError(f"{field_name} must be a WorkAttr value.")
        if value not in (cls.NONE, cls.AUXILIARY):
            raise ValueError(f"{field_name} has unsupported combination {value}.")

    def is_auxiliary(self) -> bool:
        return bool(self & WorkAttr.AUXILIARY)


class ScheduleStage(Enum):
    ConsumerAuxWork = "ConsumerAuxWork"
    ConsumerRelease = "ConsumerRelease"
    ConsumerTryWait = "ConsumerTryWait"
    ConsumerWait = "ConsumerWait"
    ConsumerWork = "ConsumerWork"
    ProducerAuxWork = "ProducerAuxWork"
    ProducerTryAcquire = "ProducerTryAcquire"
    ProducerAcquire = "ProducerAcquire"
    ProducerCommit = "ProducerCommit"
    ProducerWork = "ProducerWork"

    def __str__(self) -> str:
        return {
            self.ConsumerAuxWork: "  ConsAux",
            self.ConsumerRelease: "<-ConsRel",
            self.ConsumerTryWait: "  ConsTrW",
            self.ConsumerWait: "->ConsWai",
            self.ConsumerWork: "  ConsWrk",
            self.ProducerAuxWork: "  ProdAux",
            self.ProducerTryAcquire: "  ProdTrA",
            self.ProducerAcquire: "  ProdAcq<-",
            self.ProducerCommit: "  ProdCmt->",
            self.ProducerWork: "  ProdWrk",
        }[self]


class ScheduleStageType(Enum):
    Head = "Head"
    Loop = "Loop"
    LoopFirstIter = "LoopFirstIter"
    LoopLastIter = "LoopLastIter"
    Tail = "Tail"


class LoopGuard(Enum):
    Always = "Always"
    LastIter = "LastIter"
    FirstIter = "FirstIter"


def is_iteration_predicate(guard: object) -> bool:
    return isinstance(guard, IterationPredicate) or guard in (
        LoopGuard.Always,
        LoopGuard.FirstIter,
        LoopGuard.LastIter,
    )


def is_loop_guard(guard: object) -> bool:
    return isinstance(guard, (LoopGuard, IterationPredicate, OpaqueCondition))


def guard_fires(
    guard: object,
    iter_idx: int,
    num_iters: int,
    *,
    opaque_assignment: dict[Hashable, bool] | None = None,
) -> bool:
    if guard == LoopGuard.Always:
        return True
    if guard == LoopGuard.FirstIter or guard is FIRST_ITER:
        return iter_idx == 0
    if guard == LoopGuard.LastIter or guard is LAST_ITER:
        return num_iters > 0 and iter_idx == num_iters - 1
    if isinstance(guard, IterationPredicate):
        return guard.fires(iter_idx, num_iters)
    if isinstance(guard, OpaqueCondition):
        active = (opaque_assignment or {}).get(guard.key, False)
        return active != guard.negated
    raise TypeError(f"unsupported loop guard type: {type(guard).__name__}")
