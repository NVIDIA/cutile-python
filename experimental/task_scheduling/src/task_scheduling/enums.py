# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-visible task-scheduling enums and conditional guards."""

from dataclasses import dataclass, replace
from enum import Enum, IntFlag
from typing import Hashable, Iterator


_INT32_MAX = (1 << 31) - 1


class BlockGuard:
    """Base class for a captured conditional block guard.

    ``&`` and ``|`` build flattened boolean guard trees. ``~`` converts a guard
    to negation-normal form, including first-class complements of iteration
    predicates.
    """

    def __and__(self, other: object) -> "BlockGuard":
        if not isinstance(other, BlockGuard):
            return NotImplemented
        return _combine_guard(GuardAnd, self, other)

    def __rand__(self, other: object) -> "BlockGuard":
        if not isinstance(other, BlockGuard):
            return NotImplemented
        return _combine_guard(GuardAnd, other, self)

    def __or__(self, other: object) -> "BlockGuard":
        if not isinstance(other, BlockGuard):
            return NotImplemented
        return _combine_guard(GuardOr, self, other)

    def __ror__(self, other: object) -> "BlockGuard":
        if not isinstance(other, BlockGuard):
            return NotImplemented
        return _combine_guard(GuardOr, other, self)

    def __invert__(self) -> "BlockGuard":
        return _negate_guard(self)

    def __bool__(self) -> bool:
        from .schedule_builder import ScheduleError

        raise ScheduleError(
            "schedule conditions cannot be converted to bool; use '&', '|', "
            "and '~'. Python 'and', 'or', and 'not' cannot be overloaded."
        )


class IterationPredicate(BlockGuard):
    """A guard decidable from a zero-based loop iteration."""

    def fires(self, iter_idx: int, num_iters: int) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ComplementedIterationPredicate(IterationPredicate):
    """Boolean complement of an iteration predicate."""

    predicate: IterationPredicate

    def fires(self, iter_idx: int, num_iters: int) -> bool:
        return not self.predicate.fires(iter_idx, num_iters)


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
class GuardAnd(BlockGuard):
    """Conjunction of two or more guards, flattened at construction."""

    operands: tuple[BlockGuard, ...]

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError("GuardAnd requires at least two operands.")


@dataclass(frozen=True)
class GuardOr(BlockGuard):
    """Disjunction of two or more guards, flattened at construction."""

    operands: tuple[BlockGuard, ...]

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError("GuardOr requires at least two operands.")


def _combine_guard(
    operator: type[GuardAnd] | type[GuardOr],
    left: BlockGuard,
    right: BlockGuard,
) -> BlockGuard:
    """Build one flattened AND/OR node."""
    operands: list[BlockGuard] = []
    for operand in (left, right):
        if isinstance(operand, operator):
            operands.extend(operand.operands)
        else:
            operands.append(operand)
    return operator(tuple(operands))


def _negate_guard(guard: BlockGuard) -> BlockGuard:
    """Return ``guard`` in negation-normal form."""
    if isinstance(guard, OpaqueCondition):
        return replace(guard, negated=not guard.negated)
    if isinstance(guard, (GuardAnd, GuardOr)):
        dual = GuardOr if isinstance(guard, GuardAnd) else GuardAnd
        result = _negate_guard(guard.operands[0])
        for operand in guard.operands[1:]:
            result = _combine_guard(dual, result, _negate_guard(operand))
        return result
    if isinstance(guard, ComplementedIterationPredicate):
        return guard.predicate
    if isinstance(guard, IterationPredicate):
        return ComplementedIterationPredicate(guard)
    if type(guard).__invert__ is not BlockGuard.__invert__:
        # Capture-time operands, such as a schedule value, negate themselves.
        return ~guard
    from .schedule_builder import ScheduleError

    raise ScheduleError(f"{type(guard).__name__} guards cannot be negated")


def iter_leaves(guard: object) -> Iterator[object]:
    """Yield the non-composite leaves of a guard tree."""
    if isinstance(guard, (GuardAnd, GuardOr)):
        for operand in guard.operands:
            yield from iter_leaves(operand)
    else:
        yield guard


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
    return isinstance(
        guard, (LoopGuard, IterationPredicate, OpaqueCondition, GuardAnd, GuardOr)
    )


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
    if isinstance(guard, (GuardAnd, GuardOr)):
        combine = all if isinstance(guard, GuardAnd) else any
        return combine(
            guard_fires(
                operand, iter_idx, num_iters, opaque_assignment=opaque_assignment
            )
            for operand in guard.operands
        )
    raise TypeError(f"unsupported loop guard type: {type(guard).__name__}")
