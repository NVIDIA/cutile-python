# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the CUDA Lang Blackwell FMHA-prefill tutorial.

This module mirrors DKG's single fmha_prefill_helpers.py dependency. It
contains the pure-Python mask/work-tile formulas, specialization traits,
CUDA Lang leaf device helpers, and host-side tensor/reference utilities used
by fmha_prefill.py and its focused tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import cuda.lang as cl

try:
    import torch
except ImportError:  # Keep planning/configuration usable when PyTorch is absent.
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Forward-mask and work-context helpers.
# ---------------------------------------------------------------------------

QUERY_SUPER_TILE = 256
QUERY_HALF_TILE = 128
KEY_TILE = 128


class ForwardMask(Enum):
    """Forward mask kinds supported by the authoritative prefill kernel."""

    RESIDUAL = auto()
    WINDOW = auto()
    WINDOW_INFERENCE = auto()


@dataclass(frozen=True)
class ForwardMaskSpec:
    """Resolved mask kind and inclusive sliding-window bounds."""

    kind: ForwardMask
    window_left: int | None
    window_right: int | None


@dataclass(frozen=True)
class TripInterval:
    """K-tile interval ``[start, end)`` visited for one Q tile."""

    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class HalfTripCounts:
    """Source-ordered mask partition for one 128-row half of a super-tile."""

    interval: TripInterval
    local_interval: TripInterval
    leading_masked: int
    unmasked: int
    trailing_masked: int
    trailing_invalid: int

    @property
    def count(self) -> int:
        return self.interval.count

    def ranges(self) -> tuple[TripInterval, TripInterval, TripInterval, TripInterval]:
        """Return leading, unmasked, trailing, and token-only sub-intervals."""

        cursor = self.interval.start
        leading = TripInterval(cursor, cursor + self.leading_masked)
        cursor = leading.end
        unmasked = TripInterval(cursor, cursor + self.unmasked)
        cursor = unmasked.end
        trailing = TripInterval(cursor, cursor + self.trailing_masked)
        cursor = trailing.end
        invalid = TripInterval(cursor, cursor + self.trailing_invalid)
        return leading, unmasked, trailing, invalid


@dataclass(frozen=True)
class WorkContext:
    """Host oracle for one scheduled ``(Q super-tile, Q head, batch)`` work item.

    ``q_base`` and ``k_base`` are zero for fixed-length tensor maps, whose batch
    coordinate is ``batch``.  They are cumulative flattened offsets for varlen
    maps, whose tensor-map batch coordinate is always zero.
    """

    seq_tile: int
    q_head: int
    k_head: int
    batch: int
    is_varlen: bool
    tensor_map_batch: int
    seq_q: int
    seq_k: int
    q_base: int
    k_base: int
    q_local_start: int
    q_local_limit: int
    q_start: int
    q_limit: int
    valid_super_tile: bool
    key_tile_count: int
    trip_interval: TripInterval | None
    half_counts: tuple[HalfTripCounts, HalfTripCounts] | None
    mask: ForwardMaskSpec

    @property
    def valid_query_rows(self) -> int:
        return self.q_local_limit - self.q_local_start

    def query_row_is_valid(self, row_in_super_tile: int) -> bool:
        if not 0 <= row_in_super_tile < QUERY_SUPER_TILE:
            return False
        return is_query_row_valid(
            self.q_local_start + row_in_super_tile,
            self.seq_q,
        )

    def score_element_is_valid(self, row_in_super_tile: int, key: int) -> bool:
        if not 0 <= row_in_super_tile < QUERY_SUPER_TILE:
            return False
        return is_score_element_valid(
            self.mask.kind,
            self.q_local_start + row_in_super_tile,
            key,
            self.seq_q,
            self.seq_k,
            self.mask.window_left,
            self.mask.window_right,
        )


def _require_nonnegative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")


def _validate_window(name: str, value: int | None) -> None:
    if value is not None:
        _require_nonnegative(name, value)


def cumulative_offsets(lengths: Iterable[int]) -> tuple[int, ...]:
    """Return source-compatible int32-style cumulative sequence offsets."""

    result = [0]
    for index, length in enumerate(lengths):
        _require_nonnegative(f"lengths[{index}]", length)
        result.append(result[-1] + length)
    return tuple(result)


def validate_cumulative_offsets(
    offsets: Sequence[int], *, name: str = "offsets"
) -> tuple[int, ...]:
    """Validate and normalize a cumulative-offset array."""

    normalized = tuple(offsets)
    if len(normalized) < 2:
        raise ValueError(f"{name} must contain at least [0, end]")
    if normalized[0] != 0:
        raise ValueError(f"{name}[0] must be zero, got {normalized[0]!r}")
    previous = 0
    for index, value in enumerate(normalized):
        _require_nonnegative(f"{name}[{index}]", value)
        if value < previous:
            raise ValueError(f"{name} must be nondecreasing")
        previous = value
    return normalized


def query_super_tile_count(seq_q: int, tile_m: int = QUERY_SUPER_TILE) -> int:
    _require_nonnegative("seq_q", seq_q)
    if tile_m <= 0:
        raise ValueError(f"tile_m must be positive, got {tile_m!r}")
    return cl.cdiv(seq_q, tile_m)


def key_tile_count(seq_k: int, tile_n: int = KEY_TILE) -> int:
    _require_nonnegative("seq_k", seq_k)
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got {tile_n!r}")
    return cl.cdiv(seq_k, tile_n)


def grid_shape(
    seq_q: int | Sequence[int], batch_count: int, q_head_count: int
) -> tuple[int, int, int]:
    """Return the one-dimensional source launch grid.

    Varlen launches schedule the maximum Q length for every batch; work items
    outside a batch's actual Q length are rejected by ``make_work_context``.
    """

    _require_nonnegative("batch_count", batch_count)
    _require_nonnegative("q_head_count", q_head_count)
    if isinstance(seq_q, int):
        max_seq_q = seq_q
    else:
        lengths = tuple(seq_q)
        if len(lengths) != batch_count:
            raise ValueError(
                f"seq_q has {len(lengths)} lengths for batch_count={batch_count}"
            )
        for index, length in enumerate(lengths):
            _require_nonnegative(f"seq_q[{index}]", length)
        max_seq_q = max(lengths, default=0)
    tiles = query_super_tile_count(max_seq_q)
    return tiles * batch_count * q_head_count, 1, 1


def resolve_forward_mask(
    seq_k: int | Sequence[int],
    *,
    is_causal: bool,
    bottom_right: bool,
    window_left: int | None = None,
    window_right: int | None = None,
    tile_n: int = KEY_TILE,
) -> ForwardMaskSpec:
    """Reproduce the source host's forward mask selection.

    Causal mode overrides the supplied right bound with zero.  A noncausal,
    unwindowed K tail selects ``RESIDUAL``.  Otherwise bottom-right alignment
    selects ``WINDOW_INFERENCE`` and ordinary alignment selects ``WINDOW``.
    """

    _validate_window("window_left", window_left)
    _validate_window("window_right", window_right)
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got {tile_n!r}")

    if isinstance(seq_k, int):
        lengths = (seq_k,)
    else:
        lengths = tuple(seq_k)
    for index, length in enumerate(lengths):
        _require_nonnegative(f"seq_k[{index}]", length)

    kind = ForwardMask.WINDOW_INFERENCE if bottom_right else ForwardMask.WINDOW
    if is_causal:
        window_right = 0
    elif window_left is None and window_right is None:
        if any(length % tile_n != 0 for length in lengths):
            kind = ForwardMask.RESIDUAL
    return ForwardMaskSpec(kind, window_left, window_right)


def bottom_right_offset(kind: ForwardMask, seq_q: int, seq_k: int) -> int:
    """Return ``Sk-Sq`` only for the bottom-right forward mask."""

    _require_nonnegative("seq_q", seq_q)
    _require_nonnegative("seq_k", seq_k)
    if kind is ForwardMask.WINDOW_INFERENCE:
        return seq_k - seq_q
    return 0


def get_trip_start(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> int:
    """Port of forward ``FusedMask.get_trip_start``."""

    _validate_trip_arguments(
        block_row, tile_m, tile_n, seq_q, seq_k, window_left, window_right
    )
    if kind in (ForwardMask.WINDOW, ForwardMask.WINDOW_INFERENCE):
        if window_left is not None:
            offset = bottom_right_offset(kind, seq_q, seq_k)
            first_key = block_row * tile_m + offset - window_left
            return max(first_key // tile_n, 0)
    return 0


def get_trip_end(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> int:
    """Return the exclusive end corresponding to source trip count + start."""

    _validate_trip_arguments(
        block_row, tile_m, tile_n, seq_q, seq_k, window_left, window_right
    )
    max_key_tiles = cl.cdiv(seq_k, tile_n)
    if kind is ForwardMask.RESIDUAL or window_right is None:
        return max_key_tiles
    offset = bottom_right_offset(kind, seq_q, seq_k)
    exclusive_key = (block_row + 1) * tile_m + offset + window_right
    return min(max_key_tiles, cl.cdiv(exclusive_key, tile_n))


def get_trip_interval(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> TripInterval:
    return TripInterval(
        get_trip_start(
            kind,
            block_row,
            tile_m,
            tile_n,
            seq_q,
            seq_k,
            window_left,
            window_right,
        ),
        get_trip_end(
            kind,
            block_row,
            tile_m,
            tile_n,
            seq_q,
            seq_k,
            window_left,
            window_right,
        ),
    )


def get_trip_count(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> int:
    return get_trip_interval(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    ).count


def _leading_mask_id(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None,
    window_right: int | None,
) -> tuple[int, int]:
    interval = get_trip_interval(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    leading_end = interval.start - 1
    if kind in (ForwardMask.WINDOW, ForwardMask.WINDOW_INFERENCE):
        if window_left is not None:
            offset = bottom_right_offset(kind, seq_q, seq_k)
            first_key = (block_row + 1) * tile_m + offset - window_left
            leading_end = min(cl.cdiv(first_key, tile_n) - 1, interval.end - 1)
    return interval.start, leading_end


def get_masked_leading_count(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> int:
    if kind is ForwardMask.RESIDUAL:
        return 0
    if window_left is None and window_right is None:
        return 0
    begin, end = _leading_mask_id(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    return max(end - begin + 1, 0)


def _trailing_mask_id(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None,
    window_right: int | None,
) -> tuple[int, int]:
    interval = get_trip_interval(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    trailing_end = interval.end - 1
    if window_right is not None:
        offset = bottom_right_offset(kind, seq_q, seq_k)
        last_key_for_first_row = block_row * tile_m + offset + window_right
        trailing_begin = min(last_key_for_first_row // tile_n, trailing_end)
    else:
        trailing_begin = trailing_end
    return trailing_begin, trailing_end


def get_masked_trailing_count(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
    remaining_count: int = 0,
) -> int:
    """Port of the source trailing count, including residual K tails."""

    _require_nonnegative("remaining_count", remaining_count)
    if kind is ForwardMask.RESIDUAL:
        return int(seq_k % tile_n != 0) + remaining_count
    if window_left is None and window_right is None:
        return remaining_count

    trailing_begin, trailing_end = _trailing_mask_id(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    _, leading_end = _leading_mask_id(
        kind,
        block_row,
        tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    if trailing_begin <= leading_end:
        count = max(trailing_end - leading_end, 0)
    else:
        count = max(trailing_end - trailing_begin + 1, 0)
    return count + remaining_count


def get_unmasked_trip_count(
    kind: ForwardMask,
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> int:
    return (
        get_trip_count(
            kind,
            block_row,
            tile_m,
            tile_n,
            seq_q,
            seq_k,
            window_left,
            window_right,
        )
        - get_masked_leading_count(
            kind,
            block_row,
            tile_m,
            tile_n,
            seq_q,
            seq_k,
            window_left,
            window_right,
        )
        - get_masked_trailing_count(
            kind,
            block_row,
            tile_m,
            tile_n,
            seq_q,
            seq_k,
            window_left,
            window_right,
        )
    )


def get_half_trip_counts(
    kind: ForwardMask,
    seq_tile: int,
    half: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
    *,
    super_tile_m: int = QUERY_SUPER_TILE,
    half_tile_m: int = QUERY_HALF_TILE,
    tile_n: int = KEY_TILE,
) -> HalfTripCounts:
    """Map one half's local source counts onto the common super-tile interval."""

    _require_nonnegative("seq_tile", seq_tile)
    if half not in (0, 1):
        raise ValueError(f"half must be 0 or 1, got {half!r}")
    if super_tile_m != 2 * half_tile_m:
        raise ValueError("super_tile_m must contain exactly two half tiles")

    interval = get_trip_interval(
        kind,
        seq_tile,
        super_tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    local_block_row = seq_tile * 2 + half
    local_interval = get_trip_interval(
        kind,
        local_block_row,
        half_tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    local_leading = get_masked_leading_count(
        kind,
        local_block_row,
        half_tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )
    local_trailing = get_masked_trailing_count(
        kind,
        local_block_row,
        half_tile_m,
        tile_n,
        seq_q,
        seq_k,
        window_left,
        window_right,
    )

    leading_gap = local_interval.start - interval.start
    trailing_invalid = interval.count - local_interval.count - leading_gap
    leading_masked = local_leading + leading_gap
    unmasked = (
        interval.count - leading_masked - local_trailing - trailing_invalid
    )
    counts = HalfTripCounts(
        interval,
        local_interval,
        leading_masked,
        unmasked,
        local_trailing,
        trailing_invalid,
    )
    if min(
        counts.leading_masked,
        counts.unmasked,
        counts.trailing_masked,
        counts.trailing_invalid,
    ) < 0:
        raise ValueError(f"source mask partition produced a negative count: {counts}")
    if sum(
        (
            counts.leading_masked,
            counts.unmasked,
            counts.trailing_masked,
            counts.trailing_invalid,
        )
    ) != interval.count:
        raise AssertionError(f"mask partition does not cover {interval}: {counts}")
    return counts


def is_query_row_valid(query: int, seq_q: int) -> bool:
    _require_nonnegative("seq_q", seq_q)
    return 0 <= query < seq_q


def is_key_row_valid(key: int, seq_k: int) -> bool:
    _require_nonnegative("seq_k", seq_k)
    return 0 <= key < seq_k


def is_score_element_valid(
    kind: ForwardMask,
    query: int,
    key: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None = None,
    window_right: int | None = None,
) -> bool:
    """Return the exact forward per-element keep predicate."""

    _validate_window("window_left", window_left)
    _validate_window("window_right", window_right)
    if not is_query_row_valid(query, seq_q) or not is_key_row_valid(key, seq_k):
        return False
    if kind is ForwardMask.RESIDUAL:
        return True

    offset = bottom_right_offset(kind, seq_q, seq_k)
    if window_left is not None and key < query + offset - window_left:
        return False
    if window_right is not None and key > query + offset + window_right:
        return False
    return True


def make_work_context(
    *,
    seq_tile: int,
    q_head: int,
    batch: int,
    head_ratio: int,
    mask: ForwardMaskSpec,
    seq_q: int | None = None,
    seq_k: int | None = None,
    cumulative_q: Sequence[int] | None = None,
    cumulative_k: Sequence[int] | None = None,
) -> WorkContext:
    """Construct fixed- or variable-length work context with source arithmetic."""

    _require_nonnegative("seq_tile", seq_tile)
    _require_nonnegative("q_head", q_head)
    _require_nonnegative("batch", batch)
    if not isinstance(head_ratio, int) or isinstance(head_ratio, bool) or head_ratio < 1:
        raise ValueError(f"head_ratio must be a positive integer, got {head_ratio!r}")

    has_cumulative_q = cumulative_q is not None
    has_cumulative_k = cumulative_k is not None
    if has_cumulative_q != has_cumulative_k:
        raise ValueError("cumulative_q and cumulative_k must both be provided or None")

    if has_cumulative_q:
        if seq_q is not None or seq_k is not None:
            raise ValueError("fixed lengths and cumulative offsets are mutually exclusive")
        q_offsets = validate_cumulative_offsets(cumulative_q, name="cumulative_q")
        k_offsets = validate_cumulative_offsets(cumulative_k, name="cumulative_k")
        if len(q_offsets) != len(k_offsets):
            raise ValueError("cumulative_q and cumulative_k must have equal batch size")
        if batch >= len(q_offsets) - 1:
            raise ValueError(f"batch {batch} is outside cumulative offsets")
        q_base = q_offsets[batch]
        k_base = k_offsets[batch]
        actual_seq_q = q_offsets[batch + 1] - q_base
        actual_seq_k = k_offsets[batch + 1] - k_base
        tensor_map_batch = 0
    else:
        if seq_q is None or seq_k is None:
            raise ValueError("seq_q and seq_k are required for fixed-length context")
        _require_nonnegative("seq_q", seq_q)
        _require_nonnegative("seq_k", seq_k)
        actual_seq_q = seq_q
        actual_seq_k = seq_k
        q_base = 0
        k_base = 0
        tensor_map_batch = batch

    q_local_start = seq_tile * QUERY_SUPER_TILE
    q_local_limit = min(q_local_start + QUERY_SUPER_TILE, actual_seq_q)
    valid_super_tile = q_local_start < actual_seq_q
    if not valid_super_tile:
        q_local_limit = q_local_start

    interval = None
    halves = None
    if valid_super_tile:
        interval = get_trip_interval(
            mask.kind,
            seq_tile,
            QUERY_SUPER_TILE,
            KEY_TILE,
            actual_seq_q,
            actual_seq_k,
            mask.window_left,
            mask.window_right,
        )
        halves = (
            get_half_trip_counts(
                mask.kind,
                seq_tile,
                0,
                actual_seq_q,
                actual_seq_k,
                mask.window_left,
                mask.window_right,
            ),
            get_half_trip_counts(
                mask.kind,
                seq_tile,
                1,
                actual_seq_q,
                actual_seq_k,
                mask.window_left,
                mask.window_right,
            ),
        )

    return WorkContext(
        seq_tile=seq_tile,
        q_head=q_head,
        k_head=q_head // head_ratio,
        batch=batch,
        is_varlen=has_cumulative_q,
        tensor_map_batch=tensor_map_batch,
        seq_q=actual_seq_q,
        seq_k=actual_seq_k,
        q_base=q_base,
        k_base=k_base,
        q_local_start=q_local_start,
        q_local_limit=q_local_limit,
        q_start=q_base + q_local_start,
        q_limit=q_base + q_local_limit,
        valid_super_tile=valid_super_tile,
        key_tile_count=key_tile_count(actual_seq_k),
        trip_interval=interval,
        half_counts=halves,
        mask=mask,
    )


def _validate_trip_arguments(
    block_row: int,
    tile_m: int,
    tile_n: int,
    seq_q: int,
    seq_k: int,
    window_left: int | None,
    window_right: int | None,
) -> None:
    _require_nonnegative("block_row", block_row)
    _require_nonnegative("seq_q", seq_q)
    _require_nonnegative("seq_k", seq_k)
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"tile sizes must be positive, got {(tile_m, tile_n)!r}")
    _validate_window("window_left", window_left)
    _validate_window("window_right", window_right)


# ---------------------------------------------------------------------------
# Compile-time specialization traits.
# ---------------------------------------------------------------------------

MMA_M = 128
MMA_N = 128
P_READY_STAGES = 2
SUPPORTED_HEAD_DIMS = (32, 64, 128)


class ElementType(str, Enum):
    """FMHA tensor element families."""

    F16 = "F16"
    BF16 = "BF16"
    E4M3 = "E4M3"


class SmemSwizzle(str, Enum):
    """TMA/shared-memory swizzles, using the source table's short names."""

    S32 = "s32"
    S64 = "s64"
    S128 = "s128"

    @property
    def bytes(self) -> int:
        return {self.S32: 32, self.S64: 64, self.S128: 128}[self]

    @property
    def cuda_lang_name(self) -> str:
        return {
            self.S32: "SWIZZLE_32B",
            self.S64: "SWIZZLE_64B",
            self.S128: "SWIZZLE_128B",
        }[self]


class L2Promotion(str, Enum):
    """Input tensor-map L2 promotion requested by the source."""

    NONE = "none"
    L2_64B = "l2_64b"
    L2_128B = "l2_128b"


class Tcgen05Kind(str, Enum):
    """Names of the corresponding ``cl.Tcgen05MMAKind`` members."""

    F16 = "F16"
    F8F6F4 = "F8F6F4"


class InstructionFormat(str, Enum):
    """A/B format fields used in a tcgen05 instruction descriptor."""

    F16 = "F16"
    BF16 = "BF16"


class UnsupportedTraitCombination(ValueError):
    """Raised when a mechanically defined row lacks source-parity support."""


FP8_D32_SOURCE_CAVEAT = (
    "E4M3 input with D32 is not source-parity supported: the authoritative "
    "source divides its single QK phase across two floor-divided half loops, "
    "so neither loop issues an MMA. A source GPU experiment is required before "
    "enabling this specialization."
)


@dataclass(frozen=True)
class InstructionTraits:
    """tcgen05 kind and descriptor fields shared by QK and PV."""

    kind: Tcgen05Kind
    qk_a_format: InstructionFormat
    qk_b_format: InstructionFormat
    pv_a_format: InstructionFormat
    pv_b_format: InstructionFormat
    accumulator_format: str = "F32"
    pv_b_major: bool = True
    source_fp8_uses_f16_descriptor_fields: bool = False


@dataclass(frozen=True)
class InputTraits:
    """Compile-time traits controlled by the common Q/K/V input type."""

    element_type: ElementType
    head_dim: int
    bit_width: int
    cuda_lang_dtype_name: str
    inner_bytes: int
    swizzle: SmemSwizzle
    l2_promotion: L2Promotion
    tma_slices: int
    tma_slice_elements: int
    tma_transaction_bytes: int
    k_step: int
    qk_phases: int
    pv_phases_total: int
    pv_phases_per_ready_chunk: int
    p_values_per_32bit_word: int
    p_conversion_helper: str
    p_conversion_calls_per_word: int
    instruction: InstructionTraits
    qk_descriptor_increment: int
    pv_descriptor_increment: int
    packed_p_tmem_increment: int
    extra_bytes_qk: int
    pv_leading_byte_offset: int
    source_parity_supported: bool
    source_parity_caveat: str | None

    def require_source_parity(self) -> None:
        if not self.source_parity_supported:
            raise UnsupportedTraitCombination(self.source_parity_caveat)


@dataclass(frozen=True)
class OutputTraits:
    """Compile-time traits controlled independently by the output type."""

    element_type: ElementType
    head_dim: int
    bit_width: int
    cuda_lang_dtype_name: str
    inner_bytes: int
    swizzle: SmemSwizzle
    l2_promotion: L2Promotion
    tma_slices: int
    tma_slice_elements: int
    tma_transaction_bytes: int
    values_per_32bit_word: int
    conversion_helper: str
    conversion_calls_per_word: int


@dataclass(frozen=True)
class KernelTraits:
    """Resolved input and output traits for one kernel specialization."""

    head_dim: int
    input: InputTraits
    output: OutputTraits

    def require_source_parity(self) -> None:
        self.input.require_source_parity()


_DTYPE_ALIASES = {
    "f16": ElementType.F16,
    "float16": ElementType.F16,
    "bf16": ElementType.BF16,
    "bfloat16": ElementType.BF16,
    "e4m3": ElementType.E4M3,
    "float8e4m3fn": ElementType.E4M3,
}

_BIT_WIDTH = {
    ElementType.F16: 16,
    ElementType.BF16: 16,
    ElementType.E4M3: 8,
}

_CUDA_LANG_DTYPE = {
    ElementType.F16: "float16",
    ElementType.BF16: "bfloat16",
    ElementType.E4M3: "float8_e4m3fn",
}

_PACK_HELPER = {
    ElementType.F16: ("ff2f16x2_rn", 1),
    ElementType.BF16: ("ff2bf16x2_rn", 1),
    ElementType.E4M3: ("ff_to_e4m3x2_rn", 2),
}

_INPUT_ROWS = {
    # dtype, D: inner bytes, swizzle, slices, K step, QK, PV total, PV/chunk
    (ElementType.F16, 32): (64, SmemSwizzle.S64, 1, 16, 2, 8, 4),
    (ElementType.F16, 64): (128, SmemSwizzle.S128, 1, 16, 4, 8, 4),
    (ElementType.F16, 128): (256, SmemSwizzle.S128, 2, 16, 8, 8, 4),
    (ElementType.BF16, 32): (64, SmemSwizzle.S64, 1, 16, 2, 8, 4),
    (ElementType.BF16, 64): (128, SmemSwizzle.S128, 1, 16, 4, 8, 4),
    (ElementType.BF16, 128): (256, SmemSwizzle.S128, 2, 16, 8, 8, 4),
    (ElementType.E4M3, 32): (32, SmemSwizzle.S32, 1, 32, 1, 4, 2),
    (ElementType.E4M3, 64): (64, SmemSwizzle.S64, 1, 32, 2, 4, 2),
    (ElementType.E4M3, 128): (128, SmemSwizzle.S128, 1, 32, 4, 4, 2),
}

_OUTPUT_ROWS = {
    # dtype, D: inner bytes, swizzle, slices
    (ElementType.F16, 32): (64, SmemSwizzle.S64, 1),
    (ElementType.F16, 64): (128, SmemSwizzle.S128, 1),
    (ElementType.F16, 128): (256, SmemSwizzle.S128, 2),
    (ElementType.BF16, 32): (64, SmemSwizzle.S64, 1),
    (ElementType.BF16, 64): (128, SmemSwizzle.S128, 1),
    (ElementType.BF16, 128): (256, SmemSwizzle.S128, 2),
    (ElementType.E4M3, 32): (32, SmemSwizzle.S32, 1),
    (ElementType.E4M3, 64): (64, SmemSwizzle.S64, 1),
    (ElementType.E4M3, 128): (128, SmemSwizzle.S128, 1),
}


def _normalize_element_type(value: ElementType | str) -> ElementType:
    if isinstance(value, ElementType):
        return value
    if not isinstance(value, str):
        raise TypeError(f"element type must be ElementType or str, got {type(value)!r}")
    key = value.strip().lower().replace("_", "")
    try:
        return _DTYPE_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(member.value for member in ElementType)
        raise ValueError(
            f"unsupported element type {value!r}; expected {supported}"
        ) from error


def _validate_head_dim(head_dim: int) -> int:
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise TypeError(f"head_dim must be an int, got {type(head_dim)!r}")
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"unsupported head_dim {head_dim}; expected one of {SUPPORTED_HEAD_DIMS}"
        )
    return head_dim


def _l2_promotion(swizzle: SmemSwizzle) -> L2Promotion:
    return {
        SmemSwizzle.S32: L2Promotion.NONE,
        SmemSwizzle.S64: L2Promotion.L2_64B,
        SmemSwizzle.S128: L2Promotion.L2_128B,
    }[swizzle]


def _instruction_traits(element_type: ElementType) -> InstructionTraits:
    if element_type is ElementType.E4M3:
        # Deliberately reproduce the source encoding: .kind::f8f6f4 with F16
        # descriptor fields. Do not substitute E4M3 fields by intuition.
        return InstructionTraits(
            kind=Tcgen05Kind.F8F6F4,
            qk_a_format=InstructionFormat.F16,
            qk_b_format=InstructionFormat.F16,
            pv_a_format=InstructionFormat.F16,
            pv_b_format=InstructionFormat.F16,
            source_fp8_uses_f16_descriptor_fields=True,
        )
    descriptor_format = (
        InstructionFormat.BF16
        if element_type is ElementType.BF16
        else InstructionFormat.F16
    )
    return InstructionTraits(
        kind=Tcgen05Kind.F16,
        qk_a_format=descriptor_format,
        qk_b_format=descriptor_format,
        pv_a_format=descriptor_format,
        pv_b_format=descriptor_format,
    )


def resolve_input_traits(
    element_type: ElementType | str,
    head_dim: int,
    *,
    allow_unsupported_fp8_d32: bool = False,
) -> InputTraits:
    """Resolve Q/K/V traits, rejecting E4M3/D32 source parity by default."""

    dtype = _normalize_element_type(element_type)
    dim = _validate_head_dim(head_dim)
    inner_bytes, swizzle, slices, k_step, qk, pv, pv_chunk = _INPUT_ROWS[dtype, dim]
    bits = _BIT_WIDTH[dtype]
    p_helper, helper_calls = _PACK_HELPER[dtype]
    transaction_bytes = MMA_N * dim * bits // 8
    qk_inc_bytes = k_step * bits // 8
    source_supported = not (dtype is ElementType.E4M3 and dim == 32)
    caveat = None if source_supported else FP8_D32_SOURCE_CAVEAT
    traits = InputTraits(
        element_type=dtype,
        head_dim=dim,
        bit_width=bits,
        cuda_lang_dtype_name=_CUDA_LANG_DTYPE[dtype],
        inner_bytes=inner_bytes,
        swizzle=swizzle,
        l2_promotion=_l2_promotion(swizzle),
        tma_slices=slices,
        tma_slice_elements=dim // slices,
        tma_transaction_bytes=transaction_bytes,
        k_step=k_step,
        qk_phases=qk,
        pv_phases_total=pv,
        pv_phases_per_ready_chunk=pv_chunk,
        p_values_per_32bit_word=32 // bits,
        p_conversion_helper=p_helper,
        p_conversion_calls_per_word=helper_calls,
        instruction=_instruction_traits(dtype),
        qk_descriptor_increment=qk_inc_bytes >> 4,
        pv_descriptor_increment=(k_step * (dim // slices) * bits // 8) >> 4,
        packed_p_tmem_increment=k_step * bits // 32,
        extra_bytes_qk=(
            qk_inc_bytes * (qk // 2) if slices == 1 else transaction_bytes // slices
        ),
        pv_leading_byte_offset=0 if slices == 1 else transaction_bytes // slices,
        source_parity_supported=source_supported,
        source_parity_caveat=caveat,
    )
    _validate_input_traits(traits)
    if not allow_unsupported_fp8_d32:
        traits.require_source_parity()
    return traits


def resolve_output_traits(
    element_type: ElementType | str, head_dim: int
) -> OutputTraits:
    """Resolve output packing and full-half TMA traits."""

    dtype = _normalize_element_type(element_type)
    dim = _validate_head_dim(head_dim)
    inner_bytes, swizzle, slices = _OUTPUT_ROWS[dtype, dim]
    bits = _BIT_WIDTH[dtype]
    helper, helper_calls = _PACK_HELPER[dtype]
    traits = OutputTraits(
        element_type=dtype,
        head_dim=dim,
        bit_width=bits,
        cuda_lang_dtype_name=_CUDA_LANG_DTYPE[dtype],
        inner_bytes=inner_bytes,
        swizzle=swizzle,
        l2_promotion=L2Promotion.NONE,
        tma_slices=slices,
        tma_slice_elements=dim // slices,
        tma_transaction_bytes=MMA_M * dim * bits // 8,
        values_per_32bit_word=32 // bits,
        conversion_helper=helper,
        conversion_calls_per_word=helper_calls,
    )
    _validate_output_traits(traits)
    return traits


def resolve_kernel_traits(
    input_type: ElementType | str,
    output_type: ElementType | str,
    head_dim: int,
    *,
    allow_unsupported_fp8_d32: bool = False,
) -> KernelTraits:
    """Resolve independent input/output tables for one FMHA specialization."""

    dim = _validate_head_dim(head_dim)
    return KernelTraits(
        head_dim=dim,
        input=resolve_input_traits(
            input_type,
            dim,
            allow_unsupported_fp8_d32=allow_unsupported_fp8_d32,
        ),
        output=resolve_output_traits(output_type, dim),
    )


def _validate_input_traits(traits: InputTraits) -> None:
    """Guard the table against changes that violate source formulas."""

    if traits.inner_bytes != traits.head_dim * traits.bit_width // 8:
        raise RuntimeError("invalid input inner-byte trait")
    if traits.tma_slices * traits.swizzle.bytes != traits.inner_bytes:
        raise RuntimeError("input slices do not cover the swizzled inner dimension")
    if traits.qk_phases * traits.k_step != traits.head_dim:
        raise RuntimeError("QK phases do not cover D")
    if traits.pv_phases_total * traits.k_step != MMA_N:
        raise RuntimeError("PV phases do not cover the 128-key tile")
    if traits.pv_phases_per_ready_chunk * P_READY_STAGES != traits.pv_phases_total:
        raise RuntimeError("PV ready chunks do not cover every PV phase")
    if traits.p_values_per_32bit_word * traits.bit_width != 32:
        raise RuntimeError("invalid packed-P ratio")
    if traits.packed_p_tmem_increment != 8:
        raise RuntimeError("packed-P TMEM increment must remain eight words")
    if traits.tma_transaction_bytes != MMA_N * traits.inner_bytes:
        raise RuntimeError("input TMA transaction must cover the complete tile")


def _validate_output_traits(traits: OutputTraits) -> None:
    if traits.inner_bytes != traits.head_dim * traits.bit_width // 8:
        raise RuntimeError("invalid output inner-byte trait")
    if traits.tma_slices * traits.swizzle.bytes != traits.inner_bytes:
        raise RuntimeError("output slices do not cover the swizzled inner dimension")
    if traits.l2_promotion is not L2Promotion.NONE:
        raise RuntimeError("source output tensor maps do not request L2 promotion")
    if traits.values_per_32bit_word * traits.bit_width != 32:
        raise RuntimeError("invalid output packing ratio")
    if traits.tma_transaction_bytes != MMA_M * traits.inner_bytes:
        raise RuntimeError("output TMA transaction must cover one complete half")


# ---------------------------------------------------------------------------
# Leaf CUDA Lang device helpers.
# ---------------------------------------------------------------------------

# CUTLASS Swizzle<B, 4, 3> uses B={1,2,3} for 32/64/128-byte TMA
# swizzles.  The XOR source bits start at byte-address bit 7 and the
# destination bits start at bit 4.
SWIZZLE_32B_BITS = 1
SWIZZLE_64B_BITS = 2
SWIZZLE_128B_BITS = 3


@cl.function
def pack_f16x2(lo, hi):
    """Round two FP32 values to F16 and return ``lo | hi << 16``."""

    return cl.bitcast(cl._nvvm.ff2f16x2_rn(hi, lo), cl.uint32)


@cl.function
def pack_bf16x2(lo, hi):
    """Round two FP32 values to BF16 and return ``lo | hi << 16``."""

    return cl.bitcast(cl._nvvm.ff2bf16x2_rn(hi, lo), cl.uint32)


@cl.function
def pack_e4m3x4(value0, value1, value2, value3):
    """Round four FP32 values to E4M3 and pack them in lane order."""

    lo = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(value1, value0)))
    hi = cl.uint32(cl.uint16(cl._nvvm.ff_to_e4m3x2_rn(value3, value2)))
    return lo | (hi << 16)


# Distinct P/output names make call-site intent visible while sharing the exact
# same numeric conversion.  In particular, FP8 uses two x2 conversions per
# 32-bit word in both paths.
@cl.function
def pack_p_f16x2(lo, hi):
    return pack_f16x2(lo, hi)


@cl.function
def pack_p_bf16x2(lo, hi):
    return pack_bf16x2(lo, hi)


@cl.function
def pack_p_e4m3x4(value0, value1, value2, value3):
    return pack_e4m3x4(value0, value1, value2, value3)


@cl.function
def pack_output_f16x2(lo, hi):
    return pack_f16x2(lo, hi)


@cl.function
def pack_output_bf16x2(lo, hi):
    return pack_bf16x2(lo, hi)


@cl.function
def pack_output_e4m3x4(value0, value1, value2, value3):
    return pack_e4m3x4(value0, value1, value2, value3)


@cl.function
def unpack_f16_lane(packed, lane):
    """Extract one F16 lane from a packed word and widen it to FP32."""

    bits = cl.uint16(cl.uint32(packed) >> (cl.uint32(lane) * 16))
    return cl.float32(cl.bitcast(bits, cl.float16))


@cl.function
def unpack_bf16_lane(packed, lane):
    """Extract one BF16 lane from a packed word and widen it to FP32."""

    bits = cl.uint16(cl.uint32(packed) >> (cl.uint32(lane) * 16))
    return cl.float32(cl.bitcast(bits, cl.bfloat16))


@cl.function
def unpack_e4m3_lane(packed, lane):
    """Extract one E4M3 lane from a packed word and widen it to FP32."""

    # CUDA Lang's SIMT MLIR lowering does not expose scalar FP8 arithmetic
    # types.  Decode the containing x2 pair directly to F16, then widen.
    pair_index = cl.uint32(lane) // 2
    pair_bits = cl.int16(cl.uint32(packed) >> (pair_index * 16))
    pair = cl._nvvm.e4m3x2_to_f16x2_rn(pair_bits)
    return cl.float32(pair[cl.int32(cl.uint32(lane) % 2)])


@cl.function
def swizzle_byte_offset(byte_offset, swizzle_bits):
    """Apply CUTLASS ``Swizzle<B,4,3>`` to a slice-local byte offset.

    The operation is its own inverse because it XORs bits 4..6 using bits
    7..9, which are not modified.  ``swizzle_bits`` must be 1, 2, or 3.
    """

    xor_mask = (cl.uint32(1) << cl.uint32(swizzle_bits)) - cl.uint32(1)
    xor_bits = (cl.uint32(byte_offset) >> 7) & xor_mask
    return cl.uint32(byte_offset) ^ (xor_bits << 4)


@cl.function
def unswizzle_byte_offset(byte_offset, swizzle_bits):
    """Invert :func:`swizzle_byte_offset`."""

    return swizzle_byte_offset(byte_offset, swizzle_bits)


@cl.function
def split_d_swizzle_byte_offset(
    row,
    byte_in_head,
    row_count,
    slice_bytes,
    swizzle_bits,
):
    """Map a logical row/head byte to its (possibly split-D) SMEM byte.

    Each D slice is a separate ``row_count x slice_bytes`` TMA transaction.
    This is important for 16-bit D128, whose two 128-byte swizzled slices are
    separated by 16 KiB for a 128-row tile.
    """

    slice_index = cl.uint32(byte_in_head) // cl.uint32(slice_bytes)
    byte_in_slice = cl.uint32(byte_in_head) % cl.uint32(slice_bytes)
    slice_span = cl.uint32(row_count) * cl.uint32(slice_bytes)
    logical_in_slice = cl.uint32(row) * cl.uint32(slice_bytes) + byte_in_slice
    return slice_index * slice_span + swizzle_byte_offset(
        logical_in_slice, swizzle_bits
    )


@cl.function
def split_d_unswizzle_byte_offset(
    physical_byte_offset,
    row_count,
    slice_bytes,
    head_bytes,
    swizzle_bits,
):
    """Invert a split-D physical byte offset to a linear logical byte offset."""

    slice_span = cl.uint32(row_count) * cl.uint32(slice_bytes)
    slice_index = cl.uint32(physical_byte_offset) // slice_span
    physical_in_slice = cl.uint32(physical_byte_offset) % slice_span
    logical_in_slice = unswizzle_byte_offset(physical_in_slice, swizzle_bits)
    row = logical_in_slice // cl.uint32(slice_bytes)
    byte_in_slice = logical_in_slice % cl.uint32(slice_bytes)
    return (
        row * cl.uint32(head_bytes)
        + slice_index * cl.uint32(slice_bytes)
        + byte_in_slice
    )


@cl.function
def qk_descriptor_phase_offset(
    phase,
    qk_phase_count,
    descriptor_increment,
    second_half_offset,
):
    """Return the source Q/K SMEM descriptor offset for one K phase.

    ``second_half_offset`` is ``extra_bytes_qk >> 4``.  The source emits two
    equal-sized loops; callers must reject odd phase counts (notably FP8 D32).
    """

    half = cl.uint32(qk_phase_count) // 2
    phase_u32 = cl.uint32(phase)
    if phase_u32 < half:
        return phase_u32 * cl.uint32(descriptor_increment)
    return (phase_u32 - half) * cl.uint32(
        descriptor_increment
    ) + cl.uint32(second_half_offset)


@cl.function
def pv_descriptor_phase_offset(phase, descriptor_increment):
    """Return the V SMEM descriptor offset for one PV K phase."""

    return cl.uint32(phase) * cl.uint32(descriptor_increment)


@cl.function
def packed_p_tmem_phase_offset(phase, packed_p_increment):
    """Return the packed-P TMEM word offset for one PV K phase."""

    return cl.uint32(phase) * cl.uint32(packed_p_increment)


@cl.function
def qk_instruction_descriptor_f16():
    """Encode the source 128x128 FP16 QK instruction descriptor."""

    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        n=128,
        m=128,
    ).encode()


@cl.function
def qk_instruction_descriptor_bf16():
    """Encode the source 128x128 BF16 QK instruction descriptor."""

    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
        n=128,
        m=128,
    ).encode()


@cl.function
def qk_instruction_descriptor_e4m3():
    """Encode the source FP8 QK descriptor (F16 fields, F8F6F4 kind)."""

    return qk_instruction_descriptor_f16()


@cl.function
def pv_instruction_descriptor_f16(head_dim):
    """Encode the source 128xD FP16 PV descriptor with B major."""

    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.F16,
        transpose_b=True,
        n=head_dim,
        m=128,
    ).encode()


@cl.function
def pv_instruction_descriptor_bf16(head_dim):
    """Encode the source 128xD BF16 PV descriptor with B major."""

    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.Tcgen05InstructionDescriptor.DType.F32,
        a_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
        b_type=cl.Tcgen05InstructionDescriptor.F16Type.BF16,
        transpose_b=True,
        n=head_dim,
        m=128,
    ).encode()


@cl.function
def pv_instruction_descriptor_e4m3(head_dim):
    """Encode the source FP8 PV descriptor (F16 fields, F8F6F4 kind)."""

    return pv_instruction_descriptor_f16(head_dim)


__all__ = (
    "ElementType",
    "FP8_D32_SOURCE_CAVEAT",
    "InputTraits",
    "InstructionFormat",
    "InstructionTraits",
    "KernelTraits",
    "L2Promotion",
    "OutputTraits",
    "SWIZZLE_32B_BITS",
    "SWIZZLE_64B_BITS",
    "SWIZZLE_128B_BITS",
    "SmemSwizzle",
    "Tcgen05Kind",
    "UnsupportedTraitCombination",
    "pack_f16x2",
    "pack_bf16x2",
    "pack_e4m3x4",
    "pack_p_f16x2",
    "pack_p_bf16x2",
    "pack_p_e4m3x4",
    "pack_output_f16x2",
    "pack_output_bf16x2",
    "pack_output_e4m3x4",
    "unpack_f16_lane",
    "unpack_bf16_lane",
    "unpack_e4m3_lane",
    "swizzle_byte_offset",
    "unswizzle_byte_offset",
    "split_d_swizzle_byte_offset",
    "split_d_unswizzle_byte_offset",
    "qk_descriptor_phase_offset",
    "pv_descriptor_phase_offset",
    "packed_p_tmem_phase_offset",
    "qk_instruction_descriptor_f16",
    "qk_instruction_descriptor_bf16",
    "qk_instruction_descriptor_e4m3",
    "pv_instruction_descriptor_f16",
    "pv_instruction_descriptor_bf16",
    "pv_instruction_descriptor_e4m3",
    "resolve_input_traits",
    "resolve_kernel_traits",
    "resolve_output_traits",
)


# ---------------------------------------------------------------------------
# Host-side data, reference, validation, and benchmark helpers.
# ---------------------------------------------------------------------------

ShapeSpec: TypeAlias = tuple[int, int | tuple[int, ...], int, int]

SUPPORTED_DTYPE_NAMES = ("Float16", "BFloat16", "Float8E4M3FN")
DEFAULT_Q_SHAPE: ShapeSpec = (4, 1024, 8, 64)
DEFAULT_K_SHAPE: ShapeSpec = (4, 1024, 8, 64)
DEFAULT_MMA_TILER = (128, 128)
DEFAULT_TOLERANCE = 1.0e-1
FP8_MINIMUM_TOLERANCE = 1.3e-1


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for tensor preparation and reference checks"
        )
    return torch


def torch_dtype(dtype_name: str) -> Any:
    """Return the PyTorch dtype corresponding to a public FMHA dtype name."""

    torch_module = _require_torch()
    mapping = {
        "Float16": torch_module.float16,
        "BFloat16": torch_module.bfloat16,
    }
    if hasattr(torch_module, "float8_e4m3fn"):
        mapping["Float8E4M3FN"] = torch_module.float8_e4m3fn
    if dtype_name not in mapping:
        if dtype_name == "Float8E4M3FN":
            raise RuntimeError("this PyTorch build has no float8_e4m3fn dtype")
        raise ValueError(f"unsupported dtype name {dtype_name!r}")
    return mapping[dtype_name]


def _shape_lengths(shape: ShapeSpec, *, name: str) -> tuple[int, ...]:
    batch, sequence, _, _ = shape
    if isinstance(sequence, tuple):
        if len(sequence) != batch:
            raise ValueError(f"{name} varlen tuple must contain one length per batch")
        return sequence
    return (sequence,) * batch


@dataclass(frozen=True)
class FmhaConfig:
    """Source-compatible host configuration for an FMHA prefill problem."""

    q_shape: ShapeSpec = DEFAULT_Q_SHAPE
    k_shape: ShapeSpec = DEFAULT_K_SHAPE
    in_dtype: str = "Float16"
    out_dtype: str = "Float16"
    qk_acc_dtype: str = "Float32"
    pv_acc_dtype: str = "Float32"
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER
    is_causal: bool = False
    bottom_right_align: bool = False
    lse_calculation: bool = False
    window_size: tuple[int, int] = (-1, -1)
    scale_q: float = 1.0
    scale_k: float = 1.0
    scale_v: float = 1.0
    inv_scale_o: float = 1.0
    scale_softmax: float = 0.0
    use_sinks: bool = False
    enable_skip_correction: bool = True
    enable_approx_epilogue_rcp: bool = True

    @property
    def variable_length(self) -> bool:
        return isinstance(self.q_shape[1], tuple)

    @property
    def batch(self) -> int:
        return self.q_shape[0]

    @property
    def heads_q(self) -> int:
        return self.q_shape[2]

    @property
    def heads_k(self) -> int:
        return self.k_shape[2]

    @property
    def head_dim(self) -> int:
        return self.q_shape[3]

    @property
    def q_lengths(self) -> tuple[int, ...]:
        return _shape_lengths(self.q_shape, name="Q")

    @property
    def k_lengths(self) -> tuple[int, ...]:
        return _shape_lengths(self.k_shape, name="K")

    @property
    def attention_scale(self) -> float:
        base = self.scale_softmax
        if base == 0.0:
            base = 1.0 / math.sqrt(self.head_dim)
        return self.scale_q * self.scale_k * base

    @property
    def attention_scale_log2(self) -> float:
        return self.attention_scale * math.log2(math.e)

    @property
    def output_scale(self) -> float:
        return self.scale_v * self.inv_scale_o

    @property
    def effective_window(self) -> tuple[int | None, int | None]:
        left, right = self.window_size
        left_value = None if left == -1 else left
        right_value = None if right == -1 else right
        if self.is_causal:
            right_value = 0
        return left_value, right_value

    @property
    def mask(self) -> ForwardMaskSpec:
        left, right = self.effective_window
        sequence: int | Sequence[int]
        sequence = self.k_lengths if self.variable_length else self.k_lengths[0]
        return resolve_forward_mask(
            sequence,
            is_causal=self.is_causal,
            bottom_right=self.bottom_right_align,
            window_left=left,
            window_right=right,
        )

    def fixed_lengths(self) -> tuple[int, int]:
        if self.variable_length:
            raise ValueError("fixed_lengths() is not valid for a varlen problem")
        return self.q_lengths[0], self.k_lengths[0]

    def validate(self) -> None:
        if len(self.q_shape) != 4 or len(self.k_shape) != 4:
            raise ValueError("Q and K shapes must have four fields: B,S,H,D")
        bq, sq, hq, dq = self.q_shape
        bk, sk, hk, dk = self.k_shape
        scalar_dimensions = (bq, hq, dq, bk, hk, dk)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in scalar_dimensions
        ):
            raise ValueError(
                "all non-sequence shape dimensions must be positive integers"
            )
        if bq != bk:
            raise ValueError("Q and K batch dimensions must match")
        if dq != dk:
            raise ValueError("Q and K head dimensions must match")
        if dq not in (32, 64, 128):
            raise ValueError("head dimension must be 32, 64, or 128")
        if hq % hk:
            raise ValueError("Hq must be divisible by Hk")
        if isinstance(sq, tuple) != isinstance(sk, tuple):
            raise ValueError("Q and K must both be fixed length or both varlen")
        q_lengths = self.q_lengths
        k_lengths = self.k_lengths
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (*q_lengths, *k_lengths)
        ):
            raise ValueError("all sequence lengths must be positive integers")
        if self.in_dtype not in SUPPORTED_DTYPE_NAMES:
            raise ValueError(f"in_dtype must be one of {SUPPORTED_DTYPE_NAMES}")
        if self.out_dtype not in SUPPORTED_DTYPE_NAMES:
            raise ValueError(f"out_dtype must be one of {SUPPORTED_DTYPE_NAMES}")
        if self.qk_acc_dtype != "Float32" or self.pv_acc_dtype != "Float32":
            raise ValueError("QK and PV accumulator dtypes must both be Float32")
        if self.mma_tiler_mn != DEFAULT_MMA_TILER:
            raise ValueError("mma_tiler_mn must be (128, 128)")
        if len(self.window_size) != 2 or any(
            not isinstance(value, int) or value < -1 for value in self.window_size
        ):
            raise ValueError("window_size must contain two integers >= -1")
        scale_names = (
            "scale_q",
            "scale_k",
            "scale_v",
            "inv_scale_o",
            "scale_softmax",
        )
        for name in scale_names:
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.use_sinks and self.attention_scale == 0.0:
            raise ValueError("sink logits require a nonzero attention scale")

        mask = self.mask
        for seq_q, seq_k in zip(q_lengths, k_lengths):
            # Preserve the source diagnostic: interior empty rows are rejected.
            if mask.kind is ForwardMask.RESIDUAL:
                continue
            offset = bottom_right_offset(mask.kind, seq_q, seq_k)
            for query in range(1, max(1, seq_q - 1)):
                lower = (
                    0
                    if mask.window_left is None
                    else max(0, query + offset - mask.window_left)
                )
                upper = (
                    seq_k - 1
                    if mask.window_right is None
                    else min(seq_k - 1, query + offset + mask.window_right)
                )
                if lower > upper:
                    raise ValueError(
                        f"sliding window removes every key for query row {query}"
                    )


def coerce_config(config: FmhaConfig | Any) -> FmhaConfig:
    """Normalize a same-field config owned by a launch module.

    This lets ``fmha_prefill.py`` adopt the support module incrementally: its
    existing ``FmhaConfig`` can be passed directly before the class definition
    itself is eventually centralized here.
    """

    if isinstance(config, FmhaConfig):
        return config
    values = {}
    for field in fields(FmhaConfig):
        if not hasattr(config, field.name):
            raise TypeError(f"config is missing required field {field.name!r}")
        values[field.name] = getattr(config, field.name)
    return FmhaConfig(**values)


@dataclass(frozen=True)
class TensorPlan:
    """Allocation and layout plan independent of a concrete tensor device."""

    config: FmhaConfig
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    cumulative_q: tuple[int, ...] | None
    cumulative_k: tuple[int, ...] | None
    q_storage_shape: tuple[int, ...]
    kv_storage_shape: tuple[int, ...]
    output_storage_shape: tuple[int, ...]
    lse_storage_shape: tuple[int, ...]
    mask: ForwardMaskSpec

    @property
    def variable_length(self) -> bool:
        return self.cumulative_q is not None

    @property
    def total_q(self) -> int:
        return sum(self.q_lengths)

    @property
    def total_k(self) -> int:
        return sum(self.k_lengths)

    def q_slice(self, batch: int) -> slice:
        if not 0 <= batch < self.config.batch:
            raise IndexError(f"batch {batch} is outside [0, {self.config.batch})")
        if self.cumulative_q is None:
            return slice(batch, batch + 1)
        return slice(self.cumulative_q[batch], self.cumulative_q[batch + 1])

    def k_slice(self, batch: int) -> slice:
        if not 0 <= batch < self.config.batch:
            raise IndexError(f"batch {batch} is outside [0, {self.config.batch})")
        if self.cumulative_k is None:
            return slice(batch, batch + 1)
        return slice(self.cumulative_k[batch], self.cumulative_k[batch + 1])


def make_tensor_plan(config: FmhaConfig | Any) -> TensorPlan:
    """Validate ``config`` and derive fixed or flattened-varlen storage shapes."""

    config = coerce_config(config)
    config.validate()
    q_lengths = config.q_lengths
    k_lengths = config.k_lengths
    if config.variable_length:
        cumulative_q = cumulative_offsets(q_lengths)
        cumulative_k = cumulative_offsets(k_lengths)
        q_shape = (sum(q_lengths), config.heads_q, config.head_dim)
        kv_shape = (sum(k_lengths), config.heads_k, config.head_dim)
        lse_shape = (1, config.heads_q, sum(q_lengths))
    else:
        cumulative_q = None
        cumulative_k = None
        q_shape = (config.batch, q_lengths[0], config.heads_q, config.head_dim)
        kv_shape = (config.batch, k_lengths[0], config.heads_k, config.head_dim)
        lse_shape = (config.batch, config.heads_q, q_lengths[0])
    return TensorPlan(
        config=config,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        cumulative_q=cumulative_q,
        cumulative_k=cumulative_k,
        q_storage_shape=q_shape,
        kv_storage_shape=kv_shape,
        output_storage_shape=q_shape,
        lse_storage_shape=lse_shape,
        mask=config.mask,
    )


@dataclass
class PreparedTensors:
    """Concrete tensors plus the source-compatible FP32 reference inputs."""

    plan: TensorPlan
    q: Any
    k: Any
    v: Any
    out: Any
    lse: Any | None
    sinks: Any | None
    cumulative_q: Any | None
    cumulative_k: Any | None
    q_reference: Any
    k_reference: Any
    v_reference: Any
    sinks_reference: Any | None

    @property
    def config(self) -> FmhaConfig:
        return self.plan.config

    def as_mapping(self) -> dict[str, Any]:
        """Return names suitable for a thin launch adapter in ``fmha_prefill.py``."""

        return {
            "q": self.q,
            "k": self.k,
            "v": self.v,
            "out": self.out,
            "lse": self.lse,
            "sinks": self.sinks,
            "cum_seqlen_q": self.cumulative_q,
            "cum_seqlen_k": self.cumulative_k,
            "q_reference": self.q_reference,
            "k_reference": self.k_reference,
            "v_reference": self.v_reference,
            "sinks_reference": self.sinks_reference,
        }


def _input_values(value: Any, shape: tuple[int, ...], name: str) -> Any:
    torch_module = _require_torch()
    if isinstance(value, torch_module.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch_module.float32)
    else:
        tensor = torch_module.as_tensor(value, dtype=torch_module.float32)
    expected = math.prod(shape)
    if tensor.numel() != expected:
        raise ValueError(f"{name} has {tensor.numel()} values; expected {expected}")
    return tensor.reshape(shape).contiguous()


def prepare_tensors(
    config: FmhaConfig | Any,
    *,
    device: str | Any = "cuda",
    seed: int = 1111,
    q_input: Any | None = None,
    k_input: Any | None = None,
    v_input: Any | None = None,
    sinks_input: Any | None = None,
) -> PreparedTensors:
    """Allocate fixed or flattened-varlen tensors on ``device``.

    Custom Q/K/V inputs are interpreted in storage order and must be supplied
    together.  Reference inputs retain the original FP32 source values, matching
    the CUTLASS example's ``create_and_pad_tensor`` contract.
    """

    torch_module = _require_torch()
    plan = make_tensor_plan(config)
    config = plan.config
    supplied = (q_input is not None, k_input is not None, v_input is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("q_input, k_input, and v_input must be provided together")
    generator = torch_module.Generator(device="cpu").manual_seed(seed)
    if all(supplied):
        q_source = _input_values(q_input, plan.q_storage_shape, "q_input")
        k_source = _input_values(k_input, plan.kv_storage_shape, "k_input")
        v_source = _input_values(v_input, plan.kv_storage_shape, "v_input")
    else:
        q_source = torch_module.randn(
            plan.q_storage_shape, generator=generator, dtype=torch_module.float32
        )
        k_source = torch_module.randn(
            plan.kv_storage_shape, generator=generator, dtype=torch_module.float32
        )
        v_source = torch_module.randn(
            plan.kv_storage_shape, generator=generator, dtype=torch_module.float32
        )

    input_dtype = torch_dtype(config.in_dtype)
    output_dtype = torch_dtype(config.out_dtype)
    q = q_source.to(device=device, dtype=input_dtype)
    k = k_source.to(device=device, dtype=input_dtype)
    v = v_source.to(device=device, dtype=input_dtype)
    out = torch_module.empty(
        plan.output_storage_shape, device=device, dtype=output_dtype
    )
    lse = (
        torch_module.empty(
            plan.lse_storage_shape, device=device, dtype=torch_module.float32
        )
        if config.lse_calculation
        else None
    )

    sinks = None
    sinks_reference = None
    if config.use_sinks:
        if sinks_input is None:
            sink_generator = torch_module.Generator(device=device).manual_seed(seed)
            sinks = torch_module.randn(
                (config.heads_q,),
                generator=sink_generator,
                device=device,
                dtype=torch_module.float16,
            )
        else:
            sink_source = _input_values(sinks_input, (config.heads_q,), "sinks_input")
            sinks = sink_source.to(device=device, dtype=torch_module.float16)
        sinks_reference = sinks.detach().to(dtype=torch_module.float32)
    elif sinks_input is not None:
        raise ValueError("sinks_input requires config.use_sinks=True")

    cumulative_q_tensor = None
    cumulative_k_tensor = None
    if plan.variable_length:
        cumulative_q_tensor = torch_module.tensor(
            plan.cumulative_q, device=device, dtype=torch_module.int32
        )
        cumulative_k_tensor = torch_module.tensor(
            plan.cumulative_k, device=device, dtype=torch_module.int32
        )

    return PreparedTensors(
        plan=plan,
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        sinks=sinks,
        cumulative_q=cumulative_q_tensor,
        cumulative_k=cumulative_k_tensor,
        q_reference=q_source.to(device=device, dtype=torch_module.float32),
        k_reference=k_source.to(device=device, dtype=torch_module.float32),
        v_reference=v_source.to(device=device, dtype=torch_module.float32),
        sinks_reference=sinks_reference,
    )


def _batch_values(tensor: Any, plan: TensorPlan, batch: int, *, query: bool) -> Any:
    if plan.variable_length:
        offsets = plan.cumulative_q if query else plan.cumulative_k
        assert offsets is not None
        return tensor[offsets[batch]:offsets[batch + 1]]
    return tensor[batch]


def score_keep_mask(
    plan: TensorPlan,
    batch: int,
    *,
    query_start: int = 0,
    query_end: int | None = None,
    device: str | Any | None = None,
) -> Any:
    """Vectorized keep mask matching :func:`is_score_element_valid`."""

    torch_module = _require_torch()
    seq_q = plan.q_lengths[batch]
    seq_k = plan.k_lengths[batch]
    if query_end is None:
        query_end = seq_q
    if not 0 <= query_start <= query_end <= seq_q:
        raise ValueError("query range is outside the selected sequence")
    if plan.mask.kind is ForwardMask.RESIDUAL:
        return torch_module.ones(
            (query_end - query_start, seq_k), device=device, dtype=torch_module.bool
        )
    query = torch_module.arange(
        query_start, query_end, device=device, dtype=torch_module.int64
    ).view(-1, 1)
    key = torch_module.arange(seq_k, device=device, dtype=torch_module.int64).view(
        1, -1
    )
    offset = bottom_right_offset(plan.mask.kind, seq_q, seq_k)
    keep = torch_module.ones(
        (query_end - query_start, seq_k), device=device, dtype=torch_module.bool
    )
    if plan.mask.window_left is not None:
        keep &= key >= query + offset - plan.mask.window_left
    if plan.mask.window_right is not None:
        keep &= key <= query + offset + plan.mask.window_right
    return keep


@dataclass(frozen=True)
class ReferenceResult:
    output: Any
    lse: Any | None


def narrow_reference_output(output: Any, out_dtype: str) -> Any:
    """Apply the source's explicit narrow-then-widen step for FP8 output."""

    torch_module = _require_torch()
    if out_dtype == "Float8E4M3FN":
        return output.to(torch_dtype(out_dtype)).to(torch_module.float32)
    return output


def torch_reference(
    tensors: PreparedTensors,
    *,
    query_chunk_size: int = 128,
) -> ReferenceResult:
    """Compute source-style attention from the original FP32 input values."""

    torch_module = _require_torch()
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    config = tensors.config
    plan = tensors.plan
    outputs = []
    lse_batches = []
    for batch in range(config.batch):
        q_values = _batch_values(tensors.q_reference, plan, batch, query=True)
        k_values = _batch_values(tensors.k_reference, plan, batch, query=False)
        v_values = _batch_values(tensors.v_reference, plan, batch, query=False)
        if config.heads_q != config.heads_k:
            ratio = config.heads_q // config.heads_k
            k_values = k_values.repeat_interleave(ratio, dim=1)
            v_values = v_values.repeat_interleave(ratio, dim=1)
        q_heads = q_values.transpose(0, 1)
        k_heads = k_values.transpose(0, 1)
        v_heads = v_values.transpose(0, 1)
        output_chunks = []
        lse_chunks = []
        for start in range(0, plan.q_lengths[batch], query_chunk_size):
            end = min(start + query_chunk_size, plan.q_lengths[batch])
            scores = (
                torch_module.einsum("hqd,hkd->hqk", q_heads[:, start:end], k_heads)
                * config.attention_scale
            )
            keep = score_keep_mask(
                plan,
                batch,
                query_start=start,
                query_end=end,
                device=scores.device,
            )
            scores = scores.masked_fill(~keep.unsqueeze(0), -torch_module.inf)
            if config.use_sinks:
                assert tensors.sinks_reference is not None
                sink_logits = tensors.sinks_reference.to(
                    dtype=scores.dtype, device=scores.device
                ).view(config.heads_q, 1, 1)
                logits = torch_module.cat(
                    (sink_logits.expand(-1, end - start, -1), scores), dim=-1
                )
                probabilities = torch_module.softmax(logits, dim=-1)[..., 1:]
            else:
                logits = scores
                probabilities = torch_module.softmax(scores, dim=-1)
            output = (
                torch_module.einsum("hqk,hkd->qhd", probabilities, v_heads)
                * config.output_scale
            )
            output_chunks.append(output)
            if config.lse_calculation:
                lse_chunks.append(torch_module.logsumexp(logits, dim=-1))
        outputs.append(torch_module.cat(output_chunks, dim=0))
        if config.lse_calculation:
            lse_batches.append(torch_module.cat(lse_chunks, dim=1))

    if plan.variable_length:
        output_result = torch_module.cat(outputs, dim=0)
        lse_result = (
            torch_module.cat(lse_batches, dim=1).unsqueeze(0)
            if config.lse_calculation
            else None
        )
    else:
        output_result = torch_module.stack(outputs)
        lse_result = torch_module.stack(lse_batches) if config.lse_calculation else None
    return ReferenceResult(
        narrow_reference_output(output_result, config.out_dtype), lse_result
    )


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    max_rel: float
    mismatched: int
    elements: int
    atol: float
    rtol: float

    @property
    def passed(self) -> bool:
        return self.mismatched == 0


@dataclass(frozen=True)
class VerificationResult:
    output: ErrorMetrics
    lse: ErrorMetrics | None
    reference: ReferenceResult

    @property
    def passed(self) -> bool:
        return self.output.passed and (self.lse is None or self.lse.passed)


def effective_tolerance(config: FmhaConfig, tolerance: float) -> float:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if config.out_dtype == "Float8E4M3FN":
        return max(FP8_MINIMUM_TOLERANCE, tolerance)
    return tolerance


def error_metrics(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float = 1.0e-5,
) -> ErrorMetrics:
    torch_module = _require_torch()
    actual_f32 = actual.detach().to(dtype=torch_module.float32)
    expected_f32 = expected.detach().to(
        device=actual_f32.device, dtype=torch_module.float32
    )
    if actual_f32.shape != expected_f32.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual_f32.shape)}, "
            f"expected={tuple(expected_f32.shape)}"
        )
    difference = (actual_f32 - expected_f32).abs()
    denominator = expected_f32.abs().clamp_min(
        torch_module.finfo(torch_module.float32).tiny
    )
    close = torch_module.isclose(
        actual_f32, expected_f32, atol=atol, rtol=rtol, equal_nan=False
    )
    return ErrorMetrics(
        max_abs=float(difference.max().item()),
        max_rel=float((difference / denominator).max().item()),
        mismatched=int((~close).sum().item()),
        elements=actual_f32.numel(),
        atol=atol,
        rtol=rtol,
    )


def verify(
    tensors: PreparedTensors,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    rtol: float = 1.0e-5,
    query_chunk_size: int = 128,
    raise_on_error: bool = True,
) -> VerificationResult:
    """Reference-check output and optional LSE, returning useful metrics."""

    torch_module = _require_torch()
    reference = torch_reference(tensors, query_chunk_size=query_chunk_size)
    atol = effective_tolerance(tensors.config, tolerance)
    output_metrics = error_metrics(tensors.out, reference.output, atol=atol, rtol=rtol)
    lse_metrics = None
    if tensors.config.lse_calculation:
        if tensors.lse is None or reference.lse is None:
            raise ValueError(
                "LSE is enabled but actual or reference storage is missing"
            )
        lse_metrics = error_metrics(tensors.lse, reference.lse, atol=atol, rtol=rtol)
    result = VerificationResult(output_metrics, lse_metrics, reference)
    if raise_on_error and not result.passed:
        torch_module.testing.assert_close(
            tensors.out.float(), reference.output, atol=atol, rtol=rtol
        )
        if lse_metrics is not None:
            torch_module.testing.assert_close(
                tensors.lse.float(), reference.lse, atol=atol, rtol=rtol
            )
    return result


@dataclass(frozen=True)
class PrefillCorrectnessTestCase:
    name: str
    q_shape: ShapeSpec
    k_shape: ShapeSpec
    in_dtype: str = "Float16"
    out_dtype: str = "Float16"
    is_causal: bool = False
    bottom_right_align: bool = False
    window_size: tuple[int, int] = (-1, -1)

    def config(self) -> FmhaConfig:
        return FmhaConfig(
            q_shape=self.q_shape,
            k_shape=self.k_shape,
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
            is_causal=self.is_causal,
            bottom_right_align=self.bottom_right_align,
            window_size=self.window_size,
        )


PREFILL_CORRECTNESS_TESTS = (
    PrefillCorrectnessTestCase("shape=2,512,8,128", (2, 512, 8, 128), (2, 512, 8, 128)),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64", (2, 512, 64, 64), (2, 512, 8, 64)
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,32", (2, 512, 64, 32), (2, 512, 8, 32)
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 causal",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,259,64/8,128 non-power-of-2 seqlen causal",
        (2, 259, 64, 128),
        (2, 259, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,5000,64/8,128 non-power-of-2 long seqlen causal",
        (2, 5000, 64, 128),
        (2, 5000, 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,(512,256),64/8,128 var seqlen",
        (2, (512, 256), 64, 128),
        (2, (512, 256), 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,(511,768),64/8,128 non-power-of-two var seqlen",
        (2, (511, 768), 64, 128),
        (2, (511, 768), 8, 128),
        is_causal=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 fp8 input fp16 output",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        in_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,128 fp8 input fp8 output",
        (2, 512, 64, 128),
        (2, 512, 8, 128),
        in_dtype="Float8E4M3FN",
        out_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 fp8 input fp16 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 fp8 input fp8 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="Float8E4M3FN",
        out_dtype="Float8E4M3FN",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 bf16 input bf16 output",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        in_dtype="BFloat16",
        out_dtype="BFloat16",
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/97,64/8,64 decode phase",
        (2, 1, 64, 64),
        (2, 97, 8, 64),
        is_causal=True,
        bottom_right_align=True,
    ),
    PrefillCorrectnessTestCase(
        "shape=2,512,64/8,64 left-window causal",
        (2, 512, 64, 64),
        (2, 512, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,511,64/8,64 seqlen=511 left-window causal",
        (2, 511, 64, 64),
        (2, 511, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,513,64/8,64 seqlen=513 left-window causal",
        (2, 513, 64, 64),
        (2, 513, 8, 64),
        is_causal=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/97,64/8,64 left-window decode phase",
        (2, 1, 64, 64),
        (2, 97, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/251,64/8,64 seqlen=511 left-window decode phase",
        (2, 1, 64, 64),
        (2, 251, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
    PrefillCorrectnessTestCase(
        "shape=2,1/253,64/8,64 seqlen=513 left-window decode phase",
        (2, 1, 64, 64),
        (2, 253, 8, 64),
        is_causal=True,
        bottom_right_align=True,
        window_size=(128, -1),
    ),
)


def workspace_bytes(tensors: PreparedTensors) -> int:
    """Return the byte footprint used by cold-L2 workspace planning."""

    total = 0
    for value in (
        tensors.q,
        tensors.k,
        tensors.v,
        tensors.out,
        tensors.lse,
        tensors.sinks,
        tensors.cumulative_q,
        tensors.cumulative_k,
    ):
        if value is not None:
            total += value.numel() * value.element_size()
    return total


def prepared_from_mapping(
    config: FmhaConfig | Any,
    tensors: Mapping[str, Any],
) -> PreparedTensors:
    """Adapt an existing launch mapping, preserving supplied FP32 references."""

    torch_module = _require_torch()
    plan = make_tensor_plan(config)
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    return PreparedTensors(
        plan=plan,
        q=q,
        k=k,
        v=v,
        out=tensors["out"],
        lse=tensors.get("lse"),
        sinks=tensors.get("sinks"),
        cumulative_q=tensors.get("cum_seqlen_q"),
        cumulative_k=tensors.get("cum_seqlen_k"),
        q_reference=(
            tensors["q_reference"]
            if "q_reference" in tensors
            else q.detach().to(torch_module.float32)
        ),
        k_reference=(
            tensors["k_reference"]
            if "k_reference" in tensors
            else k.detach().to(torch_module.float32)
        ),
        v_reference=(
            tensors["v_reference"]
            if "v_reference" in tensors
            else v.detach().to(torch_module.float32)
        ),
        sinks_reference=(
            tensors["sinks"].detach().to(torch_module.float32)
            if tensors.get("sinks") is not None
            else None
        ),
    )
