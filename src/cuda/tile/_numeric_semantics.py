# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RoundingMode(Enum):
    """
    Rounding mode for floating-point operations.
    """

    RN = "nearest_even"
    """Round to nearest (ties to even)."""

    RZ = "zero"
    """Round towards zero (truncate)."""

    RM = "negative_inf"
    """Round towards negative infinity."""

    RP = "positive_inf"
    """Round towards positive infinity."""

    RA = "nearest_away"
    """Round to nearest (ties away from zero)."""

    FULL = "full"
    """Full precision rounding mode."""

    APPROX = "approx"
    """Approximate rounding mode."""

    RZI = "nearest_int_to_zero"
    """Round towards zero to the nearest integer."""


class PaddingMode(Enum):
    """
    Padding mode for load operation.
    """

    UNDETERMINED = "undetermined"
    """The padding value is not determined."""

    ZERO = "zero"
    """The padding value is zero."""

    NEG_ZERO = "neg_zero"
    """The padding value is negative zero."""

    NAN = "nan"
    """The padding value is NaN."""

    POS_INF = "pos_inf"
    """The padding value is positive infinity."""

    NEG_INF = "neg_inf"
    """The padding value is negative infinity."""
