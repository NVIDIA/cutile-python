# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.lang._execution import stub


@stub(static_eval_ok=True)
def add(x, y, /):
    """Compute ``x + y``."""
    ...


@stub(static_eval_ok=True)
def sub(x, y, /):
    """Compute ``x - y``."""
    ...


@stub(static_eval_ok=True)
def mul(x, y, /):
    """Compute ``x * y``."""
    ...


@stub(static_eval_ok=True)
def truediv(x, y, /, *, approx=False):
    """Compute ``x / y``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """


@stub(static_eval_ok=True)
def floordiv(x, y, /):
    """Compute ``x // y``."""
    ...


@stub(static_eval_ok=True)
def mod(x, y, /):
    """Compute ``x % y``."""
    ...


@stub(static_eval_ok=True)
def divmod(x, y, /):
    """Compute divmod(x, y)."""


@stub(static_eval_ok=True)
def negative(x, /):
    """Compute ``-x``."""
    ...


@stub
def ceil(x, /):
    """Round ``x`` toward positive infinity."""
    ...


@stub
def exp(x, /, *, approx=False):
    """Compute ``e`` raised to the power ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def exp2(x, /):
    """Compute ``2`` raised to the power ``x``."""
    ...


@stub
def sin(x, /, *, approx=False):
    """Compute the sine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def cos(x, /, *, approx=False):
    """Compute the cosine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def sincos(x, /, *, approx=False):
    """Compute both the sine and cosine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def tan(x, /, *, approx=False):
    """Compute the tangent of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def sinh(x, /):
    """Compute the hyperbolic sine of ``x``."""
    ...


@stub
def cosh(x, /):
    """Compute the hyperbolic cosine of ``x``."""
    ...


@stub
def tanh(x, /):
    """Compute the hyperbolic tangent of ``x``."""
    ...


@stub
def sqrt(x, /):
    """Compute the square root of ``x``."""
    ...


@stub
def rsqrt(x, /):
    """Compute the reciprocal square root of ``x``."""
    ...


@stub
def floor(x, /):
    """Round ``x`` toward negative infinity."""
    ...


@stub
def log(x, /, *, approx=False):
    """Compute the natural logarithm of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def log2(x, /, *, approx=False):
    """Compute the base-2 logarithm of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub
def abs(x, /):
    """Compute the absolute value of ``x``."""
    ...


@stub
def atan2(x, y, /):
    """Compute the angle whose tangent is ``x / y``."""
    ...


@stub
def isnan(x, /):
    """Return whether ``x`` is NaN."""
    ...


@stub
def isinf(x, /):
    """Return whether ``x`` is positive or negative infinity."""
    ...


@stub
def isfinite(x, /):
    """Return whether ``x`` is finite."""
    ...


@stub
def isnormal(x, /):
    """Return whether ``x`` is a normal floating-point value."""
    ...


@stub
def pow(x, y, /, *, approx=False):
    """Calculate the value of ``x`` to the power of ``y``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@stub(static_eval_ok=True)
def maximum(x, y, /, *, propagate_nan=False):
    """Compute the element-wise maximum of ``x`` and ``y``.

    Args:
        propagate_nan: If ``True``, the result is ``NaN`` whenever either
            operand is ``NaN`` (IEEE-754 ``maximum``). If ``False`` (the
            default), a ``NaN`` operand is ignored and the other operand is
            returned, matching C ``fmax`` (IEEE-754 ``maximumNumber``).
            Has no effect for integer operands.
    """
    ...


@stub(static_eval_ok=True)
def minimum(x, y, /, *, propagate_nan=False):
    """Compute the element-wise minimum of ``x`` and ``y``.

    Args:
        propagate_nan: If ``True``, the result is ``NaN`` whenever either
            operand is ``NaN`` (IEEE-754 ``minimum``). If ``False`` (the
            default), a ``NaN`` operand is ignored and the other operand is
            returned, matching C ``fmin`` (IEEE-754 ``minimumNumber``).
            Has no effect for integer operands.
    """
    ...


@stub(static_eval_ok=True)
def bitwise_and(x, y, /):
    """Compute ``x & y``."""


@stub(static_eval_ok=True)
def bitwise_or(x, y, /):
    """Compute ``x | y``."""


@stub(static_eval_ok=True)
def bitwise_xor(x, y, /):
    """Compute ``x ^ y``."""


@stub(static_eval_ok=True)
def bitwise_not(x, /):
    """Compute ``~x``."""


@stub(static_eval_ok=True)
def greater(x, y, /):
    """Compute ``x > y``."""


@stub(static_eval_ok=True)
def greater_equal(x, y, /):
    """Compute ``x >= y``."""


@stub(static_eval_ok=True)
def less(x, y, /):
    """Compute ``x < y``."""


@stub(static_eval_ok=True)
def less_equal(x, y, /):
    """Compute ``x <= y``."""


@stub(static_eval_ok=True)
def equal(x, y, /):
    """Compute ``x == y``."""


@stub(static_eval_ok=True)
def not_equal(x, y, /):
    """Compute ``x != y``."""


__all__ = (
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "negative",
    "ceil",
    "exp",
    "exp2",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "sqrt",
    "rsqrt",
    "floor",
    "log",
    "log2",
    "abs",
    "atan2",
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
    "pow",
    "maximum",
    "minimum",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
)
