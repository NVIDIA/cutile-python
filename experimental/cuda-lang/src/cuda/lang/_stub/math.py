# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import simt_stub
from cuda.lang._enums import RoundingMode, SaturationMode
from cuda.tile._cext import _binary_wrapping_arithmetic_tensorlike


@simt_stub(static_eval_ok=True, compiled_host=True)
def add(x, y, /, *, rounding_mode: RoundingMode | None = None,
        flush_to_zero: bool = False):
    """Compute ``x + y``.

    Args:
        rounding_mode: The rounding mode.
        flush_to_zero: Whether to flush subnormal inputs and results to
            sign-preserving zero.
    """
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def sub(x, y, /, *, rounding_mode: RoundingMode | None = None,
        flush_to_zero: bool = False):
    """Compute ``x - y``.

    Args:
        rounding_mode: The rounding mode.
        flush_to_zero: Whether to flush subnormal inputs and results to
            sign-preserving zero.
    """
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def mul(x, y, /):
    """Compute ``x * y``."""
    ...


@simt_stub(host=True, compiled_host=True)
def wrapping_add(x, y, /):
    """Compute ``x + y`` for integers, wrapping around on overflow.

    Unlike :func:`add`, the dtypes are not promoted: ``x`` and ``y`` must have the same
    integer dtype, and the result is computed modulo 2**bitwidth of that dtype.
    A loosely typed constant is converted to the dtype of the other operand,
    also modulo 2**bitwidth: for example, -1 becomes 255 for uint8.

    At least one of the operands must have a concrete dtype, since loosely typed constants
    don't have a bit width. For example, ``wrapping_add(1, 2)`` is an error,
    but ``wrapping_add(uint8(150), 110)`` is ``uint8(4)``.
    """
    return _binary_wrapping_arithmetic_tensorlike("wrapping_add", x, y, False)


@simt_stub(host=True, compiled_host=True)
def wrapping_sub(x, y, /):
    """Compute ``x - y`` for integers, wrapping around on overflow.

    Unlike :func:`sub`, the dtypes are not promoted: ``x`` and ``y`` must have the same
    integer dtype, and the result is computed modulo 2**bitwidth of that dtype.
    A loosely typed constant is converted to the dtype of the other operand,
    also modulo 2**bitwidth: for example, -1 becomes 255 for uint8.

    At least one of the operands must have a concrete dtype, since loosely typed constants
    don't have a bit width. For example, ``wrapping_sub(1, 2)`` is an error,
    but ``wrapping_sub(uint8(0), 1)`` is ``uint8(255)``.
    """
    return _binary_wrapping_arithmetic_tensorlike("wrapping_sub", x, y, False)


@simt_stub(host=True, compiled_host=True)
def wrapping_mul(x, y, /):
    """Compute ``x * y`` for integers, wrapping around on overflow.

    Unlike :func:`mul`, the dtypes are not promoted: ``x`` and ``y`` must have the same
    integer dtype, and the result is computed modulo 2**bitwidth of that dtype.
    A loosely typed constant is converted to the dtype of the other operand,
    also modulo 2**bitwidth: for example, -1 becomes 255 for uint8.

    At least one of the operands must have a concrete dtype, since loosely typed constants
    don't have a bit width. For example, ``wrapping_mul(1, 2)`` is an error,
    but ``wrapping_mul(uint8(16), 17)`` is ``uint8(16)``.
    """
    return _binary_wrapping_arithmetic_tensorlike("wrapping_mul", x, y, False)


@simt_stub
def fma(
    x,
    y,
    z,
    /,
    *,
    rounding_mode=RoundingMode.RN,
    saturation_mode=SaturationMode.NONE,
    flush_to_zero=False,
    relu=False,
    oob=False,
):
    """Compute fused ``x * y + z`` with one rounding step.

    Args:
        x: First multiplicand.
        y: Second multiplicand.
        z: Value added to ``x * y``.
        rounding_mode: The rounding mode.
        saturation_mode: The saturation mode. ``SAT`` limits the result to
            ``[0.0, 1.0]`` and flushes a NaN result to positive zero.
        flush_to_zero: Whether to flush subnormal inputs and results to
            sign-preserving zero.
        relu: Whether to apply ReLU to the result.
        oob: Whether to set the result to zero when an operand is an OOB NaN.
    """
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def truediv(x, y, /, *, approx=False):
    """Compute ``x / y``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """


@simt_stub(static_eval_ok=True, compiled_host=True)
def floordiv(x, y, /):
    """Compute ``x // y``."""
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def mod(x, y, /):
    """Compute ``x % y``."""
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def integer_remainder(x, y, /):
    """Compute the remainder of integer division following C truncation rules.
    """
    ...


@simt_stub(static_eval_ok=True, compiled_host=True)
def divmod(x, y, /):
    """Compute divmod(x, y)."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def negative(x, /):
    """Compute ``-x``."""
    ...


@simt_stub
def ceil(x, /):
    """Round ``x`` toward positive infinity."""
    ...


@simt_stub
def exp(x, /, *, approx=False):
    """Compute ``e`` raised to the power ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def exp2(x, /, *, flush_to_zero=False):
    """Compute ``2`` raised to the power ``x``.

    Args:
        flush_to_zero: If true, flush subnormal inputs and results to
            sign-preserving zero. This option requires ``float32`` operands.
    """
    ...


@simt_stub
def sin(x, /, *, approx=False):
    """Compute the sine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def cos(x, /, *, approx=False):
    """Compute the cosine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def sincos(x, /, *, approx=False):
    """Compute both the sine and cosine of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def tan(x, /, *, approx=False):
    """Compute the tangent of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def sinh(x, /):
    """Compute the hyperbolic sine of ``x``."""
    ...


@simt_stub
def cosh(x, /):
    """Compute the hyperbolic cosine of ``x``."""
    ...


@simt_stub
def tanh(x, /):
    """Compute the hyperbolic tangent of ``x``."""
    ...


@simt_stub
def sqrt(x, /):
    """Compute the square root of ``x``."""
    ...


@simt_stub
def rsqrt(x, /):
    """Compute the reciprocal square root of ``x``."""
    ...


@simt_stub
def floor(x, /):
    """Round ``x`` toward negative infinity."""
    ...


@simt_stub
def log(x, /, *, approx=False):
    """Compute the natural logarithm of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def log2(x, /, *, approx=False):
    """Compute the base-2 logarithm of ``x``.

    Args:
        approx: Whether to use an approximate implementation if one is available
            for the given operand types.
    """
    ...


@simt_stub
def abs(x, /):
    """Compute the absolute value of ``x``."""
    ...


@simt_stub
def atan2(x, y, /):
    """Compute the angle whose tangent is ``x / y``."""
    ...


@simt_stub
def isnan(x, /):
    """Return whether ``x`` is NaN."""
    ...


@simt_stub
def isinf(x, /):
    """Return whether ``x`` is positive or negative infinity."""
    ...


@simt_stub
def isfinite(x, /):
    """Return whether ``x`` is finite."""
    ...


@simt_stub
def isnormal(x, /):
    """Return whether ``x`` is a normal floating-point value."""
    ...


@simt_stub
def pow(x, y, /, *, approx=False):
    """Calculate the value of ``x`` to the power of ``y``.

    Args:
        approx: Whether to use an approximate implementation. Only supported for float32.
            Has no effect for integer exponents, which use an exact implementation.
    """
    ...


@simt_stub(static_eval_ok=True)
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


@simt_stub(static_eval_ok=True)
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


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_and(x, y, /):
    """Compute ``x & y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_or(x, y, /):
    """Compute ``x | y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_xor(x, y, /):
    """Compute ``x ^ y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_lshift(x, y, /):
    """Compute ``x << y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_rshift(x, y, /):
    """Compute ``x >> y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def bitwise_not(x, /):
    """Compute ``~x``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def greater(x, y, /):
    """Compute ``x > y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def greater_equal(x, y, /):
    """Compute ``x >= y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def less(x, y, /):
    """Compute ``x < y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def less_equal(x, y, /):
    """Compute ``x <= y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def equal(x, y, /):
    """Compute ``x == y``."""


@simt_stub(static_eval_ok=True, compiled_host=True)
def not_equal(x, y, /):
    """Compute ``x != y``."""


@simt_stub
def where(condition, x, y, /):
    """Select elements from ``x`` or ``y`` with a scalar or vector condition.

    Scalar inputs broadcast to the vector shape, if any. All vector inputs
    must have the same length. The result uses the promoted type of ``x``
    and ``y``.
    """
    ...


__all__ = (
    "add",
    "sub",
    "mul",
    "wrapping_add",
    "wrapping_sub",
    "wrapping_mul",
    "fma",
    "truediv",
    "floordiv",
    "mod",
    "integer_remainder",
    "divmod",
    "negative",
    "ceil",
    "exp",
    "exp2",
    "where",
    "sin",
    "cos",
    "sincos",
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
