# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.arithmetic_ops import binop_propagate_constant
import operator

from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.type import LooselyTypedScalar
from cuda.tile._ir.ops_utils import promote_types

import cuda.lang._datatype as datatype
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ir import Var, add_operation
from .vector_impl import vector_elementwise_apply
from cuda.lang._ir.type import (
    ScalarTy,
    TensorLikeTy,
    VectorTy,
)
from cuda.lang._ir.op_defs import (
    RawLLVMIntrinsic,
    ForeignFunction,
    FmaOperation,
    MathBinaryOperation,
    MathUnaryOperation,
)
from cuda.lang._ir.type_checking_helpers import (
    common_type,
    require_scalar_type,
    require_scalar_or_vector_float_type,
    require_scalar_or_vector_type,
)
from cuda.tile._datatype import is_float, is_integral
from cuda.tile._stub import cdiv
from cuda.tile._ir.arithmetic_ops import (
    UNARY_INT_FLOAT,
    astype,
    binary_arithmetic_tensorlike,
    binary_arithmetic_tensorlike_raw,
    binary_bitwise_tensorlike,
    binary_wrapping_arithmetic_tensorlike,
    bitwise_shift_tensorlike,
    compare_tensorlike,
    promote_and_broadcast_to,
    unary,
    where, invert_tensorlike, logical_not_impl, divmod_tensorlike,
)
from cuda.tile._ir.core_ops import build_tuple, loosely_typed_const, strictly_typed_const
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_optional_constant_enum,
)
from cuda.lang._enums import RoundingMode, SaturationMode
from ..._stub import math as cl_math


_registry = ImplRegistry()
impl = _registry.impl


def math_impl_registry() -> ImplRegistry:
    return _registry


@impl(cdiv, fixed_args=["cdiv"])
@impl(cl_math.mul, fixed_args=["mul"])
@impl(cl_math.floordiv, fixed_args=["floordiv"])
def math_binary_arithmetic_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_arithmetic_tensorlike(fn, x, y)


@impl(cl_math.wrapping_add, fixed_args=["wrapping_add"])
@impl(cl_math.wrapping_sub, fixed_args=["wrapping_sub"])
@impl(cl_math.wrapping_mul, fixed_args=["wrapping_mul"])
def math_binary_wrapping_arithmetic_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_wrapping_arithmetic_tensorlike(fn, x, y)


@impl(cl_math.add, fixed_args=["add"])
@impl(cl_math.sub, fixed_args=["sub"])
def math_binary_arithmetic_with_options_impl(
    fn: str,
    x: Var,
    y: Var,
    rounding_mode: Var,
    flush_to_zero: Var,
):
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_arithmetic_tensorlike(
        fn,
        x,
        y,
        rounding_mode=rounding_mode,
        flush_to_zero=flush_to_zero,
    )


@impl(cl_math.fma)
def math_fma_impl(
    x: Var,
    y: Var,
    z: Var,
    rounding_mode: Var,
    saturation_mode: Var,
    flush_to_zero: Var,
    relu: Var,
    oob: Var,
):
    rounding_mode = require_constant_enum(rounding_mode, RoundingMode)
    saturation_mode = require_constant_enum(
        saturation_mode, SaturationMode
    )
    flush_to_zero = require_constant_bool(flush_to_zero)
    relu = require_constant_bool(relu)
    oob = require_constant_bool(oob)

    require_scalar_or_vector_float_type(x)
    require_scalar_or_vector_float_type(y)
    require_scalar_or_vector_float_type(z)
    ty = promote_types(
        common_type(x, y), z.get_loose_type(), x.ctx.typing_hooks
    )
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    z = promote_and_broadcast_to(z, ty)

    rounding_modes = (
        RoundingMode.RM,
        RoundingMode.RN,
        RoundingMode.RP,
        RoundingMode.RZ,
    )
    if rounding_mode not in rounding_modes:
        valid = ", ".join(str(mode) for mode in rounding_modes)
        raise TypeCheckingError(
            f"fma does not support {rounding_mode}. Use one of {valid}."
        )

    if saturation_mode == SaturationMode.SATFINITE:
        raise TypeCheckingError(
            "fma does not implement SaturationMode.SATFINITE yet"
        )
    return add_operation(
        FmaOperation,
        ty,
        x=x,
        y=y,
        z=z,
        rounding_mode=rounding_mode,
        saturation_mode=saturation_mode,
        flush_to_zero=flush_to_zero,
        relu=relu,
        oob=oob,
    )


@impl(cl_math.truediv)
def math_truediv_impl(x: Var, y: Var, approx: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    if not require_constant_bool(approx):
        return binary_arithmetic_tensorlike("truediv", x, y)

    ty = common_type(x, y)
    dtype = ty.tensor_dtype()
    if not is_float(dtype) or dtype == datatype.float64:
        return binary_arithmetic_tensorlike("truediv", x, y)

    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    x = astype(x, datatype.float32)
    y = astype(y, datatype.float32)

    def fast_fdividef(lhs, rhs):
        return add_operation(
            ForeignFunction,
            ScalarTy(datatype.float32),
            function_name="__nv_fast_fdividef",
            operands_=(lhs, rhs),
        )

    value = vector_elementwise_apply(fast_fdividef, x, y)
    return astype(value, dtype)


@impl(cl_math.bitwise_and, fixed_args=["and_"])
@impl(cl_math.bitwise_or, fixed_args=["or_"])
@impl(cl_math.bitwise_xor, fixed_args=["xor"])
def math_binary_bitwise_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_bitwise_tensorlike(fn, x, y)


@impl(cl_math.bitwise_lshift, fixed_args=["lshift"])
@impl(cl_math.bitwise_rshift, fixed_args=["rshift"])
def math_bitwise_shift_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return bitwise_shift_tensorlike(fn, x, y)


@impl(cl_math.bitwise_not, fixed_args=[False])
@impl(operator.invert, fixed_args=[True], overload=(TensorLikeTy,))
def math_bitwise_not_impl(use_operator: bool, x: Var):
    require_scalar_or_vector_type(x)
    if x.get_type().tensor_dtype() == datatype.bool_ and x.is_constant():
        if use_operator and isinstance(x.get_loose_type(), LooselyTypedScalar):
            return loosely_typed_const(~int(x.get_constant()))
        return logical_not_impl(x)
    return invert_tensorlike(x)


@impl(cl_math.greater, fixed_args=["gt"])
@impl(cl_math.greater_equal, fixed_args=["ge"])
@impl(cl_math.less, fixed_args=["lt"])
@impl(cl_math.less_equal, fixed_args=["le"])
@impl(cl_math.equal, fixed_args=["eq"])
@impl(cl_math.not_equal, fixed_args=["ne"])
def math_comparison_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return compare_tensorlike(fn, x, y)


@impl(cl_math.mod)
def math_mod_impl(x: Var, y: Var):
    require_scalar_or_vector_type(x, datatype.is_numeric)
    require_scalar_or_vector_type(y, datatype.is_numeric)
    return binary_arithmetic_tensorlike("mod", x, y)


@impl(cl_math.integer_remainder)
def math_integer_remainder_impl(x: Var, y: Var):
    require_scalar_or_vector_type(x, is_integral)
    require_scalar_or_vector_type(y, is_integral)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    return binary_arithmetic_tensorlike_raw("c_mod", x, y)


@impl(cl_math.divmod)
def math_divmod_impl(x: Var, y: Var):
    require_scalar_or_vector_type(x, datatype.is_numeric)
    require_scalar_or_vector_type(y, datatype.is_numeric)
    return divmod_tensorlike(x, y)


@impl(cl_math.negative)
def math_negative_impl(x: Var):
    return unary("neg", UNARY_INT_FLOAT, x)


@impl(cl_math.exp2)
def math_exp2_impl(x: Var, flush_to_zero: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    if not require_constant_bool(flush_to_zero):
        return add_operation(MathUnaryOperation, x_ty, fn="exp2", x=x)

    dtype = x_ty.tensor_dtype()
    if dtype != datatype.float32:
        raise TypeCheckingError(
            "Flush to zero for exp2 requires float32 operands, "
            f"but got {dtype}"
        )

    def exp2_ftz(value):
        return add_operation(
            MathUnaryOperation,
            ScalarTy(datatype.float32),
            fn="exp2",
            x=value,
            flush_to_zero=True,
        )

    return vector_elementwise_apply(exp2_ftz, x)


@impl(cl_math.ceil, fixed_args=["ceil"])
@impl(cl_math.sinh, fixed_args=["sinh"])
@impl(cl_math.cosh, fixed_args=["cosh"])
@impl(cl_math.tanh, fixed_args=["tanh"])
@impl(cl_math.sqrt, fixed_args=["sqrt"])
@impl(cl_math.rsqrt, fixed_args=["rsqrt"])
@impl(cl_math.floor, fixed_args=["floor"])
def math_float_unary_impl(fn: str, x: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(MathUnaryOperation, x_ty, fn=fn, x=x)


@impl(cl_math.exp, fixed_args=["exp"])
@impl(cl_math.sin, fixed_args=["sin"])
@impl(cl_math.cos, fixed_args=["cos"])
@impl(cl_math.tan, fixed_args=["tan"])
@impl(cl_math.log, fixed_args=["log"])
@impl(cl_math.log2, fixed_args=["log2"])
def math_float_approx_unary_impl(fn: str, x: Var, approx: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(
        MathUnaryOperation,
        x_ty,
        fn=fn,
        x=x,
        approx=require_constant_bool(approx),
    )


@impl(cl_math.sincos)
def math_sincos_impl(x: Var, approx: Var):
    x_ty = require_scalar_type(x, is_float)
    results = add_operation_variadic(
        MathUnaryOperation,
        (x_ty, x_ty),
        fn="sincos",
        x=x,
        approx=require_constant_bool(approx),
    )
    return build_tuple(results)


@impl(cl_math.isnormal)
def math_isnormal_impl(x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_unrestricted_float)
    match x_ty:
        case ScalarTy():
            res_ty = ScalarTy(datatype.bool_)
        case VectorTy(length=length):
            res_ty = VectorTy(datatype.bool_, length=length)
        case _:
            assert False
    # see https://llvm.org/docs/LangRef.html#llvm-is-fpclass-intrinsic
    mask = strictly_typed_const((1 << 3) | (1 << 8), ScalarTy(datatype.int32))
    return add_operation(
        RawLLVMIntrinsic,
        res_ty,
        intrinsic="llvm.is.fpclass",
        operands_=(x, mask),
    )


@impl(cl_math.isnan, fixed_args=["isnan"])
@impl(cl_math.isinf, fixed_args=["isinf"])
@impl(cl_math.isfinite, fixed_args=["isfinite"])
def math_float_fpclass_impl(fn: str, x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_float)

    # Work around MathToNVVM checking boolean result types before its vector
    # scalarization pattern can run.
    def scalar_fpclass(element: Var):
        return add_operation(
            MathUnaryOperation,
            ScalarTy(datatype.bool_),
            fn=fn,
            x=element,
        )

    if isinstance(x_ty, ScalarTy):
        return scalar_fpclass(x)
    return vector_elementwise_apply(scalar_fpclass, x)


@impl(cl_math.pow)
def math_pow_impl(x: Var, y: Var, approx: Var):
    # The approximate implementation doesn't support negative bases, so it is not used for
    # integer exponents. Instead, these use FPowI, which is exact.
    rounding_mode = None
    if require_constant_bool(approx) and is_float(y.get_loose_type().tensor_dtype()):
        dtype = common_type(x, y).tensor_dtype()
        if dtype != datatype.float32:
            raise TypeCheckingError(
                f"pow(approx=True) is only supported for float32, not {dtype}"
            )
        rounding_mode = RoundingMode.APPROX
    return binary_arithmetic_tensorlike("pow", x, y, rounding_mode)


@impl(cl_math.atan2, fixed_args=["atan2"])
def math_float_binary_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_float_type(x)
    require_scalar_or_vector_float_type(y)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    return add_operation(
        MathBinaryOperation,
        ty,
        fn=fn,
        lhs=x,
        rhs=y,
    )


def minmax_impl(
    fn: str, x: Var, y: Var, propagate_nan: bool
) -> Var:
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)

    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    dtype = ty.tensor_dtype()

    if x.is_constant() and y.is_constant():
        return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), None)

    if not (datatype.is_float(dtype) or datatype.is_integral(dtype)):
        raise TypeCheckingError(f"{fn}() expects arithmetic operands, got {ty}")

    return add_operation(
        MathBinaryOperation,
        ty,
        fn=fn,
        lhs=x,
        rhs=y,
        propagate_nan=propagate_nan,
    )


@impl(cl_math.maximum, fixed_args=["max"])
@impl(cl_math.minimum, fixed_args=["min"])
def math_minmax_impl(fn: str, x: Var, y: Var, propagate_nan: Var) -> Var:
    return minmax_impl(
        fn, x, y, require_constant_bool(propagate_nan)
    )


@impl(max, fixed_args=["max"], overload=(TensorLikeTy, TensorLikeTy))
@impl(min, fixed_args=["min"], overload=(TensorLikeTy, TensorLikeTy))
def builtin_minmax_impl(fn: str, x: Var, y: Var) -> Var:
    return minmax_impl(fn, x, y, False)


@impl(cl_math.abs)
def abs_impl(x: Var) -> Var:
    x_ty = require_scalar_or_vector_type(x)
    x_dtype = x_ty.tensor_dtype()
    if datatype.is_integral(x_dtype):
        # If it's unsigned, then the absolute value is the identity
        if not datatype.is_signed(x_dtype):
            return x
    elif not datatype.is_float(x_dtype):
        raise TypeCheckingError(f"abs() expects an arithmetic scalar, got {x_ty}")
    return add_operation(
        MathUnaryOperation,
        x_ty,
        fn="abs",
        x=x,
    )


@impl(cl_math.where)
def math_where_impl(condition: Var, x: Var, y: Var):
    require_scalar_or_vector_type(condition, lambda dtype: dtype == datatype.bool_)
    require_scalar_or_vector_type(x, datatype.is_numeric)
    require_scalar_or_vector_type(y, datatype.is_numeric)
    lengths = {value.get_type().length for value in (condition, x, y)
               if isinstance(value.get_type(), VectorTy)}
    if len(lengths) > 1:
        raise TypeCheckingError("where vector inputs must have equal lengths")
    return where(condition, x, y)
