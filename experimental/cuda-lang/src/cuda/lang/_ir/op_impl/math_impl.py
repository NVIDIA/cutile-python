# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import operator

from cuda.tile._ir.ir import add_operation_variadic
from cuda.tile._ir.ops_utils import promote_dtypes

import cuda.lang._datatype as datatype
import cuda.lang._mlir as mlir
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.ir import Var, add_operation
from .vector_impl import vector_elementwise_apply
from cuda.lang._ir.type import (
    ScalarTy,
    TensorLikeTy,
    VectorTy,
)
from cuda.lang._ir.op_defs import (
    RawNVVMIntrinsic,
    RawMLIROperation,
    ForeignFunction,
)
from cuda.lang._ir.type_checking_helpers import (
    broadcast_to_same_shape,
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
    binary_bitwise_tensorlike,
    compare_tensorlike,
    mod_tensorlike,
    promote_and_broadcast_to,
    unary,
    where, invert_tensorlike, divmod_tensorlike,
)
from cuda.tile._ir.core_ops import build_tuple, strictly_typed_const
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
)
from ..._stub import math as cl_math


_registry = ImplRegistry()
impl = _registry.impl


def math_impl_registry() -> ImplRegistry:
    return _registry


@impl(cdiv, fixed_args=["cdiv"])
@impl(cl_math.add, fixed_args=["add"])
@impl(cl_math.sub, fixed_args=["sub"])
@impl(cl_math.mul, fixed_args=["mul"])
@impl(cl_math.floordiv, fixed_args=["floordiv"])
def math_binary_arithmetic_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_arithmetic_tensorlike(fn, x, y)


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

    if isinstance(ty, ScalarTy):
        value = fast_fdividef(x, y)
    else:
        value = vector_elementwise_apply(fast_fdividef, x, y)
    return astype(value, dtype)


@impl(cl_math.bitwise_and, fixed_args=["and_"])
@impl(cl_math.bitwise_or, fixed_args=["or_"])
@impl(cl_math.bitwise_xor, fixed_args=["xor"])
def math_binary_bitwise_impl(fn: str, x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return binary_bitwise_tensorlike(fn, x, y)


@impl(cl_math.bitwise_not)
def math_bitwise_not_impl(x: Var):
    require_scalar_or_vector_type(x)
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


def get_libdevice_fmod_function(dtype):
    match dtype:
        case datatype.float32:
            entrypoint = "__nv_fmodf"
        case datatype.float64:
            entrypoint = "__nv_fmod"
        case _:
            raise TypeCheckingError(f"mod is not valid for dtype {dtype}")
    return lambda x, y: add_operation(
        ForeignFunction,
        ScalarTy(dtype),
        function_name=entrypoint,
        operands_=(x, y),
    )


def float_modulo_with_corrected_sign(value: Var, y: Var) -> Var:
    """
    Python modulo takes the sign from the second operand

    https://docs.python.org/3.14/reference/expressions.html
    > The modulo operator always yields a result with the same sign as its
    > second operand

    but libdevice's mod gives the same sign as the first argument, so we correct
    the sign after calling libdevice.

    https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmod.html
    """
    ty = value.get_type()
    assert ty == y.get_type()
    zero = strictly_typed_const(0, ty)
    value_sign = compare_tensorlike("lt", value, zero)
    y_sign = compare_tensorlike("lt", y, zero)

    sign_mismatch = binary_bitwise_tensorlike("xor", value_sign, y_sign)
    fixed_value = binary_arithmetic_tensorlike("add", value, y)
    corrected_value = where(sign_mismatch, fixed_value, value)

    value_is_zero = compare_tensorlike("eq", value, zero)
    negative_zero = unary("neg", UNARY_INT_FLOAT, zero)
    signed_zero = where(y_sign, negative_zero, zero)
    return where(value_is_zero, signed_zero, corrected_value)


@impl(operator.mod, overload=(TensorLikeTy, TensorLikeTy))
@impl(cl_math.mod)
def math_mod_impl(x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    ty = common_type(x, y)
    dtype = ty.tensor_dtype()
    if not is_float(dtype):
        return mod_tensorlike(x, y)

    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    call_dtype = datatype.float32 if dtype.bitwidth < 32 else dtype
    call_x = astype(x, call_dtype)
    call_y = astype(y, call_dtype)
    scalar_fn = get_libdevice_fmod_function(call_dtype)
    if isinstance(ty, ScalarTy):
        value = scalar_fn(call_x, call_y)
    else:
        value = vector_elementwise_apply(scalar_fn, call_x, call_y)
    value = astype(value, dtype)
    return float_modulo_with_corrected_sign(value, y)


@impl(cl_math.divmod)
def math_divmod_impl(x: Var, y: Var):
    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    return divmod_tensorlike(x, y)


@impl(cl_math.negative)
def math_negative_impl(x: Var):
    return unary("neg", UNARY_INT_FLOAT, x)


@impl(cl_math.ceil, fixed_args=["math.ceil"])
@impl(cl_math.exp2, fixed_args=["math.exp2"])
@impl(cl_math.sinh, fixed_args=["math.sinh"])
@impl(cl_math.cosh, fixed_args=["math.cosh"])
@impl(cl_math.tanh, fixed_args=["math.tanh"])
@impl(cl_math.sqrt, fixed_args=["math.sqrt"])
@impl(cl_math.rsqrt, fixed_args=["math.rsqrt"])
@impl(cl_math.floor, fixed_args=["math.floor"])
def math_float_unary_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )


def _fastmath_attributes(approx: Var):
    approx = require_constant_bool(approx)
    flags = mlir.arith.FastMathFlags.afn if approx else mlir.arith.FastMathFlags.none
    return (("fastmath", mlir.arith.FastMathFlagsAttr(value=flags)),)


@impl(cl_math.exp, fixed_args=["math.exp"])
@impl(cl_math.sin, fixed_args=["math.sin"])
@impl(cl_math.cos, fixed_args=["math.cos"])
@impl(cl_math.tan, fixed_args=["math.tan"])
@impl(cl_math.log, fixed_args=["math.log"])
@impl(cl_math.log2, fixed_args=["math.log2"])
def math_float_approx_unary_impl(op_name: str, x: Var, approx: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
        mlir_attributes=_fastmath_attributes(approx),
    )


@impl(cl_math.sincos)
def math_sincos_impl(x: Var, approx: Var):
    x_ty = require_scalar_type(x, is_float)
    results = add_operation_variadic(
        RawMLIROperation,
        (x_ty, x_ty),
        op_name="math.sincos",
        operands_=(x,),
        mlir_attributes=_fastmath_attributes(approx),
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
        RawNVVMIntrinsic,
        res_ty,
        intrinsic="llvm.is.fpclass",
        operands_=(x, mask),
    )


@impl(cl_math.isnan, fixed_args=["math.isnan"])
@impl(cl_math.isinf, fixed_args=["math.isinf"])
@impl(cl_math.isfinite, fixed_args=["math.isfinite"])
def math_float_fpclass_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_float)

    # Work around MathToNVVM checking boolean result types before its vector
    # scalarization pattern can run.
    def scalar_fpclass(element: Var):
        return add_operation(
            RawMLIROperation,
            ScalarTy(datatype.bool_),
            op_name=op_name,
            operands_=(element,),
        )

    if isinstance(x_ty, ScalarTy):
        return scalar_fpclass(x)
    return vector_elementwise_apply(scalar_fpclass, x)


def _math_pow_impl(x: Var, y: Var, approx: Var):
    x_ty, y_ty = x.get_type(), y.get_type()
    base_dt, exp_dt = x_ty.tensor_dtype(), y_ty.tensor_dtype()

    # int32 is the only valid integral exponent dtype
    if is_integral(exp_dt):
        exp_dt = datatype.int32

    # integral base is promoted to float
    if is_integral(base_dt):
        # 8b-32b ints are promoted to single-precision floats
        if base_dt.bitwidth <= 32:
            base_dt = datatype.float32
        else:
            base_dt = datatype.float64

    # if either operand is half precision float, promote to single
    if base_dt == datatype.float16:
        base_dt = datatype.float32
    if exp_dt == datatype.float16:
        exp_dt = datatype.float32

    # if both operands are floats, promote to the larger one
    if is_float(base_dt) and is_float(exp_dt) and base_dt.bitwidth != exp_dt.bitwidth:
        base_dt = exp_dt = promote_dtypes(base_dt, exp_dt)

    x = astype(x, base_dt)
    y = astype(y, exp_dt)
    x, y = broadcast_to_same_shape(x, y)

    op_name = "math.fpowi" if is_integral(exp_dt) else "math.powf"
    return add_operation(
        RawMLIROperation,
        x.get_type(),
        op_name=op_name,
        operands_=(x, y),
        mlir_attributes=_fastmath_attributes(approx),
    )


@impl(cl_math.pow)
def math_pow_impl(x: Var, y: Var, approx: Var):
    return _math_pow_impl(x, y, approx)


@impl(operator.pow, overload=(TensorLikeTy, TensorLikeTy))
def math_operator_pow_impl(x: Var, y: Var):
    approx = strictly_typed_const(False, ScalarTy(datatype.bool_))
    return _math_pow_impl(x, y, approx)


@impl(cl_math.atan2, fixed_args=["math.atan2"])
def math_float_binary_impl(op_name: str, x: Var, y: Var):
    require_scalar_or_vector_float_type(x)
    require_scalar_or_vector_float_type(y)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    return add_operation(
        RawMLIROperation,
        ty,
        op_name=op_name,
        operands_=(x, y),
    )


@impl(cl_math.maximum, fixed_args=["max"])
@impl(cl_math.minimum, fixed_args=["min"])
def math_minmax_impl(kind: str, x: Var, y: Var, propagate_nan: Var) -> Var:
    propagate_nan = require_constant_bool(propagate_nan)

    require_scalar_or_vector_type(x)
    require_scalar_or_vector_type(y)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    dtype = ty.tensor_dtype()

    if datatype.is_float(dtype):
        # propagate_nan selects IEEE-754 minimum/maximum (NaN-propagating) vs
        # minimumNumber/maximumNumber (NaN-ignoring).
        if propagate_nan:
            op_name = "arith.maximumf" if kind == "max" else "arith.minimumf"
        else:
            op_name = "arith.maxnumf" if kind == "max" else "arith.minnumf"
    elif datatype.is_integral(dtype):
        if datatype.is_signed(dtype):
            op_name = "arith.maxsi" if kind == "max" else "arith.minsi"
        else:
            op_name = "arith.maxui" if kind == "max" else "arith.minui"
    else:
        raise TypeCheckingError(f"{kind}() expects arithmetic operands, got {ty}")

    return add_operation(
        RawMLIROperation,
        ty,
        op_name=op_name,
        operands_=(x, y),
    )


@impl(cl_math.abs)
def abs_impl(x: Var) -> Var:
    x_ty = require_scalar_or_vector_type(x)
    x_dtype = x_ty.tensor_dtype()
    if datatype.is_float(x_dtype):
        op_name = "math.absf"
    elif datatype.is_integral(x_dtype):
        # If it's unsigned, then the absolute value is the identity
        if not datatype.is_signed(x_dtype):
            return x
        op_name = "math.absi"
    else:
        raise TypeCheckingError(f"abs() expects an arithmetic scalar, got {x_ty}")
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )
