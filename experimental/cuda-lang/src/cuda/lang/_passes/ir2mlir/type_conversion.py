# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch, partial
from typing import Any

import cuda.lang._ir.type as ir_type
import cuda.lang._mlir as mlir
import cuda.lang._datatype as datatype
from cuda.tile._datatype import is_pointer_dtype, PointerInfo
from cuda.lang._exception import InternalError


@singledispatch
def ir_type_to_mlir_type(ir_type: Any) -> mlir.Type:
    raise NotImplementedError(f"Unable to convert {ir_type=} to MLIR type")


@ir_type_to_mlir_type.register
def tensor_map_type_to_mlir_type(src_type: ir_type.TensorMapTy) -> mlir.Type:
    return mlir.llvm.LLVMPointerType()


@ir_type_to_mlir_type.register
def scalar_type_to_mlir_type(src_type: ir_type.ScalarTy) -> mlir.Type:
    return dtype_to_mlir_type(src_type.dtype)


@ir_type_to_mlir_type.register
def pointer_type_to_mlir_type(src_type: ir_type.PointerTy) -> mlir.Type:
    return dtype_to_mlir_type(src_type.pointer_dtype)


@ir_type_to_mlir_type.register
def vector_type_to_mlir_type(src_type: ir_type.VectorTy) -> mlir.Type:
    return mlir.VectorType(
        shape=(src_type.length,),
        elementType=dtype_to_mlir_type(src_type.element_dtype),
        scalableDims=(False,)
    )


def dtype_to_mlir_type(dtype: datatype.DType) -> mlir.Type:
    if is_pointer_dtype(dtype):
        info = PointerInfo(dtype)
        return mlir.llvm.LLVMPointerType(addressSpace=int(info.memory_space.value))

    match dtype:
        case (
            datatype.int8
            | datatype.int16
            | datatype.int32
            | datatype.int64
            | datatype.bool_
            | datatype.uint8
            | datatype.uint16
            | datatype.uint32
            | datatype.uint64
        ):
            return mlir.IntegerType(
                width=dtype.bitwidth, signedness=mlir.SignednessSemantics.SIGNLESS
            )
        case datatype.float16:
            return mlir.Float16Type()
        case datatype.bfloat16:
            return mlir.BFloat16Type()
        case datatype.float32:
            return mlir.Float32Type()
        case datatype.float64:
            return mlir.Float64Type()
        case datatype.mbarrier:
            return mlir.IntegerType(
                width=64, signedness=mlir.SignednessSemantics.SIGNLESS
            )
        case datatype.clusterlaunchcontrol_token:
            return mlir.IntegerType(
                width=128, signedness=mlir.SignednessSemantics.SIGNLESS
            )
        case _:
            raise NotImplementedError(f"Unable to convert {dtype=} to MLIR type")


@singledispatch
def mlir_constant_of_type(mlir_type: mlir.Type, value) -> mlir.Value:
    raise NotImplementedError(f"Unable to convert {value=} to MLIR type {mlir_type=}")


@mlir_constant_of_type.register
def scalar_to_vector_constant(mlir_type: mlir.VectorType, value) -> mlir.Value:
    if any(mlir_type.scalableDims):
        raise NotImplementedError('Scalable vectors')
    if isinstance(mlir_type.elementType, mlir.FloatType):
        value = float(value)
    elif isinstance(mlir_type.elementType, mlir.IntegerType | mlir.IndexType):
        value = int(value)
    else:
        raise NotImplementedError(
            f"MLIR vector constant of element type {mlir_type.elementType}"
        )
    value_attr = mlir.DenseTypedElementsAttr(type=mlir_type, rawData=value)
    res = mlir.llvm.add_ConstantOp(
        res_type=mlir_type,
        value=value_attr
    )
    return res


@mlir_constant_of_type.register
def float_to_mlir_constant(mlir_type: mlir.FloatType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.FloatAttr(type=mlir_type, value=mlir.APFloat(value)),
    )


@mlir_constant_of_type.register
def int_to_mlir_constant(mlir_type: mlir.IntegerType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.IntegerAttr(
            type=mlir_type, value=mlir.APInt(int(value), mlir_type.width)
        ),
    )


@mlir_constant_of_type.register
def int_to_mlir_index(mlir_type: mlir.IndexType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.IntegerAttr(
            type=mlir_type, value=mlir.APInt(int(value), mlir_type.width)
        ),
    )


def _get_type_conversion_encoder(
    from_type: ir_type.TensorLikeTy,
    to_type: ir_type.TensorLikeTy,
):
    from_dtype = from_type.tensor_dtype()
    to_dtype = to_type.tensor_dtype()

    if from_type.tensor_shape() != to_type.tensor_shape():
        raise InternalError(
            f"Cannot convert between different shapes: {from_type} and {to_type}"
        )

    if from_dtype == to_dtype:
        return lambda x: x

    to_mlir_type = ir_type_to_mlir_type(to_type)
    from_mlir_type = ir_type_to_mlir_type(from_type)

    if datatype.is_boolean(to_dtype):
        def to_bool(in_):
            if datatype.is_float(from_dtype):
                c0 = mlir_constant_of_type(from_mlir_type, 0.0)
                result = mlir.arith.add_CmpFOp(
                    predicate=mlir.arith.CmpFPredicate.UNE,
                    lhs=in_,
                    rhs=c0,
                )
            else:
                c0 = mlir_constant_of_type(from_mlir_type, 0)
                result = mlir.arith.add_CmpIOp(
                    predicate=mlir.arith.CmpIPredicate.ne,
                    lhs=in_,
                    rhs=c0,
                )

            return mlir.arith.add_ExtUIOp(
                out_type=to_mlir_type,
                in_=result,
            )
        return to_bool

    def kind(t):
        if datatype.is_float(t):
            return "f"
        if datatype.is_integral(t) or datatype.is_boolean(t):
            return "si" if datatype.is_signed(t) else "ui"
        raise InternalError(f"Unsupported dtype: {t}")

    from_kind, to_kind = kind(from_dtype), kind(to_dtype)
    lhs_width = from_dtype.bitwidth
    rhs_width = to_dtype.bitwidth

    # TODO: rounding modes
    match from_kind, to_kind:
        case "f", "f":
            if lhs_width < rhs_width:
                return partial(mlir.arith.add_ExtFOp, out_type=to_mlir_type)
            else:
                return partial(mlir.arith.add_TruncFOp, out_type=to_mlir_type)
        case "f", "si":
            return partial(mlir.arith.add_FPToSIOp, out_type=to_mlir_type)
        case "f", "ui":
            return partial(mlir.arith.add_FPToUIOp, out_type=to_mlir_type)
        case "si", "f":
            return partial(mlir.arith.add_SIToFPOp, out_type=to_mlir_type)
        case "ui", "f":
            return partial(mlir.arith.add_UIToFPOp, out_type=to_mlir_type)

    if from_dtype.bitwidth < to_dtype.bitwidth:
        assert from_kind in ("si", "ui")
        if datatype.is_signed(from_dtype):
            return partial(mlir.arith.add_ExtSIOp, out_type=to_mlir_type)
        else:
            return partial(mlir.arith.add_ExtUIOp, out_type=to_mlir_type)
    elif from_dtype.bitwidth > to_dtype.bitwidth:
        return partial(mlir.arith.add_TruncIOp, out_type=to_mlir_type)
    elif from_kind in ("si", "ui") and to_kind in ("si", "ui"):
        return lambda in_: in_
    raise NotImplementedError(
        f"Conversion from {from_dtype} to {to_dtype} not implemented"
    )


def convert_dtype(
    src_type: ir_type.TensorLikeTy,
    dst_type: ir_type.TensorLikeTy,
    value: mlir.Value,
) -> mlir.Value:
    encoder = _get_type_conversion_encoder(src_type, dst_type)
    return encoder(in_=value)


def mlir_integer_cast(
    int_value: mlir.Value, to_type: mlir.IntegerType, signed: bool
) -> mlir.Value:
    if int_value.type == to_type:
        return int_value

    if isinstance(int_value.type, mlir.IndexType) or isinstance(
        to_type, mlir.IndexType
    ):
        return mlir.arith.add_IndexCastOp(out_type=to_type, in_=int_value)

    src_bw = int_value.type.width
    dst_bw = to_type.width

    if src_bw < dst_bw:
        if signed:
            return mlir.arith.add_ExtSIOp(out_type=to_type, in_=int_value)
        else:
            return mlir.arith.add_ExtUIOp(out_type=to_type, in_=int_value)
    else:
        return mlir.arith.add_TruncIOp(out_type=to_type, in_=int_value)


__all__ = (
    "ir_type_to_mlir_type",
    "mlir_constant_of_type",
    "mlir_integer_cast",
    "convert_dtype",
)
