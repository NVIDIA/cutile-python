# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.tile import TileTypeError, TileValueError
from cuda.tile._datatype import DType, is_pointer_dtype, PointerInfo, opaque_pointer_dtype, \
    pointer_dtype, is_numeric, numeric_dtype_category, NumericDTypeCategory, IntegerInfo, \
    _DTypePromotionImpl
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.ir import Operation, Var, operand, add_operation
from cuda.tile._ir.type import TensorLikeTy, LooselyTypedScalar
from cuda.tile._ir.typing_support import dtype_of_constant_scalar
from cuda.tile._memory_model import MemorySpace


@dataclass(eq=False)
class ReinterpretPointer(Operation, opcode="reinterpret_pointer"):
    pointer: Var[TensorLikeTy] = operand()


def reinterpret_pointer(pointer: Var[TensorLikeTy], target_ptr_dtype: DType) -> Var:
    src_ty = pointer.get_type()
    src_dtype = src_ty.tensor_dtype()
    assert is_pointer_dtype(src_dtype)
    assert is_pointer_dtype(target_ptr_dtype)

    if src_dtype == target_ptr_dtype:
        return pointer

    src_space = PointerInfo(src_dtype).memory_space
    target_space = PointerInfo(target_ptr_dtype).memory_space
    if src_space != target_space:
        raise TileTypeError(f"Source memory space '{src_space._name_}'"
                            f" does not match target memory space '{target_space}'")

    return add_operation(ReinterpretPointer,
                         pointer.ctx.typing_hooks.get_tensor_like_type(
                                target_ptr_dtype, src_ty.tensor_shape()),
                         pointer=pointer)


@dataclass(eq=False)
class AddrSpaceCast(Operation, opcode="address_space_cast"):
    pointer: Var = operand()


def address_space_cast(value: Var[TensorLikeTy], memory_space: MemorySpace) -> Var:
    src_ty = value.get_type()
    src_dtype = src_ty.tensor_dtype()
    assert is_pointer_dtype(src_dtype)

    info = PointerInfo(src_dtype)
    if info.memory_space == memory_space:
        return value

    if info.opaque:
        new_dtype = opaque_pointer_dtype(memory_space)
    else:
        new_dtype = pointer_dtype(info.pointee_dtype, memory_space)
    result_ty = value.ctx.typing_hooks.get_tensor_like_type(new_dtype, src_ty.tensor_shape())
    return add_operation(AddrSpaceCast, result_ty, pointer=value)


def check_implicit_cast(src_ty: TensorLikeTy | LooselyTypedScalar, target_dtype: DType):
    if isinstance(src_ty, LooselyTypedScalar):
        if not is_numeric(target_dtype):
            raise TileValueError(f"cannot implicitly cast {src_ty.value}"
                                 f" to a non-numeric dtype {target_dtype}")

        concrete_dtype = dtype_of_constant_scalar(src_ty.value)
        src_cat = numeric_dtype_category(concrete_dtype)
        dst_cat = numeric_dtype_category(target_dtype)
        if dst_cat == NumericDTypeCategory.Boolean:
            if src_cat not in (NumericDTypeCategory.Boolean, NumericDTypeCategory.Integral) \
                    or src_ty.value not in (0, 1):
                raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat > dst_cat:
            raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat == dst_cat == NumericDTypeCategory.Integral:
            info = IntegerInfo(target_dtype)
            if not (info.min <= src_ty.value <= info.max):
                raise TileValueError(f"{src_ty.value} is out of range of {target_dtype}")
    else:
        assert isinstance(src_ty, TensorLikeTy)
        if not _is_implicit_cast_ok(src_ty.tensor_dtype(), target_dtype):
            raise TileTypeError(f"cannot implicitly cast {src_ty.tensor_dtype()} to {target_dtype}")


def implicit_cast(src: Var[TensorLikeTy], target_dtype: DType, error_context: str) -> Var:
    ty = src.get_loose_type()
    try:
        check_implicit_cast(ty, target_dtype)
    except TileTypeError as e:
        raise TileTypeError(f"{error_context}: {str(e)}")
    except TileValueError as e:
        raise TileValueError(f"{error_context}: {str(e)}")

    if is_pointer_dtype(target_dtype):
        p = address_space_cast(src, PointerInfo(target_dtype).memory_space)
        return reinterpret_pointer(p, target_dtype)

    return astype(src, target_dtype)


def _is_implicit_cast_ok(src_dtype: DType, target_dtype: DType) -> bool:
    if is_numeric(src_dtype) and is_numeric(target_dtype):
        try:
            common_dtype = _DTypePromotionImpl.promote_dtypes(src_dtype, target_dtype)
        except TileTypeError:
            return False
        return common_dtype == target_dtype
    elif is_pointer_dtype(src_dtype) and is_pointer_dtype(target_dtype):
        src_info = PointerInfo(src_dtype)
        target_info = PointerInfo(target_dtype)
        if (src_info.memory_space != target_info.memory_space
                and target_info.memory_space != MemorySpace.GENERIC):
            # Only allow implicit cast to the GENERIC memory space
            return False

        if src_info.opaque:
            # Disallow implicit opaque to concrete cast
            if not target_info.opaque:
                return False
        elif not target_info.opaque and src_info.pointee_dtype != target_info.pointee_dtype:
            # Disallow changing the pointee_dtype
            return False

        return True
    else:
        return False
