# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import itertools
import math

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Sequence
from enum import Enum

from cuda.tile import _datatype as datatype

from cuda.tile._numeric_semantics import RoundingMode, PaddingMode
from cuda.tile._exception import Loc, TileTypeError, TileValueError
from cuda.tile._memory_model import MemoryOrder, MemoryScope
import cuda.tile._bytecode as bc

from .ir import Operation
from .type import TileTy, PointerTy, LooselyTypedScalar, make_tile_ty
from .typing_support import typeof_pyval
from .._datatype import DType, _DTypePromotionImpl, NumericDTypeCategory, NumericDTypeCategories, \
    get_int_min_max


class ComparisonPredicates(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


@dataclass
class MathOpDef:
    impl: callable    # Python scalar fallback
    supported_rounding_modes: Tuple[RoundingMode, ...] = ()
    support_flush_to_zero: bool = False


BINOP_REGISTRY = {
    "add": MathOpDef(lambda x, y: x + y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "sub": MathOpDef(lambda x, y: x - y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "mul": MathOpDef(lambda x, y: x * y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "floordiv": MathOpDef(lambda x, y: x // y),
    "cdiv": MathOpDef(lambda x, y: (x + y - 1) // y),
    "truediv": MathOpDef(lambda x, y: x / y,
                         (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP,
                          RoundingMode.FULL, RoundingMode.APPROX),
                         support_flush_to_zero=True),
    "mod": MathOpDef(lambda x, y: x % y),
    "pow": MathOpDef(lambda x, y: x ** y),
    "max": MathOpDef(max, (), support_flush_to_zero=True),
    "min": MathOpDef(min, (), support_flush_to_zero=True),
    "and_": MathOpDef(lambda x, y: x & y),
    "or_": MathOpDef(lambda x, y: x | y),
    "xor": MathOpDef(lambda x, y: x ^ y),
    "eq": MathOpDef(lambda x, y: x == y),
    "ne": MathOpDef(lambda x, y: x != y),
    "ge": MathOpDef(lambda x, y: x >= y),
    "gt": MathOpDef(lambda x, y: x > y),
    "le": MathOpDef(lambda x, y: x <= y),
    "lt": MathOpDef(lambda x, y: x < y),
    "is": MathOpDef(lambda x, y: x is y),
    "lshift": MathOpDef(lambda x, y: x << y),
    "rshift": MathOpDef(lambda x, y: x >> y),
}

for name in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow',
             'and_', 'or_', 'xor']:
    BINOP_REGISTRY["i" + name] = BINOP_REGISTRY[name]


UNARYOP_REGISTRY = {
    "abs": MathOpDef(abs),
    "neg": MathOpDef(lambda x: -x),
    "exp": MathOpDef(math.exp),
    "exp2": MathOpDef(lambda x: 2 ** x, (), support_flush_to_zero=True),
    "sin": MathOpDef(math.sin),
    "sinh": MathOpDef(math.sinh),
    "cos": MathOpDef(math.cos),
    "cosh": MathOpDef(math.cosh),
    "tan": MathOpDef(math.tan),
    # TODO: RoundingMode support dependent on bytecode version
    "tanh": MathOpDef(math.tanh),
    "log": MathOpDef(math.log),
    "log2": MathOpDef(math.log2),
    "sqrt": MathOpDef(math.sqrt,
                      (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP,
                       RoundingMode.APPROX),
                      support_flush_to_zero=True),
    "rsqrt": MathOpDef(lambda x: x ** -0.5, (), support_flush_to_zero=True),
    "invert": MathOpDef(lambda x: ~x),
    "not_": MathOpDef(lambda x: not x),
    "floor": MathOpDef(math.floor),
    "ceil": MathOpDef(math.ceil),
}


def get_default_rounding_mode():
    return RoundingMode.RN


rounding_mode_to_bytecode = {
    RoundingMode.RN: bc.RoundingMode.NEAREST_EVEN,
    RoundingMode.RZ: bc.RoundingMode.ZERO,
    RoundingMode.RM: bc.RoundingMode.NEGATIVE_INF,
    RoundingMode.RP: bc.RoundingMode.POSITIVE_INF,
    RoundingMode.FULL: bc.RoundingMode.FULL,
    RoundingMode.APPROX: bc.RoundingMode.APPROX,
    RoundingMode.RZI: bc.RoundingMode.NEAREST_INT_TO_ZERO
}

rounding_mode_to_bytecode[None] = rounding_mode_to_bytecode[get_default_rounding_mode()]


def get_rounding_mode(op: Operation, constants: Dict[str, Any]) -> Optional[RoundingMode]:
    return (
        constants[op.rounding_mode.name]
        if "rounding_mode" in op.operands
        else None
    )


def get_flush_to_zero(op: Operation, constants: Dict[str, Any]) -> bool:
    return (
        constants[op.flush_to_zero.name]
        if "flush_to_zero" in op.operands
        else False
    )


def check_rd_and_ftz(fn: str, rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                     dtype: datatype.DType):
    if rounding_mode is None and flush_to_zero is False:
        return

    math_op_def = BINOP_REGISTRY[fn] if fn in BINOP_REGISTRY else UNARYOP_REGISTRY[fn]
    if rounding_mode is not None:
        if rounding_mode not in math_op_def.supported_rounding_modes:
            raise TileTypeError(
                f'Rounding mode {rounding_mode.value} is not supported for {fn}')
        if not datatype.is_float(dtype):
            raise TileTypeError(
                f'Rounding mode can only be used for float types, '
                f'but got {dtype}')
        if rounding_mode in [RoundingMode.APPROX, RoundingMode.FULL]:
            if dtype != datatype.float32:
                raise TileTypeError(
                    f'Rounding mode {rounding_mode.value} can only be used for float32 type, '
                    f'but got {dtype}')
    if flush_to_zero:
        if flush_to_zero and not math_op_def.support_flush_to_zero:
            raise TileTypeError(f'Flush to zero is not supported for {fn}')
        if dtype != datatype.float32:
            raise TileTypeError(
                f'Flush to zero can only be used for float32 type, '
                f'but got {dtype}')


memory_scope_to_bytecode = {
    MemoryScope.BLOCK: bc.MemoryScope.TL_BLK,
    MemoryScope.DEVICE: bc.MemoryScope.DEVICE,
    MemoryScope.SYS: bc.MemoryScope.SYS
}


memory_order_to_bytecode = {
    MemoryOrder.RELAXED: bc.MemoryOrderingSemantics.RELAXED,
    MemoryOrder.ACQUIRE: bc.MemoryOrderingSemantics.ACQUIRE,
    MemoryOrder.RELEASE: bc.MemoryOrderingSemantics.RELEASE,
    MemoryOrder.ACQ_REL: bc.MemoryOrderingSemantics.ACQ_REL,
}


def memory_order_has_acquire(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.ACQUIRE, MemoryOrder.ACQ_REL)


def memory_order_has_release(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.RELEASE, MemoryOrder.ACQ_REL)


def get_dtype(ty: TileTy | LooselyTypedScalar) -> datatype.DType | PointerTy:
    if isinstance(ty, LooselyTypedScalar):
        ty = typeof_pyval(ty.value)
    assert isinstance(ty, TileTy)
    return ty.dtype


def change_dtype(ty: TileTy, new_dtype: datatype.DType | PointerTy) \
        -> TileTy:
    assert isinstance(ty, TileTy)
    return TileTy(new_dtype, ty.shape)


def check_shapes_eq(a: TileTy, b: TileTy,
                    a_name: str, b_name: str, loc: Loc) -> None:
    if a.shape != b.shape:
        raise TileTypeError(f"{a_name} and {b_name} shapes must match, "
                            f"got {a.shape} and {b.shape}", loc)


class CompareOrdering(Enum):
    ORDERED = "ordered"
    UNORDERED = "unordered"


padding_mode_to_bytecode = {
    PaddingMode.UNDETERMINED: bc.PaddingValue.Missing,
    PaddingMode.ZERO: bc.PaddingValue.Zero,
    PaddingMode.NEG_ZERO: bc.PaddingValue.NegZero,
    PaddingMode.NAN: bc.PaddingValue.Nan,
    PaddingMode.POS_INF: bc.PaddingValue.PosInf,
    PaddingMode.NEG_INF: bc.PaddingValue.NegInf,
}


def _promote_dtype_and_loosely_typed_constant(dtype: DType,
                                              loose_const: Any,
                                              force_float: bool) -> DType:
    loose_ty = typeof_pyval(loose_const)
    assert isinstance(loose_ty, TileTy) and loose_ty.ndim == 0
    loose_dtype = loose_ty.dtype

    cat = NumericDTypeCategories.get_category(dtype)
    if cat == NumericDTypeCategory.RestrictedFloat:
        # Treat restricted floats as regular floats.
        cat = NumericDTypeCategory.Float
    loose_cat = NumericDTypeCategories.get_category(loose_dtype)

    if loose_cat == cat:
        # Both values are of the same dtype category. Use the concrete dtype in this case.
        ret = dtype

        # For integers, verify that the loosely typed constant is within the range of dtype.
        if cat == NumericDTypeCategory.Integral and not force_float:
            min, max = get_int_min_max(dtype)
            if not (min <= loose_const <= max):
                raise TileValueError(f"Integer constant {loose_const} is out of range of {dtype}")
    else:
        # Strongest category always wins
        ret = loose_dtype if loose_cat > cat else dtype

    return ret if not force_float or datatype.is_float(ret) else datatype.default_float_type


def promote_dtypes(t1: DType | LooselyTypedScalar,
                   t2: DType | LooselyTypedScalar,
                   force_float: bool = False) -> DType:
    match t1, t2:
        case LooselyTypedScalar(val1), LooselyTypedScalar(val2):
            type1 = typeof_pyval(val1)
            assert isinstance(type1, TileTy)
            type2 = typeof_pyval(val2)
            assert isinstance(type2, TileTy)
            return _DTypePromotionImpl.promote_dtypes(type1.dtype, type2.dtype, force_float)
        case LooselyTypedScalar(val), dtype:
            return _promote_dtype_and_loosely_typed_constant(dtype, val, force_float)
        case dtype, LooselyTypedScalar(val):
            return _promote_dtype_and_loosely_typed_constant(dtype, val, force_float)
        case dtype1, dtype2:
            return _DTypePromotionImpl.promote_dtypes(dtype1, dtype2, force_float)


def promote_types(t1: TileTy | LooselyTypedScalar,
                  t2: TileTy | LooselyTypedScalar,
                  force_float: bool = False) -> TileTy:
    dtype_1 = t1 if isinstance(t1, LooselyTypedScalar) else t1.dtype
    dtype_2 = t2 if isinstance(t2, LooselyTypedScalar) else t2.dtype
    dtype = promote_dtypes(dtype_1, dtype_2, force_float)
    shape = broadcast_shapes2(t1.shape_value, t2.shape_value)
    return make_tile_ty(dtype, shape)


def _is_implicit_cast_ok(src_dtype: DType, target_dtype: DType) -> bool:
    try:
        common_dtype = _DTypePromotionImpl.promote_dtypes(src_dtype, target_dtype)
    except TileTypeError:
        return False
    return common_dtype == target_dtype


def check_implicit_cast(src_ty: TileTy | LooselyTypedScalar, target_dtype: DType):
    if isinstance(src_ty, LooselyTypedScalar):
        cocnrete_ty = typeof_pyval(src_ty.value)
        src_cat = NumericDTypeCategories.get_category(cocnrete_ty.dtype)
        dst_cat = NumericDTypeCategories.get_category(target_dtype)
        if dst_cat == NumericDTypeCategory.Boolean:
            if src_cat not in (NumericDTypeCategory.Boolean, NumericDTypeCategory.Integral) \
                    or src_ty.value not in (0, 1):
                raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat > dst_cat:
            raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat == dst_cat == NumericDTypeCategory.Integral:
            min, max = datatype.get_int_min_max(target_dtype)
            if not (min <= src_ty.value <= max):
                raise TileValueError(f"{src_ty.value} is out of range of {target_dtype}")
    else:
        assert isinstance(src_ty, TileTy)
        if not _is_implicit_cast_ok(src_ty.dtype, target_dtype):
            raise TileTypeError(f"cannot implicitly cast {src_ty.dtype} to {target_dtype}")


class BroadcastError(Exception):
    pass


# FIXME: rename to broadcast_shapes() after we remove broadcast_shapes()
def broadcast_shapes2(s1: Sequence[int], s2: Sequence[int]) -> Tuple[int, ...]:
    result_shape = []
    for d1, d2 in itertools.zip_longest(reversed(s1), reversed(s2), fillvalue=1):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise BroadcastError(f"Shapes are not broadcastable: {tuple(s1)}, {tuple(s2)}")
        result_shape.append(max(d1, d2))
    return tuple(reversed(result_shape))


def is_shape_broadcastable_to(src: Sequence[int], dst: Sequence[int]) -> bool:
    return len(src) <= len(dst) and all(x in (y, 1) for x, y in zip(reversed(src), reversed(dst)))
