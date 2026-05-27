# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import itertools
import math

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Sequence, Literal
from enum import Enum

from cuda.tile import _datatype as datatype

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._numeric_semantics import RoundingMode, PaddingMode
from cuda.tile._exception import Loc, TileTypeError, TileValueError, TileUnsupportedFeatureError
from cuda.tile._memory_model import MemoryOrder, MemoryScope, MemorySpace
import cuda.tile._bytecode as bc

from .ir import Operation, Builder
from .type import TileTy, LooselyTypedScalar
from .typing_support import dtype_of_constant_scalar
from .._datatype import DType, _DTypePromotionImpl, NumericDTypeCategory, PointerInfo


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
    supported_rounding_modes: Dict[RoundingMode, Optional[BytecodeVersion]] = field(
        default_factory=dict)
    support_flush_to_zero: bool = False


_RD_BASIC = {RoundingMode.RN: None, RoundingMode.RZ: None,
             RoundingMode.RM: None, RoundingMode.RP: None}
_RD_TRUEDIV = {**_RD_BASIC, RoundingMode.FULL: None, RoundingMode.APPROX: None}
_RD_SQRT = {**_RD_BASIC, RoundingMode.APPROX: None}
_RD_TANH = {RoundingMode.FULL: None, RoundingMode.APPROX: BytecodeVersion.V_13_2}
_RD_EXP = {RoundingMode.FULL: None, RoundingMode.APPROX: BytecodeVersion.V_13_3}

BINOP_REGISTRY = {
    "add": MathOpDef(lambda x, y: x + y, _RD_BASIC, support_flush_to_zero=True),
    "sub": MathOpDef(lambda x, y: x - y, _RD_BASIC, support_flush_to_zero=True),
    "mul": MathOpDef(lambda x, y: x * y, _RD_BASIC, support_flush_to_zero=True),
    "floordiv": MathOpDef(lambda x, y: x // y),
    "cdiv": MathOpDef(lambda x, y: (x + y - 1) // y),
    "truediv": MathOpDef(lambda x, y: x / y, _RD_TRUEDIV, support_flush_to_zero=True),
    "mod": MathOpDef(lambda x, y: x % y),
    "pow": MathOpDef(lambda x, y: x ** y),
    "atan2": MathOpDef(math.atan2),
    "max": MathOpDef(max, support_flush_to_zero=True),
    "min": MathOpDef(min, support_flush_to_zero=True),
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


def _invert(x: int | bool, bool_action: Literal['raise'] | Literal['not']):
    if isinstance(x, bool):
        if bool_action == 'not':
            return not x
        else:
            assert bool_action == 'raise'
            raise TileTypeError(
                '`~` on boolean constant is not supported, please use ct.bitwise_not')
    return ~x


UNARYOP_REGISTRY = {
    "abs": MathOpDef(abs),
    "neg": MathOpDef(lambda x: -x),
    "exp": MathOpDef(math.exp, _RD_EXP),
    "exp2": MathOpDef(lambda x: 2 ** x, support_flush_to_zero=True),
    "sin": MathOpDef(math.sin),
    "sinh": MathOpDef(math.sinh),
    "cos": MathOpDef(math.cos),
    "cosh": MathOpDef(math.cosh),
    "tan": MathOpDef(math.tan),
    "tanh": MathOpDef(math.tanh, _RD_TANH),
    "log": MathOpDef(math.log),
    "log2": MathOpDef(math.log2),
    "sqrt": MathOpDef(math.sqrt, _RD_SQRT, support_flush_to_zero=True),
    "rsqrt": MathOpDef(lambda x: x ** -0.5, support_flush_to_zero=True),
    "invert": MathOpDef(lambda x: _invert(x, bool_action='raise')),
    "bitwise_not": MathOpDef(lambda x: _invert(x, bool_action='not')),
    "not_": MathOpDef(lambda x: not x),
    "floor": MathOpDef(math.floor),
    "ceil": MathOpDef(math.ceil),
    "isnan": MathOpDef(math.isnan)
}


def get_default_rounding_mode(opname: Optional[str] = None):
    return RoundingMode.FULL if opname in ('tanh', 'exp') else RoundingMode.RN


rounding_mode_to_bytecode = {
    RoundingMode.RN: bc.RoundingMode.NEAREST_EVEN,
    RoundingMode.RZ: bc.RoundingMode.ZERO,
    RoundingMode.RM: bc.RoundingMode.NEGATIVE_INF,
    RoundingMode.RP: bc.RoundingMode.POSITIVE_INF,
    RoundingMode.FULL: bc.RoundingMode.FULL,
    RoundingMode.APPROX: bc.RoundingMode.APPROX,
    RoundingMode.RZI: bc.RoundingMode.NEAREST_INT_TO_ZERO
}


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
        min_version = math_op_def.supported_rounding_modes[rounding_mode]
        if min_version is not None:
            cur_version = Builder.get_current().ir_ctx.tileiras_version
            if cur_version < min_version:
                raise TileUnsupportedFeatureError(
                    f'{fn} rounding_mode={rounding_mode.value} requires tileiras '
                    f'{min_version.as_string()} or later. '
                    f'Current version is {cur_version.as_string()}.')
        if not datatype.is_unrestricted_float(dtype):
            raise TileTypeError(
                f'Rounding mode can only be used for unrestricted float types, '
                f'but got {dtype}')
        if rounding_mode in [RoundingMode.APPROX, RoundingMode.FULL]:
            if dtype != datatype.float32:
                raise TileTypeError(
                    f'Rounding mode {rounding_mode.value} can only be used for float32 type, '
                    f'but got {dtype}')
    if flush_to_zero:
        if not math_op_def.support_flush_to_zero:
            raise TileTypeError(f'Flush to zero is not supported for {fn}')
        if dtype != datatype.float32:
            raise TileTypeError(
                f'Flush to zero can only be used for float32 type, '
                f'but got {dtype}')


memory_scope_to_bytecode = {
    MemoryScope.NONE: None,
    MemoryScope.BLOCK: bc.MemoryScope.TL_BLK,
    MemoryScope.DEVICE: bc.MemoryScope.DEVICE,
    MemoryScope.SYS: bc.MemoryScope.SYS
}


memory_order_to_bytecode = {
    MemoryOrder.WEAK: bc.MemoryOrderingSemantics.WEAK,
    MemoryOrder.RELAXED: bc.MemoryOrderingSemantics.RELAXED,
    MemoryOrder.ACQUIRE: bc.MemoryOrderingSemantics.ACQUIRE,
    MemoryOrder.RELEASE: bc.MemoryOrderingSemantics.RELEASE,
    MemoryOrder.ACQ_REL: bc.MemoryOrderingSemantics.ACQ_REL,
}


def memory_order_has_acquire(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.ACQUIRE, MemoryOrder.ACQ_REL)


def memory_order_has_release(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.RELEASE, MemoryOrder.ACQ_REL)


def get_dtype(ty: TileTy | LooselyTypedScalar) -> datatype.DType:
    if isinstance(ty, LooselyTypedScalar):
        return dtype_of_constant_scalar(ty.value)
    assert isinstance(ty, TileTy)
    return ty.dtype


def change_dtype(ty: TileTy, new_dtype: datatype.DType) -> TileTy:
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
    loose_dtype = dtype_of_constant_scalar(loose_const)

    cat = datatype.numeric_dtype_category(dtype)
    if cat == NumericDTypeCategory.RestrictedFloat:
        # Treat restricted floats as regular floats.
        cat = NumericDTypeCategory.Float
    loose_cat = datatype.numeric_dtype_category(loose_dtype)

    if loose_cat == cat:
        # Both values are of the same dtype category. Use the concrete dtype in this case.
        ret = dtype

        # For integers, verify that the loosely typed constant is within the range of dtype.
        if cat == NumericDTypeCategory.Integral and not force_float:
            info = datatype.IntegerInfo(dtype)
            if not (info.min <= loose_const <= info.max):
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
            dtype1 = dtype_of_constant_scalar(val1)
            dtype2 = dtype_of_constant_scalar(val2)
            return _DTypePromotionImpl.promote_dtypes(dtype1, dtype2, force_float)
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
    shape = broadcast_shapes2(t1.shape, t2.shape)
    return TileTy(dtype, shape)


def _is_implicit_cast_ok(src_dtype: DType, target_dtype: DType) -> bool:
    if datatype.is_numeric(src_dtype) and datatype.is_numeric(target_dtype):
        try:
            common_dtype = _DTypePromotionImpl.promote_dtypes(src_dtype, target_dtype)
        except TileTypeError:
            return False
        return common_dtype == target_dtype
    elif datatype.is_pointer_dtype(src_dtype) and datatype.is_pointer_dtype(target_dtype):
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


def check_implicit_cast(src_ty: TileTy | LooselyTypedScalar, target_dtype: DType):
    if isinstance(src_ty, LooselyTypedScalar):
        if not datatype.is_numeric(target_dtype):
            raise TileValueError(f"cannot implicitly cast {src_ty.value}"
                                 f" to a non-numeric dtype {target_dtype}")

        concrete_dtype = dtype_of_constant_scalar(src_ty.value)
        src_cat = datatype.numeric_dtype_category(concrete_dtype)
        dst_cat = datatype.numeric_dtype_category(target_dtype)
        if dst_cat == NumericDTypeCategory.Boolean:
            if src_cat not in (NumericDTypeCategory.Boolean, NumericDTypeCategory.Integral) \
                    or src_ty.value not in (0, 1):
                raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat > dst_cat:
            raise TileTypeError(f"cannot implicitly cast {src_ty.value} to {target_dtype}")
        elif src_cat == dst_cat == NumericDTypeCategory.Integral:
            info = datatype.IntegerInfo(target_dtype)
            if not (info.min <= src_ty.value <= info.max):
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


def get_default_order(rank: int) -> tuple[int, ...]:
    return tuple(range(rank))


def validate_memory_order_and_scope(
    memory_order: MemoryOrder,
    memory_scope: MemoryScope,
    operation_type: type[Operation],
):
    opcode: str = operation_type._opcode
    if memory_order not in operation_type.VALID_MEMORY_ORDERS:
        formatted_expected = ", ".join(
            str(order) for order in operation_type.VALID_MEMORY_ORDERS
        )
        raise TileTypeError(
            f"Invalid memory order for {opcode}. "
            f"Got {memory_order}, expected one of {formatted_expected}"
        )

    if memory_scope not in operation_type.VALID_MEMORY_SCOPES:
        formatted_expected = ", ".join(
            str(scope) for scope in operation_type.VALID_MEMORY_SCOPES
        )
        raise TileTypeError(
            f"Invalid memory scope for {opcode}. "
            f"Got {memory_scope}, expected one of {formatted_expected}"
        )

    if memory_order == MemoryOrder.WEAK and memory_scope != MemoryScope.NONE:
        raise TileTypeError(
            f"{opcode} with WEAK memory ordering cannot specify a memory scope"
        )

    if memory_order != MemoryOrder.WEAK and memory_scope == MemoryScope.NONE:
        raise TileTypeError(
            f"{opcode} with {memory_order.name} memory ordering requires a memory scope"
        )
