# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import math
import operator
from dataclasses import dataclass
from typing import (
    Literal, Sequence, Tuple, Optional, Any, List, Callable, Iterable,
)

from typing_extensions import override

import cuda.tile._stub as ct
from cuda.tile import _datatype as datatype
from cuda.tile import RoundingMode, MemoryOrder, MemoryScope
from cuda.tile._exception import TileInternalError, TileTypeError, TileSyntaxError, \
    TileValueError, TileUnsupportedFeatureError
from cuda.tile._ir.ir import (
    Operation, Var, Loc, Block, add_operation, Builder, enter_nested_block, nested_block,
    make_aggregate, MemoryEffect, attribute, operand,
    BlockRestriction, add_operation_variadic,
)
from .aggregate_support import unflatten_aggregates
from .arithmetic_ops import reshape, broadcast_to, astype, compare_tensorlike, \
    binary_bitwise_tensorlike, bitwise_shift_tensorlike, binary_arithmetic_tensorlike, \
    compare_tensorlike_raw, where, binary_bitwise_tensorlike_raw, where_raw, TileReshape, \
    mod_tensorlike, promote_and_broadcast_to, arithmetic_impl_registry, \
    unary, UnaryBehavior, UNARY_INT_FLOAT, UNARY_ANYTHING, UNARY_BOOL_INT, \
    UNARY_STRICT_FLOAT, UNARY_FLOAT, divmod_tensorlike
from .cast_ops import implicit_cast
from .control_flow_ops import Loop, IfElse, control_flow_impl_registry, EndBranch
from .core_ops import loosely_typed_const, strictly_typed_const, build_tuple, bind_method, \
    sym2var, core_impl_registry, print_impl, TilePrintf, tuple_item
from .static_eval_ops import static_eval_impl_registry
from .type import (
    TupleValue, ArrayValue, ListValue, TiledViewValue, RawArrayMemoryValue,
    IndexSliceValue
)
from .op_impl import (
    ImplRegistry, is_scalar, require_constant_int, require_constant_int_tuple,
    require_signed_integer_0d_tile_type,
    require_tile_type, normalize_axis, require_dtype_spec,
    require_constant_bool, require_optional_constant_enum,
    require_constant_str, require_array_type, require_tiled_view_type, require_tuple_type,
    require_list_type, require_0d_tile_type,
    require_index_or_index_tuple_type, require_constant_shape, require_constant_axis_order,
    require_constant_enum, require_optional_constant_int, require_optional_constant_bool,
    require_optional_constant_str, PrintfValidator, require_tile_maybe_loose_type,
    require_tile_or_tile_tuple_type, require_constant_scalar_tuple, require_constant_scalar,
    require_callable_type, require_raw_array_memory_type,
    WILDCARD, ensure_tile)
from .ops_utils import (
    check_rd_and_ftz, PaddingMode, get_default_order,
    rounding_mode_to_bytecode, get_default_rounding_mode, get_dtype,
    memory_order_to_bytecode,
    memory_scope_to_bytecode, broadcast_shapes2, is_shape_broadcastable_to, BroadcastError,
    promote_dtypes, validate_memory_order_and_scope,
)
from .type import (
    PartitionViewTy, StridedViewTy, GatherScatterViewTy, TupleTy, TileTy, NoneType, ArrayTy,
    ListTy, Type, LooselyTypedScalar, TokenTy, TiledViewTy,
    RawArrayMemoryTy, IndexSliceTy,
)
from cuda.tile._datatype import (
    DType, is_integral, is_float, is_signed, is_boolean, PointerInfo,
)
from cuda.tile._ir2bytecode import (
    BytecodeContext, typeid,
    generate_bytecode_for_block, get_list_item_repr_size_in_words,
    get_list_partition_view_tile_size, tensor_view_typeid, tensor_view_typeid_for_list, dtype_typeid
)
import cuda.tile._bytecode as bc
from cuda.tile._bytecode.version import BytecodeVersion
from .._debug import CUDA_TILE_TESTING_DISABLE_DIV


tile_impl_registry = ImplRegistry()
tile_impl_registry.update(core_impl_registry())
tile_impl_registry.update(static_eval_impl_registry())
tile_impl_registry.update(arithmetic_impl_registry())
tile_impl_registry.update(control_flow_impl_registry())
impl = tile_impl_registry.impl

array_impl_registry = ImplRegistry()


@dataclass
class ReduceScanRestriction(BlockRestriction):
    """Restriction for reduction/scan body blocks: no memory effects, loops, or branching."""

    kind: Literal["reduction", "scan"]

    def validate_operation(self, op_class: type) -> None:
        if getattr(op_class, "memory_effect", MemoryEffect.NONE) != MemoryEffect.NONE:
            raise TileSyntaxError(
                f"Operations with memory effects are not supported inside {self.kind} body"
            )
        if op_class is Loop:
            raise TileSyntaxError(f"Loops inside {self.kind} body are not supported")
        if op_class is IfElse:
            raise TileSyntaxError(
                f"Branching inside {self.kind} body is not supported. "
                f"Consider ct.where() as a workaround."
            )


# Computes lhs*rhs + acc.  Also known as FMA.
@dataclass(eq=False)
class FusedMulAddOperation(Operation, opcode="fma"):
    rounding_mode: RoundingMode = attribute()
    flush_to_zero: bool = attribute()
    lhs: Var = operand()
    rhs: Var = operand()
    acc: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_type = ctx.typeof(self.result_var)
        lhs = ctx.cast(ctx.get_value(self.lhs), ctx.typeof(self.lhs), result_type)
        rhs = ctx.cast(ctx.get_value(self.rhs), ctx.typeof(self.rhs), result_type)
        acc = ctx.cast(ctx.get_value(self.acc), ctx.typeof(self.acc), result_type)
        rm = self.rounding_mode if self.rounding_mode is not None else get_default_rounding_mode()
        return bc.encode_FmaOp(ctx.builder,
                               ctx.typeid_of(self.result_var),
                               lhs, rhs, acc,
                               rounding_mode_to_bytecode[rm],
                               self.flush_to_zero)


@impl(ct.equal, fixed_args=["eq"])
@impl(ct.greater, fixed_args=["gt"])
@impl(ct.not_equal, fixed_args=["ne"])
@impl(ct.greater_equal, fixed_args=["ge"])
@impl(ct.less, fixed_args=["lt"])
@impl(ct.less_equal, fixed_args=["le"])
def tile_comparison_function_impl(fn: str, x: Var, y: Var):
    return compare_tensorlike(fn, ensure_tile(x), ensure_tile(y))


@impl(ct.bitwise_and, fixed_args=["and_"])
@impl(ct.bitwise_or, fixed_args=["or_"])
@impl(ct.bitwise_xor, fixed_args=["xor"])
def tile_binary_bitwise_function_impl(fn: str, x: Var, y: Var):
    return binary_bitwise_tensorlike(fn, ensure_tile(x), ensure_tile(y))


@impl(ct.bitwise_lshift, fixed_args=["lshift"])
@impl(ct.bitwise_rshift, fixed_args=["rshift"])
def tile_bitwise_shift_function_impl(fn: str, x: Var, y: Var):
    return bitwise_shift_tensorlike(fn, ensure_tile(x), ensure_tile(y))


@impl(ct.floordiv, fixed_args=["floordiv"])
@impl(ct.cdiv, fixed_args=["cdiv"])
@impl(ct.pow, fixed_args=["pow"])
def tile_binary_arithmetic_function_impl(fn: str, x: Var, y: Var) -> Var:
    return binary_arithmetic_tensorlike(fn, ensure_tile(x), ensure_tile(y))


@impl(ct.atan2, min_version=BytecodeVersion.V_13_2)
def atan2_impl(x1: Var, x2: Var) -> Var:
    return binary_arithmetic_tensorlike("atan2", ensure_tile(x1), ensure_tile(x2))


@impl(ct.minimum, fixed_args=["min"])
@impl(ct.maximum, fixed_args=["max"])
def tile_binary_arithmetic_function_impl_with_ftz(fn: str, x: Var, y: Var,
                                                  flush_to_zero: Var, propagate_nan: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    propagate_nan = require_constant_bool(propagate_nan)
    return binary_arithmetic_tensorlike(fn, ensure_tile(x), ensure_tile(y),
                                        flush_to_zero=flush_to_zero, propagate_nan=propagate_nan)


@impl(ct.add, fixed_args=["add"])
@impl(ct.sub, fixed_args=["sub"])
@impl(ct.mul, fixed_args=["mul"])
@impl(ct.truediv, fixed_args=["truediv"])
def binary_arithmetic_impl_with_rd_and_ftz(fn: str, x: Var, y: Var,
                                           rounding_mode: Var, flush_to_zero: Var) -> Var:
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return binary_arithmetic_tensorlike(fn, x, y, rounding_mode, flush_to_zero)


@impl(ct.mod)
def tile_mod_function_impl(x: Var, y: Var):
    return mod_tensorlike(ensure_tile(x), ensure_tile(y))


@impl(ct.divmod)
def tile_divmod_function_impl(x: Var, y: Var):
    return divmod_tensorlike(ensure_tile(x), ensure_tile(y))


@impl(slice)
def slice_impl(start: Var, stop: Var, step: Var) -> Var:
    if not (start.is_constant() and stop.is_constant() and step.is_constant()):
        raise TileTypeError("Non-constant slices are not supported")
    return loosely_typed_const(
        slice(start.get_constant(), stop.get_constant(), step.get_constant()))


# ===========================================================================================
# Tile getitem
# ===========================================================================================

def tile_expand_dims(x: Var, index: Tuple[Any, ...]) -> Var:
    x_type = x.get_type()

    for idx in index:
        if idx not in (None, Ellipsis, slice(None)):
            raise TileTypeError(
                f"Expected `None|np.newaxis` or `ellipsis` or full slice (`:`), "
                f"but got {idx}. Hint: Directly indexing a tile is not supported, "
                f"use `extract` or `item`.")

    num_slices = sum(1 for idx in index if isinstance(idx, slice))
    if num_slices > x_type.ndim:
        raise TileTypeError(f"Tile is {x_type.ndim}-dimensional, "
                            f"but {num_slices} were indexed")
    axes = []
    ellipsis_idx = None
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            if ellipsis_idx is not None:
                raise TileTypeError("Only one ellipsis is allowed")
            ellipsis_idx = i
        elif idx is None:
            axes.append(i - len(index) if ellipsis_idx is not None and i > ellipsis_idx else i)
    new_rank = x_type.ndim + len(axes)
    new_shape = list(x_type.shape)
    for axis in axes:
        normalized_axis = axis + new_rank if axis < 0 else axis
        new_shape.insert(normalized_axis, 1)
    return reshape(x, tuple(new_shape))


@impl(operator.getitem, overload=(TileTy, NoneType))
def getitem_tile_none_impl(object: Var, key: Var) -> Var:
    return tile_expand_dims(object, (None,))


@impl(operator.getitem, overload=(TileTy, TupleTy))
def getitem_tile_tuple_impl(object: Var, key: Var) -> Var:
    if not key.is_constant():
        raise TileTypeError("Tile subscript must be a constant tuple")
    return tile_expand_dims(object, key.get_constant())


@impl(operator.getitem, overload=(TileTy, WILDCARD))
def getitem_tile_fallback_impl(object: Var, key: Var) -> Var:
    raise TileTypeError("Directly indexing a tile is not supported; "
                        "use `extract()` or `item()` instead.")


# ===========================================================================================
# List getitem
# ===========================================================================================


@dataclass(eq=False)
class GetArrayListItem(Operation, opcode="get_array_list_item"):
    x: Var = operand()
    index: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        list_ty = ctx.typeof(self.x)
        assert isinstance(list_ty, ListTy)

        # First, load a (1 x item_tile_size) tile that represents the item
        partition_view = ctx.get_value(self.x)
        item_size = get_list_item_repr_size_in_words(list_ty.item_type)
        item_tile_size = get_list_partition_view_tile_size(item_size)
        pv_tile_type_id = ctx.type_table.tile(ctx.type_table.I64, (1, item_tile_size))
        index = ctx.get_value(self.index)
        index_i32 = ctx.cast(index, ctx.typeof(self.index), TileTy(datatype.int32))

        i32_ty = TileTy(datatype.int32)
        zero_i32 = ctx.constant(0, i32_ty)

        loaded_tile, _token = bc.encode_LoadViewTkoOp(
            ctx.builder,
            tile_type=pv_tile_type_id,
            result_token_type=ctx.type_table.Token,
            view=partition_view,
            index=(index_i32, zero_i32),
            token=None,
            memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
            memory_scope=None,
            optimization_hints=None,
            inbounds=(False, False)
        )

        item_typeid_tuple = tuple(typeid(ctx.type_table, ty)
                                  for ty in list_ty.item_type.flatten_aggregate())

        # Next, unpack the tile into individual values that represent the item
        assert isinstance(list_ty.item_type, ArrayTy)
        assert len(item_typeid_tuple) == item_size

        # Extract and reshape each element of the (1 x item_tile_size) tile
        # as a separate i64 scalar
        i64_scalar_ty = ctx.type_table.tile(ctx.type_table.I64, ())
        i64_1x1_ty = ctx.type_table.tile(ctx.type_table.I64, (1, 1))
        extracted_words = tuple(
            bc.encode_ReshapeOp(
                ctx.builder,
                i64_scalar_ty,
                bc.encode_ExtractOp(ctx.builder, i64_1x1_ty, loaded_tile,
                                    (zero_i32, ctx.constant(i, i32_ty)),),
            )
            for i in range(item_size)
        )

        # Cast each of the i64 words to appropriate types
        if list_ty.item_type.index_dtype.bitwidth >= 64:
            # Already i64, no truncation needed
            shape_stride_results = list(extracted_words[1:])
        else:
            shape_stride_results = [
                bc.encode_TruncIOp(ctx.builder, ty_id, w, bc.IntegerOverflow.NONE)
                for ty_id, w in zip(item_typeid_tuple[1:], extracted_words[1:], strict=True)
            ]

        return (
            # Cast the first word to data pointer
            bc.encode_IntToPtrOp(ctx.builder, item_typeid_tuple[0], extracted_words[0]),
            # Cast the remaining words to shape/stride types (i32 or i64)
            *shape_stride_results
        )


@impl(operator.getitem, overload=(ListTy, WILDCARD))
def getitem_list_impl(object: Var, key: Var) -> Var:
    list_ty = require_list_type(object)
    index_ty = require_0d_tile_type(key)
    index_dtype = get_dtype(index_ty)
    if not (isinstance(index_dtype, DType) and is_integral(index_dtype)):
        raise TileTypeError(f"Index must be an integer scalar or 0D Tile, got {index_ty}")
    item_ty = list_ty.item_type

    if not isinstance(item_ty, ArrayTy):
        raise TileTypeError(f"Indexing a list of {list_ty.item_type} is not implemented")

    flat_types = tuple(item_ty.flatten_aggregate())
    flat_results = add_operation_variadic(GetArrayListItem, flat_types, x=object, index=key)
    [ret] = unflatten_aggregates(flat_results, (item_ty,), (item_ty,))
    return ret


# ===========================================================================================
# Array getitem
# ===========================================================================================

@impl(operator.getitem, overload=(ArrayTy, WILDCARD))
def getitem_array_impl(object: Var, key: Var) -> Var:
    raise TileTypeError("Arrays are not directly subscriptable. Use load() or gather() instead.")


# ===========================================================================================


@impl(operator.setitem, overload=(ArrayTy, WILDCARD, WILDCARD))
def setitem_array_impl(object: Var, key: Var, value: Var):
    raise TileTypeError("Arrays do not support item assignment. Use store() or scatter() instead.")


@impl(operator.setitem, overload=(TileTy, WILDCARD, WILDCARD))
def setitem_tile_impl(object: Var, key: Var, value: Var):
    raise TileTypeError("Tiles are immutable: item assignment is not supported.")


@impl(len, overload=(ListTy,))
def len_list_impl(x: Var[ListTy]) -> Var:
    list_val = x.get_aggregate()
    assert isinstance(list_val, ListValue)
    return list_val.length


@impl(ct.log, fixed_args=["log", UNARY_FLOAT])
@impl(ct.log2, fixed_args=["log2", UNARY_FLOAT])
@impl(ct.tan, fixed_args=["tan", UNARY_FLOAT])
@impl(ct.sin, fixed_args=["sin", UNARY_FLOAT])
@impl(ct.sinh, fixed_args=["sinh", UNARY_FLOAT])
@impl(ct.cos, fixed_args=["cos", UNARY_FLOAT])
@impl(ct.cosh, fixed_args=["cosh", UNARY_FLOAT])
@impl(ct.bitwise_not, fixed_args=["bitwise_not", UNARY_BOOL_INT])
@impl(ct.floor, fixed_args=["floor", UNARY_STRICT_FLOAT])
@impl(ct.ceil, fixed_args=["ceil", UNARY_STRICT_FLOAT])
@impl(ct.negative, fixed_args=["neg", UNARY_INT_FLOAT])
@impl(ct.abs, fixed_args=["abs", UNARY_ANYTHING])
@impl(abs, fixed_args=["abs", UNARY_ANYTHING])
def unary_impl(fn: str, behavior: UnaryBehavior, x: Var) -> Var:
    return unary(fn, behavior, ensure_tile(x))


@impl(ct.rsqrt, fixed_args=["rsqrt", UNARY_FLOAT])
@impl(ct.exp2, fixed_args=["exp2", UNARY_FLOAT])
def unary_impl_with_ftz(fn: str, behavior: UnaryBehavior, x: Var, flush_to_zero: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, ensure_tile(x), flush_to_zero=flush_to_zero)


@impl(ct.sqrt, fixed_args=["sqrt", UNARY_FLOAT])
def unary_impl_with_rd_and_ftz(fn: str, behavior: UnaryBehavior,
                               x: Var, rounding_mode: Var, flush_to_zero: Var) -> Var:
    x = ensure_tile(x)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, x, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(ct.tanh, fixed_args=["tanh", UNARY_FLOAT])
@impl(ct.exp, fixed_args=["exp", UNARY_FLOAT])
def unary_impl_with_rd(fn: str, behavior: UnaryBehavior, x: Var, rounding_mode: Var) -> Var:
    x = ensure_tile(x)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    return unary(fn, behavior, x, rounding_mode=rounding_mode)


@impl(ct.isnan)
def isnan_impl(x: Var) -> Var:
    x_type = require_tile_maybe_loose_type(x)
    if isinstance(x_type, LooselyTypedScalar):
        res = math.isnan(x_type.value)
        return loosely_typed_const(res)

    ty = x.get_type()
    if isinstance(x_type, TileTy) and is_float(ty.dtype):
        if x.is_constant():
            res = math.isnan(x.get_constant())
            return strictly_typed_const(res, TileTy(datatype.bool_, ty.shape))
        else:
            return compare_tensorlike_raw("ne", x, x)
    raise TileTypeError(f"Unexpected input type {x_type}")


# ===========================================================================================
# Array attributes
# ===========================================================================================

@array_impl_registry.impl(getattr, overload=(ArrayTy, "dtype"))
def getattr_array_dtype_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().dtype)


@array_impl_registry.impl(getattr, overload=(ArrayTy, "ndim"))
def getattr_array_ndim_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().ndim)


@array_impl_registry.impl(getattr, overload=(ArrayTy, "shape"))
def getattr_array_shape_impl(object: Var, name: Var):
    return build_tuple(object.get_aggregate().shape)


@array_impl_registry.impl(getattr, overload=(ArrayTy, "strides"))
def getattr_array_strides_impl(object: Var, name: Var):
    return build_tuple(object.get_aggregate().strides)


@array_impl_registry.impl(getattr, overload=(ArrayTy, "slice"))
@array_impl_registry.impl(getattr, overload=(ArrayTy, "tiled_view"))
@array_impl_registry.impl(getattr, overload=(ArrayTy, "get_raw_memory"))
def getattr_array_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(ct.Array, name)
    return bind_method(object, unbound_func)


# ===========================================================================================
# Tile attributes
# ===========================================================================================

@impl(getattr, overload=(TileTy, "dtype"))
def getattr_tile_dtype_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().dtype)


@impl(getattr, overload=(TileTy, "shape"))
def getattr_tile_shape_impl(object: Var, name: Var):
    return sym2var(object.get_type().shape, constant_only=True)


@impl(getattr, overload=(TileTy, "ndim"))
def getattr_tile_ndim_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().ndim)


@impl(getattr, overload=(TileTy, "extract"))
@impl(getattr, overload=(TileTy, "insert"))
@impl(getattr, overload=(TileTy, "reshape"))
@impl(getattr, overload=(TileTy, "astype"))
@impl(getattr, overload=(TileTy, "permute"))
@impl(getattr, overload=(TileTy, "transpose"))
@impl(getattr, overload=(TileTy, "item"))
def getattr_tile_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(ct.Tile, name)
    return bind_method(object, unbound_func)


# ===========================================================================================
# TiledView attributes
# ===========================================================================================

@impl(getattr, overload=(TiledViewTy, "dtype"))
def getattr_tiled_view_dtype_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().dtype)


@impl(getattr, overload=(TiledViewTy, "tile_shape"))
def getattr_tiled_view_tile_shape_impl(object: Var, name: Var):
    return sym2var(object.get_type().tile_shape, constant_only=True)


@impl(getattr, overload=(TiledViewTy, "traversal_steps"))
def getattr_tiled_view_traversal_steps_impl(object: Var, name: Var):
    return sym2var(object.get_type().traversal_steps, constant_only=True)


@impl(getattr, overload=(TiledViewTy, "num_tiles"))
@impl(getattr, overload=(TiledViewTy, "load"))
@impl(getattr, overload=(TiledViewTy, "store"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_add"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_max"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_min"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_and"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_or"))
@impl(getattr, overload=(TiledViewTy, "atomic_store_xor"))
def getattr_tiled_view_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(ct.TiledView, name)
    return bind_method(object, unbound_func)


# ===========================================================================================
# RawArrayMemory attributes
# ===========================================================================================

@impl(getattr, overload=(RawArrayMemoryTy, "dtype"))
def getattr_raw_array_memory_dtype_impl(object: Var, name: Var):
    return loosely_typed_const(object.get_type().dtype)


@impl(getattr, overload=(RawArrayMemoryTy, "load_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "store_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_cas_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_xchg_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_add_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_max_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_min_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_and_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_or_offset"))
@impl(getattr, overload=(RawArrayMemoryTy, "atomic_xor_offset"))
def getattr_raw_array_memory_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(ct.RawArrayMemory, name)
    return bind_method(object, unbound_func)


# ================================================
# Tile specific operations
# ================================================

@dataclass(eq=False)
class TileBid(Operation, opcode="tile_bid"):
    axis: int = attribute()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        axis = self.axis
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_GetTileBlockIdOp(ctx.builder, res_typeid, res_typeid, res_typeid)[axis]


def bid(axis: int) -> Var:
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileBid, TileTy(datatype.default_int_type), axis=axis)


@impl(ct.bid)
def bid_impl(axis: Var) -> Var:
    axis = require_constant_int(axis)
    return bid(axis)


@dataclass(eq=False)
class MakeTensorView(Operation, opcode="make_tensor_view"):
    base_ptr: Var = operand()
    shape: tuple[Var, ...] = operand()
    dynamic_strides: tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        array_ty: ArrayTy = self.result_var.get_type()
        view_type_id = tensor_view_typeid(ctx.type_table, array_ty)
        base_ptr = ctx.get_value(self.base_ptr)
        shape = tuple(ctx.get_value(x) for x in self.shape)
        dynamic_strides = tuple(ctx.get_value(x) for x in self.dynamic_strides)
        return bc.encode_MakeTensorViewOp(ctx.builder,
                                          result_type=view_type_id,
                                          base=base_ptr,
                                          dynamicShape=shape,
                                          dynamicStrides=dynamic_strides)


@dataclass(eq=False)
class AssumeDivBy(Operation, opcode="assume_div_by"):
    divisor: int = attribute()
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.x)
        type_id = ctx.typeid_of(self.result_var)
        return bc.encode_AssumeOp(ctx.builder, type_id, x, bc.DivBy(self.divisor))


def assume_div_by(x: Var, divisor: int | None) -> Var:
    if divisor is None or divisor == 1 or CUDA_TILE_TESTING_DISABLE_DIV:
        return x
    if x.is_constant():
        val = x.get_constant()
        if val % divisor != 0:
            raise TileTypeError(
                f"Value {val} is not divisible by {divisor}: "
                f"`assume_divisible_by` contradicts a known constant")
        return x
    return add_operation(AssumeDivBy, x.get_type(), x=x, divisor=divisor)


@impl(ct.assume_divisible_by)
def assume_divisible_by_impl(x: Var, divisor: Var) -> Var:
    ty = x.get_type()
    if not is_scalar(ty, is_integral):
        raise TileTypeError(
            f"`assume_divisible_by` requires an integer scalar, got {ty}")
    divisor_val = require_constant_int(divisor)
    if divisor_val < 1:
        raise TileTypeError(
            f"`assume_divisible_by` requires a positive divisor, got {divisor_val}")
    return assume_div_by(x, divisor_val)


@dataclass(eq=False)
class AssumeBounded(Operation, opcode="assume_bounded"):
    lower_bound: int | None = attribute()
    upper_bound: int | None = attribute()
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.x)
        type_id = ctx.typeid_of(self.result_var)
        pred = bc.Bounded(lb=self.lower_bound, ub=self.upper_bound)
        return bc.encode_AssumeOp(ctx.builder, type_id, x, pred)


def assume_bounded(x: Var, lower_bound: int | None, upper_bound: int | None) -> Var:
    return add_operation(AssumeBounded, x.get_type(), x=x,
                         lower_bound=lower_bound, upper_bound=upper_bound)


@dataclass(eq=False)
class MakeListView(Operation, opcode="make_list_view"):
    base_ptr: Var = operand()
    length: Var = operand()

    def generate_bytecode(self, ctx: "BytecodeContext"):
        ty = self.result_var.get_type()
        assert isinstance(ty, ListTy)
        item_size = get_list_item_repr_size_in_words(ty.item_type)
        tv_ty = tensor_view_typeid_for_list(ctx.type_table, item_size)
        pv_tile_shape = 1, get_list_partition_view_tile_size(item_size)
        # On padding value:
        # We intentionally choose to have padding_value Missing, such that
        # reading a list out of bound results in undefined memref
        # A safer choice is to have zero padding, which result in a zero shaped
        # memref which cannot be written to, but we do not want user to rely
        # on the consequence of this specific implementation.
        # Another alternative is to use a different encoding the shape/stride
        # such that zero padding will end up being FFFFF once read back. This way
        # out of bound access of list[array] will result in a memref at 0x0 with 0xFFFF
        # shape and stride, such that when there is accidental write to it, guarantees
        # illegal memory access.
        pv_ty = ctx.type_table.partition_view(pv_tile_shape, tv_ty, [0, 1],
                                              bc.PaddingValue.Missing)
        ptr = ctx.get_value(self.base_ptr)
        length = ctx.get_value(self.length)
        tv = bc.encode_MakeTensorViewOp(ctx.builder, tv_ty, ptr, [length], [])
        return bc.encode_MakePartitionViewOp(ctx.builder, pv_ty, tv)


@dataclass(eq=False)
class TileNumBlocks(Operation, opcode="tile_num_blocks"):
    axis: int = attribute()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        t = ctx.typeid_of(self.result_var)
        return bc.encode_GetNumTileBlocksOp(ctx.builder, t, t, t)[self.axis]


@impl(ct.num_blocks)
def num_blocks(axis: Var) -> Var:
    axis = require_constant_int(axis)
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileNumBlocks, TileTy(datatype.default_int_type), axis=axis)


@array_impl_registry.impl(ct.Array.slice)
def array_slice_impl(self: Var, axis: Var, start: Var, stop: Var) -> Var:
    array_ty = require_array_type(self)
    const_axis = normalize_axis(require_constant_int(axis), array_ty.ndim)
    require_signed_integer_0d_tile_type(start)
    require_signed_integer_0d_tile_type(stop)

    def maybe_const_int(v: Var):
        if v.is_constant():
            v_int = v.get_constant()
            assert isinstance(v_int, int)
            return v_int
        return None

    const_start = maybe_const_int(start)
    const_stop = maybe_const_int(stop)
    if const_start is not None and const_start < 0:
        raise TileTypeError("Slice start must be non-negative")
    if const_stop is not None and const_stop < 0:
        raise TileTypeError("Slice stop must be non-negative")
    if const_start is not None and const_stop is not None and const_stop < const_start:
        raise TileTypeError("Slice stop must be greater than or equal to start")

    const_axis_size = (const_stop - const_start
                       if const_start is not None and const_stop is not None else None)
    new_shape_ty = tuple(const_axis_size if i == const_axis else dim
                         for i, dim in enumerate(array_ty.shape))
    new_array_ty = ArrayTy(
        array_ty.dtype,
        shape=new_shape_ty,
        strides=array_ty.strides,
        typing_hooks=array_ty.typing_hooks,
        index_dtype=array_ty.index_dtype,
        memory_space=array_ty.memory_space,
    )

    array_val = self.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    static_stride = array_ty.strides[const_axis]
    if static_stride == 1:
        offset = start  # skip multiplication for unit stride
    elif static_stride is not None:
        offset = binary_arithmetic_tensorlike("mul", start, loosely_typed_const(static_stride))
    else:
        offset = binary_arithmetic_tensorlike("mul", start, array_val.strides[const_axis])

    new_base_ptr = pointer_offset(array_val.base_ptr, astype(offset, datatype.uint64))
    axis_new_shape = astype(binary_arithmetic_tensorlike("sub", stop, start), array_ty.index_dtype)
    new_shape = tuple(
        axis_new_shape if i == const_axis else s for i, s in enumerate(array_val.shape)
    )

    [ret] = unflatten_aggregates(
        (new_base_ptr,) + new_shape + array_val.strides,
        (new_array_ty,), (new_array_ty,)
    )
    return ret


def _check_load_store_hints(latency_value: int | None, allow_tma_value: bool | None = None) -> None:
    if latency_value is not None:
        if not (1 <= latency_value <= 10):
            raise TileValueError(f"Latency must be between 1 and 10, got {latency_value}")
    if allow_tma_value is not None:
        if not isinstance(allow_tma_value, bool):
            raise TileTypeError(f"Allow TMA must be a boolean, got {allow_tma_value}")


@dataclass(eq=False)
class MakePartitionView(Operation, opcode="make_partition_view"):
    array: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        partition_view_ty = self.result_var.get_type()
        return bc.encode_MakePartitionViewOp(ctx.builder,
                                             typeid(ctx.type_table, partition_view_ty),
                                             ctx.get_value(self.array))


def _make_partition_view(array: Var, tile_shape: Sequence[int],
                         order: Sequence[int],
                         padding_mode: PaddingMode) -> Var:
    array_ty = array.get_type()
    assert isinstance(array_ty, ArrayTy)
    view_ty = PartitionViewTy(array_ty, tuple(tile_shape), tuple(order), padding_mode)
    ret = add_operation(MakePartitionView, view_ty, array=array)
    ret.set_aggregate(array.get_aggregate())
    return ret


@dataclass(eq=False)
class MakeStridedView(Operation, opcode="make_strided_view"):
    array: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        strided_view_ty = self.result_var.get_type()
        return bc.encode_MakeStridedViewOp(ctx.builder,
                                           typeid(ctx.type_table, strided_view_ty),
                                           ctx.get_value(self.array))


def _make_strided_view(array: Var, tile_shape: Sequence[int],
                       traversal_steps: Sequence[int],
                       order: Sequence[int],
                       padding_mode: PaddingMode) -> Var:
    array_ty = array.get_type()
    assert isinstance(array_ty, ArrayTy)
    view_ty = StridedViewTy(array_ty, tuple(tile_shape), tuple(traversal_steps),
                            tuple(order), padding_mode)
    ret = add_operation(MakeStridedView, view_ty, array=array)
    ret.set_aggregate(array.get_aggregate())
    return ret


def _use_strided_view(traversal_steps: Optional[Sequence[int]],
                      tile_shape: Sequence[int]) -> bool:
    return traversal_steps is not None and tuple(traversal_steps) != tuple(tile_shape)


def _materialize_tiled_view(array: Var,
                            tile_shape: Sequence[int],
                            order: Sequence[int],
                            padding_mode: PaddingMode,
                            traversal_steps: Optional[Sequence[int]]) -> Var:
    if _use_strided_view(traversal_steps, tile_shape):
        return _make_strided_view(array, tile_shape, traversal_steps, order, padding_mode)

    return _make_partition_view(array, tile_shape, order, padding_mode)


@dataclass(eq=False)
class MakeGatherScatterView(Operation, opcode="make_gather_scatter_view"):
    array: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        gs_view_ty = self.result_var.get_type()
        return bc.encode_MakeGatherScatterViewOp(ctx.builder,
                                                 typeid(ctx.type_table, gs_view_ty),
                                                 ctx.get_value(self.array))


def make_gather_scatter_view(array: Var, tile_shape: Sequence[int],
                             sparse_dim: int,
                             padding_mode: PaddingMode) -> Var:
    array_ty = array.get_type()
    assert isinstance(array_ty, ArrayTy)
    view_ty = GatherScatterViewTy(array_ty, tuple(tile_shape), sparse_dim, padding_mode)
    ret = add_operation(MakeGatherScatterView, view_ty, array=array)
    ret.set_aggregate(array.get_aggregate())
    return ret


def _uniform_tuple(val: Any, *, rank: int):
    return (val,) * rank


def _check_bounds_to_inbounds(check_bounds: Var, rank: int) -> tuple[bool, ...]:
    check = require_constant_bool(check_bounds)
    inbounds = _uniform_tuple(not check, rank=rank)
    if not check:
        cur_version = Builder.get_current().ir_ctx.tileiras_version
        if cur_version < BytecodeVersion.V_13_4:
            raise TileUnsupportedFeatureError(
                f"'check_bounds=False' requires tileiras {BytecodeVersion.V_13_4.as_string()}"
                f" or later. Current version is {cur_version.as_string()}.")
    return inbounds


@dataclass(eq=False)
class TileLoad(Operation, opcode="tile_load", memory_effect=MemoryEffect.LOAD):
    latency: Optional[int] = attribute()
    allow_tma: Optional[bool] = attribute()
    memory_order: MemoryOrder = attribute(default=MemoryOrder.WEAK)
    memory_scope: MemoryScope = attribute(default=MemoryScope.NONE)
    inbounds: tuple[bool, ...] = attribute(default=())
    view: Var = operand()
    index: tuple[Var, ...] = operand()
    token: Optional[Var] = operand(default=None)

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.WEAK
    )

    VALID_MEMORY_SCOPES = (
        MemoryScope.NONE,
        MemoryScope.BLOCK,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        tile_type: TileTy = self.result_vars[0].get_type()
        view_ty = self.view.get_type()
        keep_i64 = (isinstance(view_ty, PartitionViewTy)
                    and view_ty.array_ty.index_dtype.bitwidth > 32)
        res, res_token = bc.encode_LoadViewTkoOp(
            ctx.builder,
            tile_type=typeid(ctx.type_table, tile_type),
            result_token_type=ctx.type_table.Token,
            view=ctx.get_value(self.view),
            index=ctx.index_tuple(self.index, keep_i64=keep_i64),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=memory_order_to_bytecode[self.memory_order],
            memory_scope=memory_scope_to_bytecode[self.memory_scope],
            optimization_hints=ctx.load_store_hints(self.latency, self.allow_tma),
            inbounds=self.inbounds or _uniform_tuple(False, rank=len(self.index)),
        )
        return res, res_token


def _tile_load_impl_inner(array: Var, index_items: tuple[Var, ...], shape: Sequence[int],
                          order: Sequence[int], padding_mode: PaddingMode,
                          latency: Var, allow_tma: Var,
                          inbounds: tuple[bool, ...] = (),
                          traversal_steps: Optional[tuple[int, ...]] = None,
                          memory_order: MemoryOrder = MemoryOrder.WEAK,
                          memory_scope: MemoryScope = MemoryScope.NONE) -> Var:
    array_ty = require_array_type(array)
    broadcasted_shape = (1,) * array_ty.ndim if len(shape) == 0 else shape
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    _check_load_store_hints(latency, allow_tma)

    # Promote indices to i64 for big arrays so that blockId * tileSize
    # doesn't overflow i32 in the backend's address computation.
    if array_ty.index_dtype.bitwidth > 32:
        index_items = tuple(astype(idx, array_ty.index_dtype) for idx in index_items)

    view = _materialize_tiled_view(array, broadcasted_shape, order, padding_mode,
                                   traversal_steps)
    res_ty = TileTy(array_ty.dtype, broadcasted_shape)
    result, _token = add_operation_variadic(TileLoad, (res_ty, TokenTy()),
                                            view=view, index=index_items, latency=latency,
                                            allow_tma=allow_tma, memory_order=memory_order,
                                            memory_scope=memory_scope, inbounds=inbounds)
    return reshape(result, shape)


@impl(ct.Array.get_raw_memory)
def get_raw_memory_impl(self: Var) -> Var:
    array_ty = require_array_type(self)
    array_val = self.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    base_ptr = array_val.base_ptr
    raw_mem_ty = RawArrayMemoryTy(array_ty.dtype)
    [ret] = unflatten_aggregates((base_ptr,), (raw_mem_ty,), (raw_mem_ty,))
    return ret


def _process_raw_array_memory_pointer_and_mask(
        raw_array_memory: Var, offset: Var, mask: Optional[Var]):
    raw_mem_ty = require_raw_array_memory_type(raw_array_memory)
    raw_mem_val = raw_array_memory.get_aggregate()
    assert isinstance(raw_mem_val, RawArrayMemoryValue)
    base_ptr = raw_mem_val.base_ptr

    offset = astype(offset, datatype.uint64)
    pointer = pointer_offset(base_ptr, offset)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape
    array_dtype = raw_mem_ty.dtype

    final_mask = _process_custom_mask(mask, None, pointer_shape)
    return pointer, pointer_shape, final_mask, array_dtype


@impl(ct.RawArrayMemory.load_offset)
def raw_array_memory_load_offset_impl(self: Var, offset: Var, mask: Var,
                                      padding_value: Var, latency: Var) -> Var:
    pointer, pointer_shape, final_mask, array_dtype = _process_raw_array_memory_pointer_and_mask(
        self, offset, mask)

    if padding_value.is_constant() and padding_value.get_constant() is None:
        padding_var: Optional[Var] = None
    else:
        padding_ty = require_tile_type(padding_value)
        padding_shape = padding_ty.shape
        if not is_shape_broadcastable_to(padding_shape, pointer_shape):
            raise TileTypeError(f"Padding shape {padding_shape} is not broadcastable to the"
                                f" offset shape {pointer_shape}")
        padding_var = implicit_cast(padding_value, array_dtype, "Invalid padding value")
        padding_var = broadcast_to(padding_var, pointer_shape)

    latency_val = require_optional_constant_int(latency)
    _check_load_store_hints(latency_val)
    result, _token = load_pointer(pointer, final_mask, padding_var, latency_val)
    return result


@impl(ct.RawArrayMemory.store_offset)
def raw_array_memory_store_offset_impl(self: Var, offset: Var, value: Var,
                                       mask: Var, latency: Var) -> None:
    pointer, pointer_shape, final_mask, array_dtype = _process_raw_array_memory_pointer_and_mask(
        self, offset, mask)

    value = _get_scatter_value(value, pointer_shape, array_dtype, "Value",
                               array_name="RawArrayMemory")

    latency_val = require_optional_constant_int(latency)
    _check_load_store_hints(latency_val)
    store_pointer(pointer, value, final_mask, latency_val)


@impl(ct.load)
def tile_load_impl(array: Var, index: Var, shape: Var, order: Var,
                   padding_mode: Var, check_bounds: Var, latency: Var, allow_tma: Var,
                   memory_order: Var, memory_scope: Var) -> Var:
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if array_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the array rank {array_ty.ndim}")

    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    order = require_constant_axis_order(order, array_ty.ndim)
    padding_mode = require_constant_enum(padding_mode, PaddingMode)
    inbounds = _check_bounds_to_inbounds(check_bounds, array_ty.ndim)
    mem_order = require_constant_enum(memory_order, MemoryOrder)
    mem_scope = require_constant_enum(memory_scope, MemoryScope)
    validate_memory_order_and_scope(mem_order, mem_scope, TileLoad)
    return _tile_load_impl_inner(array, index_items, shape, order, padding_mode, latency, allow_tma,
                                 inbounds=inbounds, memory_order=mem_order, memory_scope=mem_scope)


@dataclass(eq=False)
class TileStore(Operation, opcode="tile_store", memory_effect=MemoryEffect.STORE):
    latency: Optional[int] = attribute()
    allow_tma: Optional[bool] = attribute()
    memory_order: MemoryOrder = attribute(default=MemoryOrder.WEAK)
    memory_scope: MemoryScope = attribute(default=MemoryScope.NONE)
    inbounds: tuple[bool, ...] = attribute(default=())
    view: Var = operand()
    index: tuple[Var, ...] = operand()
    tile: Var = operand()
    token: Optional[Var] = operand(default=None)

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED, MemoryOrder.RELEASE, MemoryOrder.WEAK
    )

    VALID_MEMORY_SCOPES = (
        MemoryScope.NONE,
        MemoryScope.BLOCK,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        view_ty = self.view.get_type()
        keep_i64 = (isinstance(view_ty, PartitionViewTy)
                    and view_ty.array_ty.index_dtype.bitwidth > 32)
        return bc.encode_StoreViewTkoOp(
            ctx.builder,
            result_token_type=ctx.type_table.Token,
            tile=ctx.get_value(self.tile),
            view=ctx.get_value(self.view),
            index=ctx.index_tuple(self.index, keep_i64=keep_i64),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=memory_order_to_bytecode[self.memory_order],
            memory_scope=memory_scope_to_bytecode[self.memory_scope],
            optimization_hints=ctx.load_store_hints(self.latency, self.allow_tma),
            inbounds=self.inbounds or _uniform_tuple(False, rank=len(self.index))
        )


def _tile_store_impl_inner(array: Var, index_items: tuple[Var, ...], tile: Var,
                           order: Sequence[int], latency: Var, allow_tma: Var,
                           inbounds: tuple[bool, ...] = (),
                           traversal_steps: Optional[tuple[int, ...]] = None,
                           memory_order: MemoryOrder = MemoryOrder.WEAK,
                           memory_scope: MemoryScope = MemoryScope.NONE):
    array_ty = require_array_type(array)
    tile_ty = require_tile_type(tile)
    broadcasted_shape = (1,) * array_ty.ndim if len(tile_ty.shape) == 0 else tile_ty.shape
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    _check_load_store_hints(latency, allow_tma)

    # Promote indices to i64 for big arrays so that blockId * tileSize
    # doesn't overflow i32 in the backend's address computation.
    if array_ty.index_dtype.bitwidth > 32:
        index_items = tuple(astype(idx, array_ty.index_dtype) for idx in index_items)

    tile = reshape(tile, broadcasted_shape)
    view = _materialize_tiled_view(array, broadcasted_shape, order, PaddingMode.UNDETERMINED,
                                   traversal_steps)
    add_operation(TileStore, TokenTy(), view=view, index=index_items, tile=tile,
                  latency=latency, allow_tma=allow_tma, memory_order=memory_order,
                  memory_scope=memory_scope, inbounds=inbounds)


@impl(ct.store)
def tile_store_impl(array: Var, index: Var, tile: Var, order: Var,
                    check_bounds: Var, latency: Var, allow_tma: Var,
                    memory_order: Var, memory_scope: Var):
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if array_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the array rank {array_ty.ndim}")

    tile = implicit_cast(tile, array_ty.dtype, "Stored tile is incompatible with array's dtype")
    order = require_constant_axis_order(order, array_ty.ndim)
    inbounds = _check_bounds_to_inbounds(check_bounds, array_ty.ndim)
    mem_order = require_constant_enum(memory_order, MemoryOrder)
    mem_scope = require_constant_enum(memory_scope, MemoryScope)
    validate_memory_order_and_scope(mem_order, mem_scope, TileStore)
    _tile_store_impl_inner(array, index_items, tile, order, latency, allow_tma,
                           inbounds=inbounds, memory_order=mem_order, memory_scope=mem_scope)


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    latency: Optional[int] = attribute()
    pointer: Var = operand()
    mask: Optional[Var] = operand(default=None)
    padding_value: Optional[Var] = operand(default=None)
    token: Optional[Var] = operand(default=None)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        return bc.encode_LoadPtrTkoOp(
            ctx.builder,
            result_type=ctx.typeid_of(self.result_vars[0]),
            result_token_type=ctx.type_table.Token,
            source=ctx.get_value(self.pointer),
            mask=None if self.mask is None else ctx.get_value(self.mask),
            paddingValue=ctx.get_value(self.padding_value),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
            memory_scope=None,
            optimization_hints=ctx.load_store_hints(self.latency, None),
        )


def load_pointer(pointer: Var, mask: Optional[Var], padding_value: Optional[Var],
                 latency: Optional[int]) -> tuple[Var[TileTy], Var[TokenTy]]:
    pointer_ty = pointer.get_type()
    shape = pointer_ty.shape
    info = PointerInfo(pointer_ty.dtype)
    dtype = info.pointee_dtype
    result_ty = TileTy(dtype, shape)
    res, tok = add_operation_variadic(LoadPointer, (result_ty, TokenTy()),
                                      pointer=pointer, mask=mask, padding_value=padding_value,
                                      latency=latency)
    return res, tok


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    latency: Optional[int] = attribute()
    pointer: Var = operand()
    value: Var = operand()
    mask: Optional[Var] = operand(default=None)
    token: Optional[Var] = operand(default=None)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return bc.encode_StorePtrTkoOp(
            ctx.builder,
            result_token_type=ctx.type_table.Token,
            destination=ctx.get_value(self.pointer),
            value=ctx.get_value(self.value),
            mask=None if self.mask is None else ctx.get_value(self.mask),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
            memory_scope=None,
            optimization_hints=ctx.load_store_hints(self.latency, None),
        )


def store_pointer(pointer: Var[TileTy], value: Var[TileTy], mask: Optional[Var],
                  latency: Optional[int]) -> Var[TokenTy]:
    return add_operation(StorePointer, TokenTy(),
                         pointer=pointer, value=value, mask=mask, latency=latency)


@dataclass(eq=False)
class PointerOffset(Operation, opcode="pointer_offset"):
    pointer: Var = operand()
    offset: Var = operand()

    @override
    def generate_bytecode(self, ctx: "BytecodeContext"):
        res_typeid = ctx.typeid_of(self.result_var)
        pointer = ctx.get_value(self.pointer)
        offset = ctx.get_value(self.offset)
        return bc.encode_OffsetOp(ctx.builder, res_typeid, pointer, offset)


def pointer_offset(pointer: Var, offset: Var) -> Var:
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape

    offset_ty = offset.get_type()
    offset_shape = offset_ty.shape

    common_shape = broadcast_shapes2(pointer_shape, offset_shape)
    pointer = broadcast_to(pointer, common_shape)
    offset = broadcast_to(offset, common_shape)
    result_ty = TileTy(pointer_ty.dtype, common_shape)
    return add_operation(PointerOffset, result_ty, pointer=pointer, offset=offset)


@impl(ct.gather)
def gather_impl(array: Var, indices: Var, mask: Var, padding_value: Var,
                check_bounds: Var, latency: Var) -> Var:
    pointer, final_mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds, mask)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape

    # Handle the padding value
    padding_ty = require_tile_type(padding_value)
    padding_shape = padding_ty.shape
    if not is_shape_broadcastable_to(padding_shape, pointer_shape):
        raise TileTypeError(f"Padding shape {padding_shape} is not broadcastable to the"
                            f" index shape {pointer_ty}")
    array_dtype = array.get_type().dtype

    padding_value = implicit_cast(padding_value, array_dtype, "Invalid padding value")
    padding_value = broadcast_to(padding_value, pointer_shape)

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    _check_load_store_hints(latency)
    result, _token = load_pointer(pointer, final_mask, padding_value, latency)
    return result


@impl(ct.scatter)
def scatter_impl(array: Var, indices: Var, value: Var, mask: Var,
                 check_bounds: Var, latency: Var):
    pointer, final_mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds, mask)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape

    # Handle the `value`
    array_dtype = array.get_type().dtype
    value = _get_scatter_value(value, pointer_shape, array_dtype, "Value")

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    _check_load_store_hints(latency)

    store_pointer(pointer, value, final_mask, latency)


def _get_scatter_value(value: Var, pointer_shape: Tuple[int, ...], array_dtype: DType,
                       value_name: str, cast_dtype: bool = True,
                       array_name: str = "array") -> Var:
    value_ty = require_tile_type(value)
    value_shape = value_ty.shape

    if not is_shape_broadcastable_to(value_shape, pointer_shape):
        raise TileTypeError(f"{value_name} shape {value_shape} is not broadcastable"
                            f" to the index shape {pointer_shape}")

    if cast_dtype:
        value = implicit_cast(value, array_dtype,
                              f"Stored value is incompatible with {array_name}'s dtype")
    return broadcast_to(value, pointer_shape)


def _process_custom_mask(mask: Optional[Var], bounds_mask: Optional[Var],
                         pointer_shape: Tuple[int, ...]) -> Optional[Var]:
    """
    Process and validate the custom mask parameter for gather/scatter operations.

    Args:
        mask: The user-provided mask (can be Python None or Var containing None)
        bounds_mask: The generated bounds-checking mask based on indices (or None)
        pointer_shape: The target shape that the mask should be broadcast to

    Returns:
        The final mask to use (custom AND bounds, or just one of them, or None)
    """
    # Check if mask is None (either Python None or Var containing None)
    if mask is None or (mask.is_constant() and mask.get_constant() is None):
        # No custom mask provided, return the bounds mask
        return bounds_mask

    # Validate the mask type
    mask_ty = require_tile_type(mask)
    mask_dtype = mask_ty.dtype

    if not is_boolean(mask_dtype):
        raise TileTypeError(f"Custom mask must have boolean dtype, but got {mask_dtype}")

    # Check that mask shape is broadcastable
    mask_shape = mask_ty.shape if isinstance(mask_ty, TileTy) else ()
    if not is_shape_broadcastable_to(mask_shape, pointer_shape):
        raise TileTypeError(f"Custom mask shape {mask_shape} is not broadcastable"
                            f" to the index shape {pointer_shape}")

    # Broadcast the mask to the pointer shape
    mask = broadcast_to(mask, pointer_shape)

    # Combine with bounds mask if both exist
    if bounds_mask is None:
        return mask
    else:
        return binary_bitwise_tensorlike("and_", bounds_mask, mask)


def _gather_scatter_pointer_and_mask(
        array: Var,
        indices: Var,
        check_bounds: Var,
        custom_mask: Optional[Var] = None) -> Tuple[Var, Optional[Var]]:
    check_bounds = require_constant_bool(check_bounds)
    array_ty = require_array_type(array)
    indices_ty = require_index_or_index_tuple_type(indices,
                                                   allow_nd_tiles=True, allow_unsigned=True)
    if isinstance(indices_ty, TupleTy):
        index_types = indices_ty.value_types
    else:
        index_types = indices_ty,

    if len(index_types) != array_ty.ndim:
        msg = (f"For array of rank {array_ty.ndim}, `indices` must be a tuple of length"
               f" {array_ty.ndim}")
        if array_ty.ndim == 1:
            msg += ", or a single scalar/tile"
        msg += f". However, `indices` has type {indices_ty}."
        raise TileTypeError(msg)

    # Check that indices are ints
    for dim, indty in enumerate(index_types):
        ind_dtype = get_dtype(indty)
        if not is_integral(ind_dtype):
            for_dim = f"for dimension {dim} " if len(index_types) > 1 else ""
            raise TileTypeError(f"Index {for_dim}has non-integer data type {ind_dtype}")

    # Calculate the common index shape
    index_shapes = tuple(indty.shape for indty in index_types)
    common_shape = ()
    for shape in index_shapes:
        try:
            common_shape = broadcast_shapes2(common_shape, shape)
        except BroadcastError:
            all_shapes = ", ".join(str(s) for s in index_shapes)
            raise TileTypeError(f"Index shapes {all_shapes}"
                                f" are not broadcastable to a common shape")

    # Calculate offset from indices (and the mask, if check_bounds is True)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    offset = None
    mask = None
    for dim in range(len(index_types)):
        if isinstance(indices_ty, TupleTy):
            ind = tuple_item(indices.get_aggregate(), dim)
        else:
            ind = indices

        ind = astype(ind, datatype.uint64)
        ind = broadcast_to(ind, common_shape)

        if check_bounds:
            array_size = array_val.shape[dim]
            array_size = astype(array_size, datatype.uint64)
            dim_mask = compare_tensorlike("lt", ind, array_size)
            if mask is None:
                mask = dim_mask
            else:
                mask = binary_bitwise_tensorlike("and_", mask, dim_mask)

        static_stride = array_ty.strides[dim]
        if static_stride == 1:
            offset_delta = ind
        else:
            if static_stride is None:
                stride = astype(array_val.strides[dim], datatype.uint64)
            else:
                stride = loosely_typed_const(static_stride)
            offset_delta = binary_arithmetic_tensorlike("mul", ind, stride)

        if offset is None:
            offset = offset_delta
        else:
            offset = binary_arithmetic_tensorlike("add", offset, offset_delta)

    # Offset the base pointer
    if offset is None:
        # 0-D array case
        pointer = array_val.base_ptr
        pointer_shape = ()
    else:
        pointer = pointer_offset(array_val.base_ptr, offset)
        pointer_shape = common_shape

    # Process custom mask and combine with bounds mask
    final_mask = _process_custom_mask(custom_mask, mask, pointer_shape)
    return pointer, final_mask


@dataclass(eq=False)
class TileAtomicCAS(Operation, opcode="tile_atomic_cas",
                    memory_effect=MemoryEffect.STORE):
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    pointer: Var = operand()
    expected: Var = operand()
    desired: Var = operand()
    mask: Optional[Var] = operand(default=None)
    token: Optional[Var] = operand(default=None)

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.RELEASE, MemoryOrder.ACQ_REL
    )

    VALID_MEMORY_SCOPES = (
        MemoryScope.NONE,
        MemoryScope.BLOCK,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    )

    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        return bc.encode_AtomicCASTkoOp(
            ctx.builder,
            result_type=ctx.typeid_of(self.result_vars[0]),
            result_token_type=ctx.type_table.Token,
            pointers=ctx.get_value(self.pointer),
            cmp=ctx.get_value(self.expected),
            val=ctx.get_value(self.desired),
            mask=None if self.mask is None else ctx.get_value(self.mask),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=memory_order_to_bytecode[self.memory_order],
            memory_scope=memory_scope_to_bytecode[self.memory_scope],
        )


def _atomic_cas_core(array_dtype: DType,
                     pointer: Var, pointer_shape,
                     mask,
                     expected: Var, desired: Var,
                     memory_order: Var, memory_scope: Var) -> Var:
    if array_dtype not in int_float_32_64_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    # Handle the `expected` and `desired` values
    expected = _get_scatter_value(expected, pointer_shape, array_dtype, "Expected value")
    desired = _get_scatter_value(desired, pointer_shape, array_dtype, "Desired value")

    # Handle `memory_order` and `memory_scope`
    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)
    validate_memory_order_and_scope(memory_order, memory_scope, TileAtomicCAS)

    result_ty = TileTy(array_dtype, pointer_shape)
    result, _token = add_operation_variadic(TileAtomicCAS, (result_ty, TokenTy()),
                                            pointer=pointer, expected=expected, desired=desired,
                                            mask=mask, memory_order=memory_order,
                                            memory_scope=memory_scope)
    return result


@impl(ct.atomic_cas)
def atomic_cas_impl(array: Var, indices: Var, expected: Var, desired: Var, check_bounds: Var,
                    memory_order: Var, memory_scope: Var) -> Var:
    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_shape = pointer.get_type().shape
    array_dtype = array.get_type().dtype
    return _atomic_cas_core(array_dtype, pointer, pointer_shape, mask,
                            expected, desired, memory_order, memory_scope)


@impl(ct.RawArrayMemory.atomic_cas_offset)
def raw_array_memory_atomic_cas_impl(self: Var, offset: Var,
                                     expected: Var, desired: Var,
                                     mask: Var, memory_order: Var,
                                     memory_scope: Var) -> Var:
    pointer, pointer_shape, final_mask, array_dtype = _process_raw_array_memory_pointer_and_mask(
        self, offset, mask)
    return _atomic_cas_core(array_dtype, pointer, pointer_shape, final_mask,
                            expected, desired, memory_order, memory_scope)


class AtomicRMWMode(enum.Enum):
    BITWISE_AND = bc.AtomicRMWMode.AND
    BITWISE_OR = bc.AtomicRMWMode.OR
    BITWISE_XOR = bc.AtomicRMWMode.XOR
    ADD_INT = bc.AtomicRMWMode.ADD
    ADD_FLOAT = bc.AtomicRMWMode.ADDF
    MAX_SIGNED_INT = bc.AtomicRMWMode.MAX
    MIN_SIGNED_INT = bc.AtomicRMWMode.MIN
    MAX_UNSIGNED_INT = bc.AtomicRMWMode.UMAX
    MIN_UNSIGNED_INT = bc.AtomicRMWMode.UMIN
    EXCHANGE = bc.AtomicRMWMode.XCHG


@dataclass(eq=False)
class TileAtomicRMW(Operation, opcode="tile_atomic_rmw", memory_effect=MemoryEffect.STORE):
    mode: AtomicRMWMode = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    pointer: Var = operand()
    update: Var = operand()
    mask: Optional[Var] = operand(default=None)
    token: Optional[Var] = operand(default=None)

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED, MemoryOrder.ACQUIRE, MemoryOrder.RELEASE, MemoryOrder.ACQ_REL
    )

    VALID_MEMORY_SCOPES = (
        MemoryScope.NONE,
        MemoryScope.BLOCK,
        MemoryScope.DEVICE,
        MemoryScope.SYS,
    )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
        return bc.encode_AtomicRMWTkoOp(
            ctx.builder,
            result_type=ctx.typeid_of(self.result_vars[0]),
            result_token_type=ctx.type_table.Token,
            pointers=ctx.get_value(self.pointer),
            arg=ctx.get_value(self.update),
            mask=None if self.mask is None else ctx.get_value(self.mask),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=memory_order_to_bytecode[self.memory_order],
            memory_scope=memory_scope_to_bytecode[self.memory_scope],
            mode=self.mode._value_
        )


int_32_64_dtypes = (datatype.int32, datatype.int64, datatype.uint32, datatype.uint64)
int_float_32_64_dtypes = (*int_32_64_dtypes, datatype.float32, datatype.float64)


def _select_rmw_mode(int_mode: Optional[AtomicRMWMode],
                     uint_mode: Optional[AtomicRMWMode],
                     float_mode: Optional[AtomicRMWMode],
                     dtype: DType):
    if is_float(dtype):
        mode = float_mode
    elif is_integral(dtype):
        mode = int_mode if is_signed(dtype) else uint_mode
    else:
        mode = None
    assert mode is not None
    return mode


def _cast_rmw_update_dtype(update: Var, target_dtype: DType, bitwise: bool) -> Var:
    update_dtype = require_tile_type(update).dtype
    if bitwise:
        if update_dtype != target_dtype:
            raise TileTypeError(
                "Bitwise atomic read-modify-write operations require the update "
                f"dtype ({update_dtype}) to exactly match the target dtype "
                f"({target_dtype})"
            )
        return update
    return implicit_cast(update, target_dtype, "Update is incompatible with the target dtype")


def _atomic_rmw_core(int_mode: Optional[AtomicRMWMode],
                     uint_mode: Optional[AtomicRMWMode],
                     float_mode: Optional[AtomicRMWMode],
                     bitwise: bool,
                     supported_dtypes: Sequence[DType],
                     array_dtype: DType,
                     pointer: Var, pointer_shape,
                     mask,
                     update: Var, memory_order: Var, memory_scope: Var) -> Var:
    if array_dtype not in supported_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    update = _get_scatter_value(update, pointer_shape, array_dtype, "Update",
                                cast_dtype=False)
    update = _cast_rmw_update_dtype(update, array_dtype, bitwise)
    mode = _select_rmw_mode(int_mode, uint_mode, float_mode, array_dtype)

    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)
    validate_memory_order_and_scope(memory_order, memory_scope, TileAtomicRMW)

    result_ty = TileTy(array_dtype, pointer_shape)
    result, _token = add_operation_variadic(TileAtomicRMW, (result_ty, TokenTy()),
                                            mode=mode, pointer=pointer, update=update,
                                            mask=mask, memory_order=memory_order,
                                            memory_scope=memory_scope)
    return result


@dataclass(frozen=True)
class _AtomicRMWSpec:
    int_mode: Optional[AtomicRMWMode]
    uint_mode: Optional[AtomicRMWMode]
    float_mode: Optional[AtomicRMWMode]
    bitwise: bool
    supported_dtypes: Sequence[DType]

    def fixed_args(self) -> list:
        return [self.int_mode, self.uint_mode, self.float_mode,
                self.bitwise, self.supported_dtypes]


_ATOMIC_RMW_SPECS: dict[str, _AtomicRMWSpec] = {
    "xchg": _AtomicRMWSpec(
        AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE,
        False, int_float_32_64_dtypes),
    "add": _AtomicRMWSpec(
        AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_FLOAT,
        False, (*int_float_32_64_dtypes, datatype.float16, datatype.bfloat16)),
    "min": _AtomicRMWSpec(
        AtomicRMWMode.MIN_SIGNED_INT, AtomicRMWMode.MIN_UNSIGNED_INT, None,
        False, int_32_64_dtypes),
    "max": _AtomicRMWSpec(
        AtomicRMWMode.MAX_SIGNED_INT, AtomicRMWMode.MAX_UNSIGNED_INT, None,
        False, int_32_64_dtypes),
    "and": _AtomicRMWSpec(
        AtomicRMWMode.BITWISE_AND, AtomicRMWMode.BITWISE_AND, None,
        True, int_32_64_dtypes),
    "or": _AtomicRMWSpec(
        AtomicRMWMode.BITWISE_OR, AtomicRMWMode.BITWISE_OR, None,
        True, int_32_64_dtypes),
    "xor": _AtomicRMWSpec(
        AtomicRMWMode.BITWISE_XOR, AtomicRMWMode.BITWISE_XOR, None,
        True, int_32_64_dtypes),
}


def _register_atomic_rmw_impls(stubs, **impl_kwargs):
    def decorator(f):
        for op, stub in stubs.items():
            f = impl(stub,
                     fixed_args=_ATOMIC_RMW_SPECS[op].fixed_args(),
                     **impl_kwargs)(f)
        return f
    return decorator


_SCATTER_ATOMIC_RMW_STUBS = {
    "xchg": ct.atomic_xchg, "add": ct.atomic_add, "min": ct.atomic_min,
    "max":  ct.atomic_max,  "and": ct.atomic_and, "or":  ct.atomic_or,
    "xor":  ct.atomic_xor,
}


@_register_atomic_rmw_impls(_SCATTER_ATOMIC_RMW_STUBS)
def atomic_rmw_impl(int_mode: Optional[AtomicRMWMode],
                    uint_mode: Optional[AtomicRMWMode],
                    float_mode: Optional[AtomicRMWMode],
                    bitwise: bool,
                    supported_dtypes: Sequence[DType],
                    # --- end of fixed args ---
                    array: Var, indices: Var, update: Var,
                    check_bounds: Var, memory_order: Var, memory_scope: Var):
    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_shape = pointer.get_type().shape
    array_dtype = array.get_type().dtype
    return _atomic_rmw_core(int_mode, uint_mode, float_mode, bitwise, supported_dtypes,
                            array_dtype, pointer, pointer_shape, mask,
                            update, memory_order, memory_scope)


_RAW_MEM_ATOMIC_RMW_STUBS = {
    "xchg": ct.RawArrayMemory.atomic_xchg_offset,
    "add":  ct.RawArrayMemory.atomic_add_offset,
    "min":  ct.RawArrayMemory.atomic_min_offset,
    "max":  ct.RawArrayMemory.atomic_max_offset,
    "and":  ct.RawArrayMemory.atomic_and_offset,
    "or":   ct.RawArrayMemory.atomic_or_offset,
    "xor":  ct.RawArrayMemory.atomic_xor_offset,
}


@_register_atomic_rmw_impls(_RAW_MEM_ATOMIC_RMW_STUBS)
def raw_array_memory_atomic_rmw_impl(int_mode: Optional[AtomicRMWMode],
                                     uint_mode: Optional[AtomicRMWMode],
                                     float_mode: Optional[AtomicRMWMode],
                                     bitwise: bool,
                                     supported_dtypes: Sequence[DType],
                                     # --- end of fixed args ---
                                     self: Var, offset: Var, update: Var,
                                     mask: Var, memory_order: Var, memory_scope: Var):
    pointer, pointer_shape, final_mask, array_dtype = _process_raw_array_memory_pointer_and_mask(
        self, offset, mask)
    return _atomic_rmw_core(int_mode, uint_mode, float_mode, bitwise, supported_dtypes,
                            array_dtype, pointer, pointer_shape, final_mask,
                            update, memory_order, memory_scope)


@dataclass(eq=False)
class MakeToken(Operation, opcode="make_token"):

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return bc.encode_MakeTokenOp(ctx.builder, ctx.type_table.Token)


def make_token(*, block: Block, res: Var, loc: Loc) -> None:
    make_token_op = MakeToken(result_vars=(res,), loc=loc)
    block.append(make_token_op)


@dataclass(eq=False)
class JoinTokens(Operation, opcode="join_tokens"):
    tokens: Tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        tokens = tuple(ctx.get_value(x) for x in self.tokens)
        return bc.encode_JoinTokensOp(ctx.builder, ctx.type_table.Token, tokens)


def join_tokens(tokens: Tuple[Var, ...], *, block: Block, res: Var, loc: Loc) -> None:
    join_tokens_op = JoinTokens(tokens=tokens, result_vars=(res,), loc=loc)
    block.append(join_tokens_op)


@dataclass(eq=False)
class NumTiles(Operation, opcode="num_tiles"):
    view: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        view_ty: PartitionViewTy = self.view.get_type()
        result_types = [ctx.type_table.tile(ctx.type_table.I32, ())] * len(view_ty.tile_shape)
        values = bc.encode_GetIndexSpaceShapeOp(ctx.builder, result_types, ctx.get_value(self.view))
        return values


def num_tiles(array: Var, shape: Sequence[int], order: Sequence[int],
              traversal_steps: Optional[Sequence[int]] = None) -> Tuple[Var, ...]:
    array_ty = require_array_type(array)
    broadcasted_shape = (1,) * array_ty.ndim if len(shape) == 0 else shape
    view = _materialize_tiled_view(array, broadcasted_shape, order, PaddingMode.UNDETERMINED,
                                   traversal_steps)
    result_tys = tuple(TileTy(datatype.default_int_type) for _s in broadcasted_shape)
    return add_operation_variadic(NumTiles, result_tys, view=view)


@impl(ct.num_tiles)
def num_tiles_impl(array: Var, axis: Var, shape: Var, order: Var) -> Var:
    array_ty = require_array_type(array)
    axis = require_constant_int(axis)
    axis = normalize_axis(axis, array_ty.ndim)
    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    order = require_constant_axis_order(order, array_ty.ndim)
    space_shape = num_tiles(array, shape, order)
    return space_shape[axis]


def _const(shape: Sequence[int], value: int | float | bool | tuple, dtype: DType) -> Var:
    res_ty = TileTy(dtype, shape)
    return strictly_typed_const(value, res_ty)


def full(shape: Sequence[int], fill_value: Var, dtype: DType) -> Var:
    if fill_value.is_constant():
        return _const(shape, fill_value.get_constant(), dtype)
    fill_value = astype(fill_value, dtype)
    return broadcast_to(fill_value, shape)


@impl(ct.full)
def full_impl(shape: Var, fill_value: Var, dtype: Var) -> Var:
    require_0d_tile_type(fill_value)
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full(shape, fill_value, dtype)


@impl(ct.ones)
def ones_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return _const(shape, 1, dtype)


@impl(ct.zeros)
def zeros_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return _const(shape, 0, dtype)


def _path_str(path: tuple[int, ...]) -> str:
    return "value" + "".join(f"[{i}]" for i in path)


def _tuple_shape(ty: Type, path: tuple[int, ...]) -> tuple[int, ...]:
    path_str = _path_str(path)
    if not isinstance(ty, TupleTy):
        if not isinstance(ty, TileTy):
            raise TileTypeError(
                f"Expected scalar elements at {path_str}; "
                f"got element of type {ty}")

        if ty.ndim != 0:
            raise TileTypeError(
                f"Expected scalar elements at {path_str}; "
                f"got a tile of shape {ty.shape}")

        assert is_scalar(ty)
        return ()

    n = len(ty)
    if not _is_power_of_2(n):
        raise TileTypeError(f"Tuple length {n} at {path_str} is not a power of 2")

    inner_shapes = {_tuple_shape(t, path + (i,)) for i, t in enumerate(ty.value_types)}
    if len(inner_shapes) != 1:
        raise TileTypeError(f"Tuple has non-uniform inner shapes at {path_str}")

    return (n,) + inner_shapes.pop()


def _flatten_tuple(value: Var) -> tuple[Var, ...]:
    value_ty = value.get_type()
    if not isinstance(value_ty, TupleTy):
        return (value,)
    return sum((_flatten_tuple(i) for i in value.get_aggregate().items), start=())


def _cat_tuple(tiles: tuple[Var, ...]) -> Var:
    if len(tiles) == 0:
        raise TileInternalError("Expected non-empty tile tuple")

    if len(tiles) == 1:
        require_0d_tile_type(tiles[0])
        return reshape(tiles[0], (1,))

    assert len(tiles) % 2 == 0
    mid = len(tiles) // 2
    left = _cat_tuple(tiles[:mid])
    right = _cat_tuple(tiles[mid:])
    return cat((left, right), axis=0)


@impl(ct.astile)
def astile_impl(value: Var, dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    value_ty = value.get_type()
    if is_scalar(value_ty):
        return astype(value, dtype)

    if not isinstance(value_ty, TupleTy):
        raise TileTypeError(
                f"Expected a scalar or (possibly nested) tuple of scalars; "
                f"got value of type {value_ty}")

    shape = _tuple_shape(value_ty, path=())
    tiles = _flatten_tuple(value)

    if value.is_constant():
        return _const(shape, value.get_constant(), dtype)

    tiles = tuple(astype(t, dtype) for t in tiles)
    flat = _cat_tuple(tiles)
    return reshape(flat, shape)


_TileShape = Tuple[int, ...]


def _matmul_broadcast_shape(x_shape: _TileShape, y_shape: _TileShape) -> \
        Tuple[_TileShape, _TileShape, _TileShape, _TileShape]:
    x_orig_ndim = len(x_shape)
    y_orig_ndim = len(y_shape)

    # Promote 1D tensors to 2D for matmul
    if x_orig_ndim == 1:
        x_shape = (1,) + x_shape

    if y_orig_ndim == 1:
        y_shape = y_shape + (1,)

    if x_shape[-1] != y_shape[-2]:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    # Compute result matrix shape
    try:
        batch_shape = datatype.broadcast_shapes(x_shape[:-2], y_shape[:-2])
    except TypeError:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    x_shape = batch_shape + x_shape[-2:]
    y_shape = batch_shape + y_shape[-2:]
    acc_shape = batch_shape + (x_shape[-2],) + (y_shape[-1],)

    output_shape = acc_shape
    # If x was 1D, squeeze the leading dim
    if x_orig_ndim == 1:
        output_shape = output_shape[:-2] + output_shape[-1:]
    # If y was 1D, squeeze the trailing dim
    if y_orig_ndim == 1:
        output_shape = output_shape[:-1]

    return (x_shape, y_shape, acc_shape, output_shape)


@dataclass(eq=False)
class TileMma(Operation, opcode="tile_mma"):
    use_fast_acc: bool = attribute(default=False)
    x: Var = operand()
    y: Var = operand()
    acc: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        y_value = ctx.get_value(self.y)
        acc_value = ctx.get_value(self.acc)
        res_typeid = ctx.typeid_of(self.result_var)

        x_type = ctx.typeof(self.x)
        y_type = ctx.typeof(self.y)
        if datatype.is_integral(x_type.dtype):
            signedness_lhs = datatype.get_signedness(x_type.dtype)
            signedness_rhs = datatype.get_signedness(y_type.dtype)
            return bc.encode_MmaIOp(ctx.builder, res_typeid, x_value, y_value,
                                    acc_value, signedness_lhs, signedness_rhs)
        else:
            return bc.encode_MmaFOp(ctx.builder, res_typeid, x_value, y_value,
                                    acc_value, fast_acc=self.use_fast_acc)


@impl(ct.mma)
def mma_impl(x: Var, y: Var, acc: Var, use_fast_acc: Var) -> Var:
    use_fast_acc = require_constant_bool(use_fast_acc)
    x_tile_type = require_tile_type(x)
    y_tile_type = require_tile_type(y)
    acc_tile_type = require_tile_type(acc)
    x_shape_orig = x_tile_type.shape
    y_shape_orig = y_tile_type.shape
    acc_shape_orig = acc_tile_type.shape
    if len(x_shape_orig) < 2:
        raise TileTypeError(f'Expect shape of `x` to be at least 2D, got {x_shape_orig}')
    if len(y_shape_orig) < 2:
        raise TileTypeError(f'Expect shape of `y` to be at least 2D, got {y_shape_orig}')
    x_shape, y_shape, _, output_shape = _matmul_broadcast_shape(x_shape_orig, y_shape_orig)
    if acc_shape_orig != output_shape:
        raise TileTypeError(f'Expect acc shape to be {output_shape}, got {acc_shape_orig}')
    if use_fast_acc:
        if x_tile_type.dtype not in (datatype.float8_e4m3fn, datatype.float8_e5m2):
            raise TileTypeError(
                f'use_fast_acc is only supported for fp8 input dtypes '
                f'(float8_e4m3fn, float8_e5m2), got {x_tile_type.dtype}')
        cur_version = Builder.get_current().ir_ctx.tileiras_version
        if cur_version < BytecodeVersion.V_13_3:
            raise TileUnsupportedFeatureError(
                f'use_fast_acc requires tileiras '
                f'{BytecodeVersion.V_13_3.as_string()} or later. '
                f'Current version is {cur_version.as_string()}.')
    datatype._resolve_mma_supported_dtype(x_tile_type.dtype, y_tile_type.dtype, acc_tile_type.dtype)
    x = promote_and_broadcast_to(x, TileTy(x_tile_type.dtype, x_shape))
    y = promote_and_broadcast_to(y, TileTy(y_tile_type.dtype, y_shape))
    return add_operation(TileMma, acc_tile_type, use_fast_acc=use_fast_acc, x=x, y=y, acc=acc)


@impl(ct.matmul)
@impl(operator.matmul, overload=(TileTy, TileTy))
def matmul_impl(x: Var, y: Var) -> Var:
    x_tile_type = require_tile_type(x)
    y_tile_type = require_tile_type(y)
    x_shape_orig = x_tile_type.shape
    y_shape_orig = y_tile_type.shape
    x_shape, y_shape, acc_shape, output_shape = _matmul_broadcast_shape(x_shape_orig, y_shape_orig)
    common_dtype = promote_dtypes(x_tile_type.dtype, y_tile_type.dtype)
    acc_dtype = datatype._resolve_mma_supported_dtype(common_dtype, common_dtype, None)
    x = promote_and_broadcast_to(x, TileTy(common_dtype, x_shape))
    if len(y_shape_orig) == 1:
        # When y is 1d, we cannot directly use cast for reshape + broadcast
        # because y is first reshaped to 2d by appending 1.
        # Therefore, we need to first reshape y from (k,) to (k, 1) and then
        # apply the reshape+broadcast rule for batch dims
        y_shape_2d = (y_shape_orig[0], 1)
        y = reshape(y, y_shape_2d)
    y = promote_and_broadcast_to(y, TileTy(common_dtype, y_shape))
    acc_ty = TileTy(acc_dtype, acc_shape)
    acc_value = strictly_typed_const(0, acc_ty)
    matmul_result = add_operation(TileMma, acc_ty, x=x, y=y, acc=acc_value)
    matmul_result = astype(matmul_result, common_dtype)
    ret = reshape(matmul_result, output_shape)
    return ret


@dataclass(eq=False)
class TileMmaScaled(Operation, opcode="tile_mma_scaled"):
    x: Var = operand()
    x_scale: Var = operand()
    y: Var = operand()
    y_scale: Var = operand()
    acc: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        x_scale_value = ctx.get_value(self.x_scale)
        y_value = ctx.get_value(self.y)
        y_scale_value = ctx.get_value(self.y_scale)
        acc_value = ctx.get_value(self.acc)
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_MmaFScaledOp(ctx.builder, res_typeid, x_value, y_value,
                                      acc_value, x_scale_value, y_scale_value)


def _verify_scaling_block_size(ty: TileTy, scale_ty: TileTy, k_axis: int,
                               name: str, scale_name: str):
    shape = ty.shape
    dtype = ty.dtype
    scale_shape = scale_ty.shape
    scale_dtype = scale_ty.dtype
    k_axis = normalize_axis(k_axis, len(shape))
    if any(x != y for i, (x, y) in enumerate(zip(shape, scale_shape, strict=True)) if i != k_axis):
        raise TileTypeError(
            f"{scale_name} shape {scale_shape} is not compatible with {name} shape {shape}. "
            f"All dimensions except K axis {k_axis} must match")

    allowed = datatype._get_mma_scaled_scaling_block_sizes(ty.dtype, scale_ty.dtype)
    scaling_block_size, rem = divmod(shape[k_axis], scale_shape[k_axis])
    if rem != 0 or scaling_block_size not in allowed:
        raise TileTypeError(
            f"For mma_scaled with dtype={dtype}, scale_dtype={scale_dtype}: "
            f"{name}.shape[{k_axis}] must be an exact multiple of {scale_name}.shape[{k_axis}] "
            f"with scaling block size B = K // K_s in {set(allowed)}, "
            f"got {name}.shape[{k_axis}] = {shape[k_axis]} and "
            f"{scale_name}.shape[{k_axis}] = {scale_shape[k_axis]}")


@impl(ct.mma_scaled, min_version=BytecodeVersion.V_13_3)
def mma_scaled_impl(x: Var, x_scale: Var, y: Var, y_scale: Var, acc: Var) -> Var:
    x_ty = require_tile_type(x)
    y_ty = require_tile_type(y)
    acc_ty = require_tile_type(acc)
    x_scale_ty = require_tile_type(x_scale)
    y_scale_ty = require_tile_type(y_scale)

    for name, shape in [("x", x_ty.shape), ("y", y_ty.shape),
                        ("acc", acc_ty.shape),
                        ("x_scale", x_scale_ty.shape),
                        ("y_scale", y_scale_ty.shape)]:
        if len(shape) not in [2, 3]:
            raise TileTypeError(
                f'Expect shape of `{name}` to be 2D or 3D, got {shape}')

    datatype._resolve_mma_scaled_supported_dtype(
        x_ty.dtype, x_scale_ty.dtype,
        y_ty.dtype, y_scale_ty.dtype,
        acc_ty.dtype)
    _verify_scaling_block_size(x_ty, x_scale_ty, k_axis=-1, name="x", scale_name="x_scale")
    _verify_scaling_block_size(y_ty, y_scale_ty, k_axis=-2, name="y", scale_name="y_scale")

    x_shape, y_shape, _, output_shape = _matmul_broadcast_shape(x_ty.shape, y_ty.shape)
    if acc_ty.shape != output_shape:
        raise TileTypeError(f'Expect acc shape to be {output_shape}, got {acc_ty.shape}')

    # Broadcast scale batch dims to match the broadcasted x/y batch dims
    batch = x_shape[:-2]
    x_scale_shape = TupleTy(batch + x_scale_ty.shape[-2:])
    y_scale_shape = TupleTy(batch + y_scale_ty.shape[-2:])

    x = promote_and_broadcast_to(x, TileTy(x_ty.dtype, x_shape))
    y = promote_and_broadcast_to(y, TileTy(y_ty.dtype, y_shape))
    x_scale = promote_and_broadcast_to(x_scale, TileTy(x_scale_ty.dtype, x_scale_shape))
    y_scale = promote_and_broadcast_to(y_scale, TileTy(y_scale_ty.dtype, y_scale_shape))
    return add_operation(TileMmaScaled, acc_ty,
                         x=x, x_scale=x_scale, y=y, y_scale=y_scale, acc=acc)


@dataclass(eq=False)
class TileReduce(Operation, opcode="tile_reduce"):
    identities: tuple[bool | int | float, ...] = attribute()
    axis: int = attribute()
    xs: tuple[Var, ...] = operand()
    body: Block = nested_block()

    @property
    def lhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[:len(self.xs)]

    @property
    def rhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[len(self.xs):]

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["do"]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, ...]:
        xs = tuple(ctx.get_value(x) for x in self.xs)
        res_typeids = tuple(ctx.typeid_of(v) for v in self.result_vars)

        identities = []
        param_type_ids = []
        for id_val, x in zip(self.identities, self.xs, strict=True):
            x_dtype = get_dtype(x.get_type())
            x_dtype_id = dtype_typeid(ctx.type_table, x_dtype)
            if datatype.is_float(x_dtype):
                x_dtype_bc = datatype.dtype_simple_bytecode_type(x_dtype)
                attr = bc.Float(float(id_val), x_dtype_bc, ctx.type_table)
            elif datatype.is_boolean(x_dtype):
                attr = bc.Bool(bool(id_val))
            else:
                assert datatype.is_integral(x_dtype)
                attr = bc.Integer(x_dtype_id, x_dtype.bitwidth, int(id_val))
            identities.append(attr)

            x_tile_typeid = ctx.type_table.tile(x_dtype_id, ())
            param_type_ids.append(x_tile_typeid)
            param_type_ids.append(x_tile_typeid)

        nested_builder = bc.encode_ReduceOp(
            ctx.builder,
            result_types=res_typeids,
            operands=xs,
            dim=self.axis,
            identities=identities
        )

        with nested_builder.new_block(param_type_ids) as block_args:
            for var, value in zip(self.body.params, block_args, strict=True):
                ctx.set_value(var, value)
            generate_bytecode_for_block(ctx, self.body)

        return nested_builder.done()


async def _get_reduce_scan_body_block(
    xs: tuple[Var, ...],
    body: Callable,
    *,
    op_name: Literal["reduction", "scan"],
) -> tuple[Block, tuple[TileTy, ...]]:
    """Build body block for reduce/scan. Caller passes result_shape; returns
    (body_block, result_types)."""
    builder = Builder.get_current()
    if isinstance(builder.block_restriction, ReduceScanRestriction):
        raise TileSyntaxError("Nested scan/reduction is not supported")

    block_params = []
    lhs_vars = []
    rhs_vars = []
    input_shape = ()
    for i, x in enumerate(xs):
        x_ty = x.get_type()
        assert isinstance(x_ty, TileTy)
        if i == 0:
            input_shape = x_ty.shape
        else:
            assert input_shape == x_ty.shape
        tile_0d_ty = TileTy(x_ty.dtype)
        for _ in range(2):
            var = builder.ir_ctx.make_temp(builder.loc)
            var.set_type(tile_0d_ty)
            block_params.append(var)
        lhs_vars.append(block_params[-2])
        rhs_vars.append(block_params[-1])

    with enter_nested_block(
            builder.loc,
            block_restriction=ReduceScanRestriction(op_name)) as body_block:
        body_block.params = tuple(block_params)
        body_results = await body(tuple(lhs_vars), tuple(rhs_vars))
        for body_res, x in zip(body_results, xs, strict=True):
            body_res_ty = body_res.get_type()
            assert body_res_ty.shape == ()
            assert body_res_ty.dtype == x.get_type().dtype

        add_operation_variadic(EndBranch, (), outputs=body_results)

    return body_block


async def raw_reduce(xs: tuple[Var, ...], identities: tuple[bool | int | float], axis: int,
                     body: Callable) -> tuple[Var, ...]:
    input_shape = require_tile_type(xs[0]).shape

    assert 0 <= axis < len(input_shape)
    result_shape = input_shape[:axis] + input_shape[axis + 1:]
    result_types = tuple(TileTy(x.get_type().dtype, result_shape) for x in xs)

    assert len(xs) == len(identities)

    body_block = await _get_reduce_scan_body_block(xs, body, op_name="reduction")

    return add_operation_variadic(TileReduce, result_types, xs=xs, identities=identities, axis=axis,
                                  body=body_block)


async def reduce(xs: tuple[Var, ...], identities: tuple[bool | int | float, ...],
                 axis: int | None | Iterable[int], keepdims: bool,
                 body: Callable) -> tuple[Var, ...]:
    if len(xs) == 0:
        raise TileTypeError("Need at least one input value to reduce")

    if len(xs) != len(identities):
        raise TileTypeError(f"Number of input values ({len(xs)}) doesn't match the"
                            f" number of identities ({len(identities)})")

    common_input_shape = ()

    x_types = tuple(require_tile_type(x) for x in xs)
    for x_ty in x_types:
        try:
            common_input_shape = broadcast_shapes2(common_input_shape, x_ty.shape)
        except BroadcastError:
            all_shapes = ", ".join(str(ty.shape) for ty in x_types)
            raise TileTypeError(f"Input shapes {all_shapes}"
                                f" are not broadcastable to a common shape")

    if axis is None:
        axis = tuple(range(len(common_input_shape)))
    else:
        if isinstance(axis, int):
            axis = (axis,)
        axis = sorted(normalize_axis(a, len(common_input_shape)) for a in axis)
        for a1, a2 in zip(axis, axis[1:]):
            if a1 == a2:
                raise TileTypeError(f"Repeated reduction axis {a1}")

    xs = tuple(broadcast_to(x, common_input_shape) for x in xs)
    for i, a in enumerate(axis):
        xs = await raw_reduce(xs, identities, a - i, body)

    result_shape = _get_reduction_shape(common_input_shape, axis, keepdims)
    return tuple(reshape(x, result_shape) for x in xs)


def _make_reduce_scan_body(
    func: Var,
    tuple_mode: bool,
    xs: tuple[Var, ...],
    op_name: Literal["Reduction", "Scan"],
) -> Callable:
    """Build the shared body(lhs, rhs) used by reduce_impl and scan_impl."""

    async def body(lhs, rhs):
        from .._passes.hir2ir import call
        res = await call(func, (*lhs, *rhs), {})
        assert isinstance(res, Var)
        res_ty = res.get_type()
        if tuple_mode:
            if not isinstance(res_ty, TupleTy):
                raise TileTypeError(f"{op_name} function returns a value of type"
                                    f" {res_ty}, but a tuple was expected", res.loc)
            if len(res_ty.value_types) != len(xs):
                raise TileTypeError(f"{op_name} function must return a tuple of {len(xs)} values"
                                    f" to match the number of inputs, but a tuple of length"
                                    f" {len(res_ty.value_types)} was found.")
            res_tupval = res.get_aggregate()
            assert isinstance(res_tupval, TupleValue)
            results = res_tupval.items
        else:
            results = (res,)

        cast_results = []
        for i, (xi, r) in enumerate(zip(xs, results, strict=True)):
            r_ty = r.get_type()
            extra_ctx = f" at position #{i}" if tuple_mode else ""
            if not isinstance(r_ty, TileTy):
                raise TileTypeError(f"{op_name} function returned"
                                    f" a value of non-tile type {r_ty}{extra_ctx}")
            if r_ty.ndim > 0:
                raise TileTypeError(f"{op_name} function returned"
                                    f" a tile of non-scalar shape {r_ty.shape}{extra_ctx}")
            error_ctx = f"{op_name} function returned a tile of unexpected dtype{extra_ctx}"
            cast_results.append(implicit_cast(r, xi.get_type().dtype, error_ctx))

        return tuple(cast_results)

    return body


@impl(ct.reduce)
async def reduce_impl(x: Var, axis: Var, func: Var, identity: Var, keepdims: Var) -> Var:
    x_ty = require_tile_or_tile_tuple_type(x)

    # Decide if we have a tuple and unpack the items of `x`
    tuple_mode = isinstance(x_ty, TupleTy)
    if tuple_mode:
        tup_val = x.get_aggregate()
        assert isinstance(tup_val, TupleValue)
        xs = tup_val.items
    else:
        xs = (x,)

    # Parse axis & func
    axis = require_constant_int(axis)
    require_callable_type(func)

    # Parse the identity
    if tuple_mode:
        id_values = require_constant_scalar_tuple(identity)
        if len(id_values) != len(xs):
            raise TileTypeError(f"Number of identity values ({len(id_values)}) must match"
                                f" the number of input tiles ({len(xs)})")
    else:
        id_values = (require_constant_scalar(identity),)

    # Parse keepdims
    keepdims = require_constant_bool(keepdims)

    body = _make_reduce_scan_body(func, tuple_mode, xs, "Reduction")
    reduced_tiles = await reduce(xs, id_values, axis, keepdims, body)
    if tuple_mode:
        return build_tuple(reduced_tiles)
    else:
        [ret] = reduced_tiles
        return ret


def _get_reduction_shape(shape: Tuple[int, ...],
                         normalized_axis: Tuple[int, ...],
                         keepdims: bool) -> Tuple[int, ...]:
    ret = []
    for i, size in enumerate(shape):
        if i in normalized_axis:
            if keepdims:
                ret.append(1)
        else:
            ret.append(size)
    return tuple(ret)


async def reduce_simple(fn: str, x: Var, axis: int | None | tuple[int, ...], keepdims: bool,
                        rounding_mode: Optional[RoundingMode] = None,
                        flush_to_zero: bool = False, propagate_nan: bool = False) -> Var:
    x_type = require_tile_type(x)
    if not datatype.is_arithmetic(x_type.dtype):
        raise TileTypeError(f"Non-arithmetic dtype {x_type.dtype} is unsupported for reduction")

    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)

    if datatype.is_boolean(x_type.dtype):
        x = astype(x, datatype.default_int_type)

    match fn:
        case "add": id_val = 0
        case "mul": id_val = 1
        case "min": id_val = _get_min_max(x_type.dtype)[1]
        case "max": id_val = _get_min_max(x_type.dtype)[0]
        case _: assert False

    async def body(lhs: tuple[Var], rhs: tuple[Var]) -> tuple[Var]:
        [lhs], [rhs] = lhs, rhs
        ret = binary_arithmetic_tensorlike(fn, lhs, rhs,
                                           rounding_mode=rounding_mode, flush_to_zero=flush_to_zero,
                                           propagate_nan=propagate_nan)
        return (ret,)

    [ret] = await reduce((x,), (id_val,), axis, keepdims, body)
    return ret


Limits = Tuple[float, float] | Tuple[int, int]


def _get_min_max(dtype: datatype.DType) -> Limits:
    use_float = datatype.is_float(dtype)
    if use_float:
        if dtype in [datatype.float16, datatype.bfloat16, datatype.float32, datatype.float64]:
            return -float("inf"), float("inf")
        else:
            raise NotImplementedError(f"Unsupported float dtype: {dtype}")
    elif datatype.is_signed(dtype):
        return -(1 << (dtype.bitwidth-1)), (1 << (dtype.bitwidth-1)) - 1
    else:
        return 0, (1 << dtype.bitwidth) - 1


def _parse_reduce_axis(axis: Var) -> Optional[tuple[int, ...]]:
    if isinstance(axis.get_type(), TupleTy):
        axis = require_constant_int_tuple(axis)
    else:
        axis = require_optional_constant_int(axis)
        if axis is not None:
            axis = (axis, )
    return axis


@impl(ct.sum, fixed_args=["add"])
@impl(ct.prod, fixed_args=["mul"])
async def reduce_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, keepdims: Var, rounding_mode: Var,
                                      flush_to_zero: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return await reduce_simple(fn, x, axis, keepdims,
                               rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(ct.max, fixed_args=["max"])
@impl(ct.min, fixed_args=["min"])
async def reduce_impl_with_ftz(fn: str, x: Var, axis: Var, keepdims: Var,
                               flush_to_zero: Var, propagate_nan: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    flush_to_zero = require_constant_bool(flush_to_zero)
    propagate_nan = require_constant_bool(propagate_nan)
    return await reduce_simple(fn, x, axis, keepdims, flush_to_zero=flush_to_zero,
                               propagate_nan=propagate_nan)


async def argmax_argmin(fn: str, x: Var, axis: Optional[int], keepdims: bool,
                        propagate_nan: bool = False) -> Var:
    require_tile_type(x)
    final_shape = None
    if axis is None:
        if keepdims:
            final_shape = (1,) * x.get_type().ndim
            keepdims = False
        x = reshape(x, (-1,))
        axis = 0
    else:
        axis = normalize_axis(axis, x.get_type().ndim)

    if datatype.is_boolean(x.get_type().dtype):
        x = astype(x, datatype.default_int_type)

    x_type = x.get_type()
    indices = arange(x_type.shape[axis], datatype.default_int_type)
    indices = reshape(indices, tuple(-1 if i == axis else 1 for i in range(x_type.ndim)))

    match fn:
        case "argmin":
            id_val = _get_min_max(x_type.dtype)[1]
            cmp = "lt"
        case "argmax":
            id_val = _get_min_max(x_type.dtype)[0]
            cmp = "gt"
        case _: assert False

    is_float_dtype = datatype.is_float(x_type.dtype)

    async def body(lhs: tuple[Var, Var], rhs: tuple[Var, Var]) -> tuple[Var, Var]:
        lhs_val, lhs_idx = lhs
        rhs_val, rhs_idx = rhs
        lhs_win = compare_tensorlike_raw(cmp, lhs_val, rhs_val)
        val_equal = compare_tensorlike_raw("eq", lhs_val, rhs_val)
        if is_float_dtype:
            lhs_is_nan = compare_tensorlike_raw("ne", lhs_val, lhs_val)
            rhs_is_nan = compare_tensorlike_raw("ne", rhs_val, rhs_val)
            if propagate_nan:
                # Mirror min/max's propagate_nan=True semantics by
                # treating NaN as the best possible value.
                rhs_not_nan = compare_tensorlike_raw("eq", rhs_val, rhs_val)
                lhs_nan_rhs_finite = binary_bitwise_tensorlike_raw("and_", lhs_is_nan, rhs_not_nan)
                lhs_win = binary_bitwise_tensorlike_raw("or_", lhs_win, lhs_nan_rhs_finite)
            else:
                # Mirror min/max's propagate_nan=False semantics by
                # treating NaN as the worst possible value.
                lhs_not_nan = compare_tensorlike_raw("eq", lhs_val, lhs_val)
                lhs_finite_rhs_nan = binary_bitwise_tensorlike_raw("and_", lhs_not_nan, rhs_is_nan)
                lhs_win = binary_bitwise_tensorlike_raw("or_", lhs_win, lhs_finite_rhs_nan)
            # two NaNs count as "equal" so the index tiebreak (smallest index) decides.
            both_nan = binary_bitwise_tensorlike_raw("and_", lhs_is_nan, rhs_is_nan)
            val_equal = binary_bitwise_tensorlike_raw("or_", val_equal, both_nan)
        index_lt = compare_tensorlike_raw("lt", lhs_idx, rhs_idx)
        val_equal_and_index_lt = binary_bitwise_tensorlike_raw("and_", val_equal, index_lt)
        cond = binary_bitwise_tensorlike_raw("or_", lhs_win, val_equal_and_index_lt)
        res = where_raw(cond, lhs_val, rhs_val)
        idx = where_raw(cond, lhs_idx, rhs_idx)
        return res, idx

    [_, ret] = await reduce((x, indices), (id_val, 0), axis, keepdims, body)

    if final_shape is not None:
        ret = reshape(ret, final_shape)

    return ret


@impl(ct.argmax, fixed_args=["argmax"])
@impl(ct.argmin, fixed_args=["argmin"])
async def argmax_argmin_impl(fn: str, x: Var, axis: Var, keepdims: Var, propagate_nan: Var) -> Var:
    axis = require_optional_constant_int(axis)
    keepdims = require_constant_bool(keepdims)
    propagate_nan = require_constant_bool(propagate_nan)
    return await argmax_argmin(fn, x, axis, keepdims, propagate_nan=propagate_nan)


@dataclass(eq=False)
class TileScan(Operation, opcode="tile_scan"):
    axis: int = attribute()
    reverse: bool = attribute()
    identities: tuple[bool | int | float, ...] = attribute()
    xs: tuple[Var, ...] = operand()
    body: Block = nested_block()

    @property
    def lhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[:len(self.xs)]

    @property
    def rhs(self):
        params = self.body.params
        assert len(params) == len(self.xs) * 2
        return params[len(self.xs):]

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["do"]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[bc.Value, ...]:
        xs = tuple(ctx.get_value(x) for x in self.xs)
        res_typeids = tuple(ctx.typeid_of(v) for v in self.result_vars)

        identities = []
        param_type_ids = []
        for id_val, x in zip(self.identities, self.xs, strict=True):
            x_dtype = get_dtype(x.get_type())
            x_dtype_id = dtype_typeid(ctx.type_table, x_dtype)
            if datatype.is_float(x_dtype):
                x_dtype_bc = datatype.dtype_simple_bytecode_type(x_dtype)
                attr = bc.Float(float(id_val), x_dtype_bc, ctx.type_table)
            elif datatype.is_boolean(x_dtype):
                attr = bc.Bool(bool(id_val))
            else:
                assert datatype.is_integral(x_dtype)
                attr = bc.Integer(x_dtype_id, x_dtype.bitwidth, int(id_val))
            identities.append(attr)

            x_tile_typeid = ctx.type_table.tile(x_dtype_id, ())
            param_type_ids.append(x_tile_typeid)
            param_type_ids.append(x_tile_typeid)

        nested_builder = bc.encode_ScanOp(
            ctx.builder,
            result_types=res_typeids,
            operands=xs,
            dim=self.axis,
            reverse=self.reverse,
            identities=identities,
        )

        with nested_builder.new_block(param_type_ids) as block_args:
            for var, value in zip(self.body.params, block_args, strict=True):
                ctx.set_value(var, value)
            generate_bytecode_for_block(ctx, self.body)

        return nested_builder.done()


async def raw_scan(xs: tuple[Var, ...], identities: tuple[bool | int | float, ...], axis: int,
                   reverse: bool, body: Callable) -> tuple[Var, ...]:
    input_shape = require_tile_type(xs[0]).shape
    assert 0 <= axis < len(input_shape)
    result_types = tuple(TileTy(x.get_type().dtype, input_shape) for x in xs)
    assert len(xs) == len(identities)
    body_block = await _get_reduce_scan_body_block(xs, body, op_name="scan")
    return add_operation_variadic(TileScan, result_types, xs=xs, identities=identities, axis=axis,
                                  reverse=reverse, body=body_block)


async def scan_simple(fn: str, x: Var, axis: int, reverse: bool,
                      rounding_mode: Optional[RoundingMode] = None,
                      flush_to_zero: bool = False) -> Var:
    x_type = require_tile_type(x)
    if not datatype.is_arithmetic(x_type.dtype):
        raise TileTypeError(f"Non-arithmetic dtype {x_type.dtype} is unsupported for prefix scans")
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)

    if datatype.is_boolean(x_type.dtype):
        x = astype(x, datatype.default_int_type)
        x_type = require_tile_type(x)

    match fn:
        case "add":
            id_val = 0
        case "mul":
            id_val = 1
        case _:
            assert False

    x_shape = x_type.shape
    axis = normalize_axis(axis, len(x_shape))
    x_dtype = x_type.dtype
    x = promote_and_broadcast_to(x, TileTy(x_dtype, x_shape))

    async def body(lhs: tuple[Var], rhs: tuple[Var]) -> tuple[Var]:
        [lhs], [rhs] = lhs, rhs
        ret = binary_arithmetic_tensorlike(fn, lhs, rhs,
                                           rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)
        return (ret,)

    [ret] = await raw_scan((x,), (id_val,), axis, reverse, body)
    return ret


@impl(ct.scan)
async def scan_impl(x: Var, axis: Var, func: Var, identity: Var, reverse: Var) -> Var:
    x_ty = require_tile_or_tile_tuple_type(x)

    tuple_mode = isinstance(x_ty, TupleTy)
    if tuple_mode:
        tup_val = x.get_aggregate()
        assert isinstance(tup_val, TupleValue)
        xs = tup_val.items
    else:
        xs = (x,)

    axis = require_constant_int(axis)
    require_callable_type(func)
    reverse = require_constant_bool(reverse)

    body = _make_reduce_scan_body(func, tuple_mode, xs, "Scan")

    if len(xs) == 0:
        raise TileTypeError("Need at least one input value to scan")

    common_input_shape = ()

    x_types = tuple(require_tile_type(x) for x in xs)
    for x_ty in x_types:
        try:
            common_input_shape = broadcast_shapes2(common_input_shape, x_ty.shape)
        except BroadcastError:
            all_shapes = ", ".join(str(ty.shape) for ty in x_types)
            raise TileTypeError(f"Input shapes {all_shapes}"
                                f" are not broadcastable to a common shape")
    xs = tuple(broadcast_to(x, common_input_shape) for x in xs)

    # Normalize axis (e.g. -1 -> last axis) before raw_scan
    axis = normalize_axis(axis, len(common_input_shape))

    if tuple_mode:
        id_values = require_constant_scalar_tuple(identity)
        if len(id_values) != len(xs):
            raise TileTypeError(f"Number of identity values ({len(id_values)}) must match"
                                f" the number of input tiles ({len(xs)})")
    else:
        id_values = (require_constant_scalar(identity),)

    scaned_tiles = await raw_scan(xs, id_values, axis, reverse, body)
    if tuple_mode:
        return build_tuple(scaned_tiles)
    else:
        [ret] = scaned_tiles
        return ret


@impl(ct.cumsum, fixed_args=["add"])
@impl(ct.cumprod, fixed_args=["mul"])
async def scan_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, reverse: Var,
                                    rounding_mode: Var, flush_to_zero: Var) -> Var:
    axis = require_constant_int(axis)
    reverse = require_constant_bool(reverse)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return await scan_simple(fn, x, axis, reverse,
                             rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def expand_dims(x: Var, axis: int) -> Var:
    x_ty = require_tile_type(x)
    axis = normalize_axis(axis, x_ty.ndim + 1)
    old_shape = x_ty.shape
    new_shape = (*old_shape[:axis], 1, *old_shape[axis:])
    res_type = TileTy(x_ty.dtype, new_shape)
    return add_operation(TileReshape, res_type, x=x)


@impl(ct.expand_dims)
def expand_dims_impl(x: Var, axis: Var) -> Var:
    axis = require_constant_int(axis)
    return expand_dims(x, axis)


@dataclass(eq=False)
class TileCat(Operation, opcode="tile_cat"):
    axis: int = attribute()
    x: Var = operand()
    y: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return_type_id = ctx.typeid_of(self.result_var)
        x_value, y_value = ctx.get_value(self.x), ctx.get_value(self.y)
        return bc.encode_CatOp(ctx.builder, return_type_id, x_value, y_value, self.axis)


def cat(tiles: tuple[Var, ...], axis: int) -> Var:
    if len(tiles) == 0:
        raise TileTypeError("cat() received an empty tuple")
    if len(tiles) == 1:
        return tiles[0]
    if len(tiles) > 2:
        raise TileTypeError(f"cat() supports at most 2 tiles, got {len(tiles)}")

    x_tile, y_tile = tiles

    if not isinstance(first_tile_ty := tiles[0].get_type(), TileTy):
        raise TileTypeError(f"Expected tuple of Tile, got a {first_tile_ty}")

    dtype = first_tile_ty.dtype
    rank = first_tile_ty.ndim
    shape_value = list(first_tile_ty.shape)
    axis = normalize_axis(axis, rank)
    for tile_ty in (t.get_type() for t in tiles[1:]):
        if not isinstance(tile_ty, TileTy):
            raise TileTypeError(f"Expected tuple of Tile, got a {tile_ty}")
        if tile_ty.ndim != rank:
            raise TileTypeError(f"Expected tiles to have the same rank: {rank} != {tile_ty.ndim}")
        if tile_ty.dtype != dtype:
            raise TileTypeError(f"Expected tiles to have the same dtype: {dtype} != {tile_ty.dtype}")  # noqa: E501
        for i, (x, y) in enumerate(zip(shape_value, tile_ty.shape, strict=True)):
            if i != axis and x != y:
                raise TileTypeError("Expected tiles to have the same shape "
                                    "for non axis dimensions, "
                                    f"got {tuple(shape_value)} and {tile_ty.shape}")
        shape_value[axis] += tile_ty.shape[axis]

    if not all(_is_power_of_2(x) for x in shape_value):
        raise TileTypeError(f"Result tile shape must be power of 2, got: {shape_value}")

    res_ty = TileTy(dtype, shape_value)
    return add_operation(TileCat, res_ty, x=x_tile, y=y_tile, axis=axis)


def _is_power_of_2(x: int):
    if x <= 0:
        return False
    return x & (x - 1) == 0


@impl(ct.cat)
def cat_impl(tiles: Var, axis: Var) -> Var:
    require_tuple_type(tiles)
    const_axis = require_constant_int(axis)
    return cat(tiles.get_aggregate().items, const_axis)


@impl(ct.where)
def tile_where_function_impl(cond, x, y):
    return where(ensure_tile(cond), ensure_tile(x), ensure_tile(y))


@impl(ct.printf)
def printf_impl(format: Var, args: Tuple[Var, ...]) -> None:
    format_str = require_constant_str(format)
    arg_types = tuple(require_tile_type(x) for x in args)
    parsed_format = PrintfValidator.parse_format(format_str, arg_types)
    add_operation_variadic(TilePrintf, (TokenTy(),), format=parsed_format, args=args)


@impl(ct.print)
async def _tile_print_impl(args: tuple[Var, ...], sep: Var, end: Var) -> None:
    return await print_impl(args, sep, end)


@dataclass(eq=False)
class TileAssert(Operation, opcode="assert", memory_effect=MemoryEffect.STORE):
    message: str = attribute()
    cond: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        bc.encode_AssertOp(ctx.builder, ctx.get_value(self.cond), self.message)
        return []


@impl(ct.assert_)
def assert_impl(cond: Var, message: Var) -> None:
    ty = require_tile_type(cond)
    if get_dtype(ty) != datatype.bool_:
        raise TileTypeError(f"Type of condition must be bool, got {ty}")
    msg_str = require_optional_constant_str(message)
    msg_str = "" if msg_str is None else msg_str
    add_operation_variadic(TileAssert, (), cond=cond, message=msg_str)


@impl(ct.astype)
def astype_impl(x: Var, dtype: Var) -> Var:
    require_tile_type(x)
    dtype = require_dtype_spec(dtype)
    return astype(x, dtype)


@dataclass(eq=False)
class TileBitCast(Operation, opcode="tile_bitcast"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return ctx.bitcast(value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def bitcast(x: Var, dtype: DType) -> Var:
    tile_ty = require_tile_type(x)
    x_dtype = tile_ty.dtype
    if x_dtype == datatype.bool_ or dtype == datatype.bool_:
        raise TileTypeError(f"Cannot bitcast from {x_dtype} to {dtype}: "
                            f"bitcast to or from bool is not supported")

    if x_dtype.bitwidth != dtype.bitwidth:
        raise TileTypeError(f"Cannot bitcast from {x_dtype} to {dtype}: "
                            f"bit width is different ({x_dtype.bitwidth} vs. {dtype.bitwidth})")

    if x_dtype == dtype:
        return x

    res_ty = TileTy(dtype, tile_ty.shape)
    return add_operation(TileBitCast, res_ty, x=x)


@impl(ct.bitcast)
def bitcast_impl(x: Var, dtype: Var) -> Var:
    dtype_val = require_dtype_spec(dtype)
    return bitcast(x, dtype_val)


@dataclass(eq=False)
class TilePack(Operation, opcode="tile_pack"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_type_id = ctx.typeid_of(self.result_var)
        x_value = ctx.get_value(self.x)
        return bc.encode_PackOp(ctx.builder, res_type_id, x_value)


def pack(x: Var) -> Var:
    tile_ty = require_tile_type(x)
    assert tile_ty.ndim == 1
    assert tile_ty.dtype.bitwidth != 8
    old_dim = tile_ty.shape[0]
    new_dim, rem = divmod(old_dim * tile_ty.dtype.bitwidth, 8)
    if rem != 0:
        raise TileTypeError(f"Cannot pack tile {tile_ty}: "
                            f"total bits ({old_dim} * {tile_ty.dtype.bitwidth}) "
                            f"not divisible by 8")
    res_ty = TileTy(datatype.uint8, (new_dim,))
    return add_operation(TilePack, res_ty, x=x)


@impl(ct.pack_to_bytes, min_version=BytecodeVersion.V_13_3)
def pack_to_bytes_impl(x: Var):
    tile_ty = require_tile_type(x)
    x_dtype = tile_ty.dtype
    x = reshape(x, (-1,))
    if x_dtype == datatype.bool_:
        raise TileTypeError(f"pack_to_bytes from a {x_dtype} tile is not supported")

    if x_dtype.bitwidth == 8:
        return bitcast(x, datatype.uint8)
    return pack(x)


@dataclass(eq=False)
class TileUnpack(Operation, opcode="tile_unpack"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_type_id = ctx.typeid_of(self.result_var)
        x_value = ctx.get_value(self.x)
        return bc.encode_UnpackOp(ctx.builder, res_type_id, x_value)


def unpack(x: Var, dtype: DType) -> Var:
    tile_ty = require_tile_type(x)
    assert tile_ty.ndim == 1
    assert tile_ty.dtype == datatype.uint8
    assert dtype.bitwidth != 8
    old_dim = tile_ty.shape[0]
    new_dim, rem = divmod(old_dim * 8, dtype.bitwidth)
    if rem != 0:
        raise TileTypeError(
            f"Cannot unpack tile {tile_ty} to {dtype}: "
            f"total bits ({old_dim} * 8) not divisible by {dtype.bitwidth}")
    res_ty = TileTy(dtype, (new_dim,))
    return add_operation(TileUnpack, res_ty, x=x)


@impl(ct.unpack_from_bytes, min_version=BytecodeVersion.V_13_3)
def unpack_from_bytes_impl(x: Var, dtype: Var):
    tile_ty = require_tile_type(x)
    x_dtype = tile_ty.dtype
    dtype = require_dtype_spec(dtype)
    if tile_ty.ndim != 1:
        raise TileTypeError(
            f"unpack_from_bytes requires a 1D tile, "
            f"got {tile_ty.ndim}D tile with shape {tile_ty.shape}")
    if x_dtype != datatype.uint8:
        raise TileTypeError(
            f"unpack_from_bytes requires uint8 tile, got {x_dtype} tile")
    if dtype == datatype.bool_:
        raise TileTypeError(f"unpack_from_bytes to a {dtype} tile is not supported")

    if dtype.bitwidth == 8:
        return bitcast(x, dtype)
    return unpack(x, dtype)


@dataclass(eq=False)
class TileArange(Operation, opcode="tile_arange"):
    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_type = ctx.typeid_of(self.result_var)
        return bc.encode_IotaOp(ctx.builder, res_type)


def arange(size: int, dtype: DType) -> Var:
    if datatype.is_integral(dtype):
        res_ty = TileTy(dtype, (size,))
    else:
        res_ty = TileTy(datatype.default_int_type, (size,))
    res = add_operation(TileArange, res_ty)
    return astype(res, dtype)


@impl(ct.arange)
def arange_impl(size: Var, dtype: Var, start: Var, step: Var) -> Var:
    size_val = require_constant_int(size)
    dtype_val = require_dtype_spec(dtype)
    if not _is_power_of_2(size_val):
        raise TileTypeError(f"Result tile shape must be power of 2, got {size_val}")
    result = arange(size_val, dtype_val)
    if not (step.is_constant() and step.get_constant() == 1):
        result = binary_arithmetic_tensorlike("mul", result, astype(step, dtype_val))
    if not (start.is_constant() and start.get_constant() == 0):
        result = binary_arithmetic_tensorlike("add", result, astype(start, dtype_val))
    return result


@impl(ct.reshape)
def reshape_impl(x: Var, shape: Var) -> Var:
    new_shape = require_constant_int_tuple(shape)
    return reshape(ensure_tile(x), new_shape)


@impl(ct.broadcast_to)
def broadcast_to_impl(x: Var, shape: Var) -> Var:
    shape = require_constant_shape(shape)
    return broadcast_to(ensure_tile(x), shape)


@dataclass(eq=False)
class TilePermute(Operation, opcode="tile_permute"):
    axes: tuple[int, ...] = attribute()
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        ret_ty_id = ctx.typeid_of(self.result_var)
        x_value = ctx.get_value(self.x)
        return bc.encode_PermuteOp(ctx.builder, ret_ty_id, x_value, self.axes)


def permute(x: Var, axes: Sequence[int]) -> Var:
    ty = require_tile_type(x)
    axes = tuple(normalize_axis(ax, ty.ndim) for ax in axes)
    shape = tuple(ty.shape[i] for i in axes)
    result_ty = TileTy(ty.dtype, shape)
    return add_operation(TilePermute, result_ty, x=x, axes=axes)


@impl(ct.permute)
def permute_impl(x: Var, axes: Var) -> Var:
    ty = require_tile_type(x)
    axes_value = require_constant_int_tuple(axes)
    if len(axes_value) != ty.ndim:
        raise TileTypeError(f"Num axes must match input's rank: {len(axes_value)} vs {ty.ndim}")
    seen_axes = set()
    for i, axis in enumerate(axes_value):
        if axis in seen_axes:
            raise TileTypeError(f"Repeated axis #{i}: {axis}")
        seen_axes.add(axis)
    return permute(x, axes_value)


def transpose(x: Var, a0: int, a1: int) -> Var:
    ty = require_tile_type(x)
    axes = list(range(ty.ndim))
    axes[a0], axes[a1] = axes[a1], axes[a0]
    return permute(x, axes)


@impl(ct.transpose)
def transpose_impl(x: Var, axis0: Var, axis1: Var) -> Var:
    ty = require_tile_type(x)
    if ty.ndim < 2:
        raise TileTypeError("Cannot transpose a tile with fewer than 2 dimensions")
    a0 = require_optional_constant_int(axis0)
    a1 = require_optional_constant_int(axis1)

    if (a0 is not None) and (a1 is not None):
        a0 = normalize_axis(a0, ty.ndim)
        a1 = normalize_axis(a1, ty.ndim)
    elif (a0 is None) and (a1 is None):
        if ty.ndim != 2:
            raise TileTypeError("`axes` must be specified for tile with more than 2 dimensions")
        a0 = ty.ndim - 1
        a1 = ty.ndim - 2
    else:
        raise TileTypeError(f"transpose axes must either both be specified or both be None, "
                            f"got axis0={a0}, axis1={a1}")
    return transpose(x, a0, a1)


@dataclass(eq=False)
class TileExtract(Operation, opcode="tile_extract"):
    shape: tuple[int, ...] = attribute()
    x: Var = operand()
    index: tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        index = tuple(ctx.get_value(idx) for idx in self.index)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ExtractOp(ctx.builder, res_type_id, x_value, index)


def extract(x: Var, index: tuple[Var, ...], shape: Sequence[int]) -> Var:
    dtype = get_dtype(x.get_type())
    res_ty = TileTy(dtype, shape)
    return add_operation(TileExtract, res_ty, x=x, index=index, shape=tuple(shape))


def _resolve_subtile_index(x_ty, shape, index, *, sub_noun):
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if x_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the tile rank {x_ty.ndim}")

    for i, (s1, s2, idx_var) in enumerate(zip(x_ty.shape, shape, index_items, strict=True)):
        if s2 == 0:
            raise TileTypeError(f"Zero shape at dimension #{i}: {shape}")
        if s1 % s2 != 0:
            raise TileTypeError(f"Input shape {x_ty.shape} is not divisible by"
                                f" {sub_noun} shape {shape} at dimension #{i}")
        n_tiles = s1 // s2
        if idx_var.is_constant():
            idx_val = idx_var.get_constant()
            if not (0 <= idx_val < n_tiles):
                raise TileTypeError(
                    f"Index {idx_val} out of bounds at dimension #{i}: "
                    f"valid range is [0, {n_tiles}) in tile space "
                    f"(input shape {x_ty.shape}, {sub_noun} shape {shape})"
                )
    return index_items


@impl(ct.extract)
def extract_impl(x: Var, index: Var, shape: Var) -> Var:
    x_ty = require_tile_type(x)
    shape = require_constant_shape(shape, expected_rank=x_ty.ndim, allow_single_int=True,
                                   allow_0d_shape=True)
    orig_shape = shape
    if len(shape) == 0:
        shape = (1,) * x_ty.ndim

    index_items = _resolve_subtile_index(x_ty, shape, index, sub_noun="result")
    result = extract(x, index_items, shape)
    return reshape(result, orig_shape)


@dataclass(eq=False)
class TileInsert(Operation, opcode="tile_insert"):
    x: Var = operand()
    value: Var = operand()
    index: tuple[Var, ...] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        value = ctx.get_value(self.value)
        index = tuple(ctx.get_value(idx) for idx in self.index)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_InsertOp(ctx.builder, res_type_id, value, x_value, index)


def insert(x: Var, index: tuple[Var, ...], value: Var) -> Var:
    return add_operation(TileInsert, x.get_type(), x=x, value=value, index=index)


@impl(ct.insert, min_version=BytecodeVersion.V_13_4)
def insert_impl(x: Var, index: Var, value: Var) -> Var:
    x_ty = require_tile_type(x)
    value_ty = require_tile_type(value)

    if x_ty.dtype != value_ty.dtype:
        raise TileTypeError(
            f"Cannot insert a tile of dtype {value_ty.dtype} into a tile"
            f" of dtype {x_ty.dtype}"
        )

    if value_ty.ndim == 0:
        shape = (1,) * x_ty.ndim
        value = reshape(value, shape)
    else:
        shape = value_ty.shape
        if value_ty.ndim != x_ty.ndim:
            raise TileTypeError(f"Value rank {value_ty.ndim}"
                                f" does not match the tile rank {x_ty.ndim}")

    index_items = _resolve_subtile_index(x_ty, shape, index, sub_noun="value")
    return insert(x, index_items, value)


@impl(ct.Tile.item)
def tile_item(self: Var) -> Var:
    return reshape(self, ())


@impl(ct.Array.tiled_view)
def array_tiled_view_impl(self: Var, tile_shape: Var, padding_mode: Var,
                          traversal_steps: Var) -> Var:
    array_ty = require_array_type(self)
    shape_val = require_constant_shape(tile_shape, allow_single_int=True,
                                       expected_rank=array_ty.ndim,
                                       allow_0d_shape=True)
    padding_mode_val = require_constant_enum(padding_mode, PaddingMode)
    if traversal_steps.is_constant() and traversal_steps.get_constant() is None:
        broadcasted_shape_val = (1,) * array_ty.ndim if len(shape_val) == 0 else shape_val
        traversal_steps_val = broadcasted_shape_val
    else:
        cur = Builder.get_current().ir_ctx.tileiras_version
        if cur < BytecodeVersion.V_13_3:
            raise TileUnsupportedFeatureError(
                f"traversal_steps requires tileiras 13.3 or later. "
                f"Current version is {cur.major()}.{cur.minor()}."
            )
        traversal_steps_val = require_constant_shape(traversal_steps, allow_single_int=True,
                                                     expected_rank=array_ty.ndim,
                                                     allow_non_power_of_two=True,
                                                     var_name="traversal_steps")
    view_ty = TiledViewTy(array_ty, shape_val, padding_mode_val, traversal_steps_val)
    return make_aggregate(TiledViewValue(self), view_ty)


@impl(ct.TiledView.num_tiles)
def tiled_view_num_tiles(self: Var, axis: Var) -> Var:
    ty = self.get_type()
    [array] = self.get_aggregate().as_tuple()
    view_shape = num_tiles(array, ty.tile_shape, get_default_order(ty.ndim), ty.traversal_steps)
    axis = require_constant_int(axis)
    axis = normalize_axis(axis, ty.ndim)
    return view_shape[axis]


@impl(ct.TiledView.load)
def tiled_view_load_impl(self: Var, index: Var, check_bounds: Var, latency: Var,
                         allow_tma: Var) -> Var:
    view_ty = require_tiled_view_type(self)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if view_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the tiled view rank {view_ty.ndim}")

    inbounds = _check_bounds_to_inbounds(check_bounds, view_ty.ndim)
    [array] = self.get_aggregate().as_tuple()
    order = get_default_order(view_ty.ndim)
    return _tile_load_impl_inner(array, index_items, view_ty.tile_shape, order,
                                 view_ty.padding_mode, latency, allow_tma,
                                 inbounds=inbounds,
                                 traversal_steps=view_ty.traversal_steps)


@impl(ct.TiledView.store)
def tiled_view_store_impl(self: Var, index: Var, tile: Var, check_bounds: Var, latency: Var,
                          allow_tma: Var):
    view_ty = require_tiled_view_type(self)
    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if view_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the tiled view rank {view_ty.ndim}")

    tile_ty = require_tile_type(tile)
    if not is_shape_broadcastable_to(tile_ty.shape, view_ty.tile_shape):
        raise TileTypeError(f"Tile shape {tile_ty.shape} is not broadcastable"
                            f" to the tiled view's tile shape {view_ty.tile_shape}")

    inbounds = _check_bounds_to_inbounds(check_bounds, view_ty.ndim)
    tile = broadcast_to(tile, view_ty.tile_shape)
    tile = implicit_cast(tile, view_ty.dtype,
                         "Stored tile is incompatible with tiled view's dtype")
    [array] = self.get_aggregate().as_tuple()
    order = get_default_order(view_ty.ndim)
    _tile_store_impl_inner(array, index_items, tile, order, latency, allow_tma,
                           inbounds=inbounds,
                           traversal_steps=view_ty.traversal_steps)


@dataclass(eq=False)
class TileAtomicRedView(Operation, opcode="tile_atomic_red_view", memory_effect=MemoryEffect.STORE):
    mode: AtomicRMWMode = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    view: Var = operand()
    index: tuple[Var, ...] = operand()
    update: Var = operand()
    token: Optional[Var] = operand(default=None)

    VALID_MEMORY_ORDERS = (MemoryOrder.RELAXED,)

    VALID_MEMORY_SCOPES = (MemoryScope.BLOCK, MemoryScope.DEVICE)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return bc.encode_AtomicRedViewTkoOp(
            ctx.builder,
            result_token_type=ctx.type_table.Token,
            view=ctx.get_value(self.view),
            index=ctx.index_tuple(self.index),
            value=ctx.get_value(self.update),
            token=None if self.token is None else ctx.get_value(self.token),
            memory_ordering_semantics=memory_order_to_bytecode[self.memory_order],
            memory_scope=memory_scope_to_bytecode[self.memory_scope],
            mode=self.mode._value_
        )


_TILED_VIEW_ATOMIC_STORE_RMW_STUBS = {
    "add": ct.TiledView.atomic_store_add,
    "min": ct.TiledView.atomic_store_min,
    "max": ct.TiledView.atomic_store_max,
    "and": ct.TiledView.atomic_store_and,
    "or":  ct.TiledView.atomic_store_or,
    "xor": ct.TiledView.atomic_store_xor,
}


@_register_atomic_rmw_impls(_TILED_VIEW_ATOMIC_STORE_RMW_STUBS,
                            min_version=BytecodeVersion.V_13_3)
def tiled_view_atomic_store_rmw_impl(int_mode: Optional[AtomicRMWMode],
                                     uint_mode: Optional[AtomicRMWMode],
                                     float_mode: Optional[AtomicRMWMode],
                                     bitwise: bool,
                                     supported_dtypes: Sequence[DType],
                                     # --- end of fixed args ---
                                     self: Var, index: Var, update: Var):
    view_ty = require_tiled_view_type(self)
    if view_ty.dtype not in supported_dtypes:
        raise TileTypeError(f"Unsupported tiled view dtype: {view_ty.dtype}")

    index_ty = require_index_or_index_tuple_type(index)
    index_items = index.get_aggregate().items if isinstance(index_ty, TupleTy) else (index,)
    if view_ty.ndim != len(index_items):
        raise TileTypeError(f"Index size {len(index_items)}"
                            f" does not match the tiled view rank {view_ty.ndim}")

    update_ty = require_tile_type(update)
    if not is_shape_broadcastable_to(update_ty.shape, view_ty.tile_shape):
        raise TileTypeError(f"Update shape {update_ty.shape} is not broadcastable"
                            f" to the tiled view's tile shape {view_ty.tile_shape}")

    broadcasted_shape = (1,) * view_ty.ndim if len(view_ty.tile_shape) == 0 else view_ty.tile_shape
    update = broadcast_to(update, broadcasted_shape)
    update = _cast_rmw_update_dtype(update, view_ty.dtype, bitwise)
    mode = _select_rmw_mode(int_mode, uint_mode, float_mode, view_ty.dtype)

    memory_order = MemoryOrder.RELAXED
    memory_scope = MemoryScope.DEVICE
    validate_memory_order_and_scope(memory_order, memory_scope, TileAtomicRedView)

    [array] = self.get_aggregate().as_tuple()
    order = get_default_order(view_ty.ndim)
    view = _materialize_tiled_view(array, broadcasted_shape, order, PaddingMode.UNDETERMINED,
                                   view_ty.traversal_steps)
    add_operation(TileAtomicRedView, TokenTy(),
                  mode=mode, memory_order=memory_order, memory_scope=memory_scope,
                  view=view, index=index_items, update=update)


@impl(ct.Slice)
def slice_index_constructor_impl(start: Var, length: Var) -> Var:
    start_ty = require_signed_integer_0d_tile_type(start)
    length_ty = require_signed_integer_0d_tile_type(length)
    res_type = IndexSliceTy(start_ty, length_ty)
    res_loose_type = IndexSliceTy(start.get_loose_type(), length.get_loose_type())
    return make_aggregate(IndexSliceValue(start, length), res_type, res_loose_type)


def _parse_advanced_index(indices: Var, ndim: int) -> tuple[int, tuple[int, ...], tuple[Var, ...]]:
    """Unpack, classify, validate, and build the gather scatter view index.

    Returns (sparse_dim, tile_shape, gs_index).
    """
    require_tuple_type(indices)
    items = list(indices.get_aggregate().items)
    if len(items) != ndim:
        raise TileTypeError(
            f"load_advanced_indexing/store_advanced_indexing index length {len(items)} does not "
            f"match array rank {ndim}")

    sparse_dims: list[int] = []
    tile_shape: list[int] = []
    gs_index: list[Var] = []

    for dim, item in enumerate(items):
        item_ty = item.get_type()
        if isinstance(item_ty, TileTy):
            if item_ty.ndim != 1:
                raise TileTypeError(
                    f"Sparse index at dim {dim} must be a 1D integer tile, "
                    f"got {item_ty.ndim}D")
            if not is_integral(item_ty.dtype):
                raise TileTypeError(
                    f"Sparse index at dim {dim} must be an integer tile, "
                    f"got dtype {item_ty.dtype}")
            sparse_dims.append(dim)
            tile_shape.append(item_ty.shape[0])
            gs_index.append(item)
        elif isinstance(item_ty, IndexSliceTy):
            length_var = item.get_aggregate().length
            if not length_var.is_constant():
                raise TileTypeError(
                    f"ct.Slice length at dim {dim} must be a compile-time constant "
                    f"in load_advanced_indexing/store_advanced_indexing")
            length_val = length_var.get_constant()
            if not isinstance(length_val, int) or length_val <= 0:
                raise TileTypeError(
                    f"ct.Slice length at dim {dim} must be a positive integer, got {length_val}")
            tile_shape.append(length_val)
            gs_index.append(item.get_aggregate().start)
        else:
            raise TileTypeError(
                f"load_advanced_indexing/store_advanced_indexing index at dim {dim} must be a "
                f"1D integer Tile (sparse dim) or ct.Slice(start, length) "
                f"(dense dim), got type {item_ty}")

    if len(sparse_dims) == 0:
        raise TileTypeError(
            "load_advanced_indexing/store_advanced_indexing: exactly one index must be a 1D "
            "integer Tile (the sparse dim); none found")
    if len(sparse_dims) > 1:
        raise TileTypeError(
            f"load_advanced_indexing/store_advanced_indexing: exactly one index must be a 1D "
            f"integer Tile (the sparse dim); found {len(sparse_dims)} at "
            f"dims {sparse_dims}")

    for dim, n in enumerate(tile_shape):
        if not _is_power_of_2(n):
            raise TileTypeError(
                f"Index at dim {dim} has size {n}; must be a power of two")

    return sparse_dims[0], tuple(tile_shape), tuple(gs_index)


@impl(ct.load_advanced_indexing, min_version=BytecodeVersion.V_13_3)
def load_advanced_impl(array: Var, indices: Var, padding_mode: Var,
                       latency: Var, allow_tma: Var) -> Var:
    array_ty = require_array_type(array)
    if array_ty.ndim < 2:
        raise TileTypeError(
            "load_advanced_indexing requires a 2D or higher-rank array; "
            "use ct.gather() for 1D arrays")
    sparse_dim, tile_shape, gs_index = _parse_advanced_index(indices, array_ty.ndim)
    padding_mode_val = require_constant_enum(padding_mode, PaddingMode)
    latency_val = require_optional_constant_int(latency)
    allow_tma_val = require_optional_constant_bool(allow_tma)
    _check_load_store_hints(latency_val, allow_tma_val)

    view = make_gather_scatter_view(array, tile_shape, sparse_dim, padding_mode_val)
    result, _token = add_operation_variadic(TileLoad,
                                            (TileTy(array_ty.dtype, tile_shape), TokenTy()),
                                            view=view, index=gs_index,
                                            latency=latency_val, allow_tma=allow_tma_val)
    return result


@impl(ct.store_advanced_indexing, min_version=BytecodeVersion.V_13_3)
def store_advanced_impl(array: Var, indices: Var, tile: Var,
                        latency: Var, allow_tma: Var):
    array_ty = require_array_type(array)
    if array_ty.ndim < 2:
        raise TileTypeError(
            "store_advanced_indexing requires a 2D or higher-rank array; "
            "use ct.scatter() for 1D arrays")
    sparse_dim, tile_shape, gs_index = _parse_advanced_index(indices, array_ty.ndim)
    tile_ty = require_tile_type(tile)
    if tile_ty.shape != tile_shape:
        raise TileTypeError(
            f"Tile shape {tile_ty.shape} does not match the shape implied by "
            f"indices {tile_shape}")
    tile = implicit_cast(tile, array_ty.dtype,
                         "Stored tile dtype is incompatible with array dtype")
    latency_val = require_optional_constant_int(latency)
    allow_tma_val = require_optional_constant_bool(allow_tma)
    _check_load_store_hints(latency_val, allow_tma_val)

    view = make_gather_scatter_view(array, tile_shape, sparse_dim, PaddingMode.UNDETERMINED)
    add_operation(TileStore, TokenTy(),
                  view=view, index=gs_index, tile=tile,
                  latency=latency_val, allow_tma=allow_tma_val)


@tile_impl_registry.unflatten_aggregate_impl(ArrayTy)
def _unflatten_aggregate_array_impl(val: ArrayValue, ty: ArrayTy, result_var: Var):
    assert isinstance(val, ArrayValue)
    base_ptr = val.base_ptr
    all_shape = []
    dynamic_shape = []
    for x, s in zip(val.shape, ty.shape, strict=True):
        if s is None:
            x = assume_bounded(x, 0, None)
            dynamic_shape.append(x)
        else:
            x = strictly_typed_const(s, TileTy(ty.index_dtype))
        all_shape.append(x)

    all_strides = []
    dynamic_strides = []
    for x, s in zip(val.strides, ty.strides, strict=True):
        if s is None:
            x = assume_bounded(x, 0, None)
            dynamic_strides.append(x)
        all_strides.append(x)

    operands = dict(base_ptr=base_ptr, shape=tuple(dynamic_shape),
                    dynamic_strides=tuple(dynamic_strides))
    ret = Builder.get_current().add_operation(MakeTensorView, ty, operands, result_var)
    ret.set_aggregate(ArrayValue(base_ptr, tuple(all_shape), tuple(all_strides)))
    return ret


@tile_impl_registry.unflatten_aggregate_impl(ListTy)
def _unflatten_aggregate_list_impl(val: ListValue, ty: ListTy, result_var: Var):
    assert isinstance(val, ListValue)
    operands = dict(base_ptr=val.base_ptr, length=val.length)
    ret = Builder.get_current().add_operation(MakeListView, ty, operands, result_var)
    ret.set_aggregate(val)
    return ret


tile_impl_registry.update(array_impl_registry)
