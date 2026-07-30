# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
from contextlib import contextmanager
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING
import warnings

from cuda.tile import _datatype as datatype
from cuda.tile._bytecode.attribute import make_load_store_hints
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._datatype import get_signedness, is_pointer_dtype, PointerInfo, \
    dtype_simple_bytecode_type
from cuda.tile import DType
import cuda.tile._bytecode as bc
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import (
    TileInternalError, TileError, TileUnsupportedFeatureError, FunctionDesc)
from cuda.tile._numeric_semantics import RoundingMode
from cuda.tile._ir.ir import Block, Loc, Var, IRContext
from cuda.tile._ir.ops_utils import (
    padding_mode_to_bytecode, rounding_mode_to_bytecode,
    get_default_rounding_mode,
)
from cuda.tile._ir.type import (
    PartitionViewTy, StridedViewTy, GatherScatterViewTy, Type, TileTy, TokenTy,
    ArrayTy, size_to_bytecode,
)

if TYPE_CHECKING:
    from cuda.tile._passes.dataflow_analysis import DataflowResult


def dtype_typeid(tt: bc.TypeTable, dtype: datatype.DType) -> bc.TypeId:
    if is_pointer_dtype(dtype):
        pointee_dtype = PointerInfo(dtype).pointee_dtype
        pointee = dtype_typeid(tt, pointee_dtype)
        return tt.pointer(pointee, bc.PtrAttr.Missing)
    return tt.simple(dtype_simple_bytecode_type(dtype))


def tensor_view_typeid(tt: bc.TypeTable, array_ty: ArrayTy,
                       shape: tuple[int | None, ...],
                       strides: tuple[int | None, ...]) -> bc.TypeId:
    dtype = dtype_typeid(tt, array_ty.dtype)
    shape_bc = [size_to_bytecode(x) for x in shape]
    strides_bc = [size_to_bytecode(x) for x in strides]
    return tt.tensor_view(dtype, shape_bc, strides_bc, bc.PtrAttr.Missing)


def tensor_view_typeid_for_list(tt: bc.TypeTable, item_size_words: int) -> bc.TypeId:
    shape = [bc.DYNAMIC_SHAPE, item_size_words]
    strides = [item_size_words, 1]
    return tt.tensor_view(tt.I64, shape, strides, bc.PtrAttr.Missing)


def view_typeid(tt: bc.TypeTable, ty: Type, array_tv_id: bc.TypeId) -> bc.TypeId:
    """Lower a view type on top of `array_tv_id`, the tensor-view id its array was lowered to.

    A view never recomputes that id from `ty.array_ty`: the array's static shape and strides are
    carried by the operands of its MakeTensorView, not by `ArrayTy` alone.
    """
    padding_value = padding_mode_to_bytecode[ty.padding_mode]
    assert isinstance(ty.array_ty, ArrayTy)
    if isinstance(ty, StridedViewTy):
        return tt.strided_view(ty.tile_shape, ty.traversal_steps, array_tv_id,
                               ty.order, padding_value)
    elif isinstance(ty, PartitionViewTy):
        return tt.partition_view(ty.tile_shape, array_tv_id, ty.order, padding_value)
    elif isinstance(ty, GatherScatterViewTy):
        return tt.gather_scatter_view(ty.tile_shape, array_tv_id, ty.sparse_dim, padding_value)
    else:
        raise NotImplementedError(f"Lowering view type '{ty}' is not supported")


def typeid(tt: bc.TypeTable, ty: Type) -> bc.TypeId:
    if isinstance(ty, TileTy):
        dtype = dtype_typeid(tt, ty.dtype)
        shape = list(ty.shape)
        return tt.tile(dtype, shape)
    elif isinstance(ty, TokenTy):
        return tt.Token
    else:
        raise NotImplementedError(f"Lowering type '{ty}' is not supported")


def get_list_item_repr_size_in_words(item_ty: Type) -> int:
    if isinstance(item_ty, ArrayTy):
        # Base pointer + shape + strides
        return 1 + 2 * item_ty.ndim
    else:
        raise NotImplementedError(f"List of type '{item_ty}' are not supported")


def get_list_partition_view_tile_size(item_size_words: int) -> int:
    # Round up the item size to the nearest power of two
    return 1 << (item_size_words - 1).bit_length()


# Encode a single int/float value according to the MLIR "DenseElementsAttr" splat format.
def _constant_to_bytes(value: int | float, dtype: DType) -> bytes:
    if dtype == datatype.bool_:
        # Note that MLIR requires "0xFF" for "True" value.
        return b"\xff" if value else b"\x00"
    elif datatype.is_integral(dtype):
        return int(value).to_bytes((dtype.bitwidth + 7) // 8, "little", signed=value < 0)
    elif datatype.is_float(dtype):
        # Note that TF32 is stored as 3 bytes despite the "32" in its name.
        # Its float_bit_size() is 19 bits, which is rounded up to 24 bits.
        bits = bc.float_to_bits(value, dtype_simple_bytecode_type(dtype))
        bit_size = bc.float_bit_size(dtype_simple_bytecode_type(dtype))
        return bits.to_bytes((bit_size + 7) // 8, "little")
    else:
        raise TypeError(f"Cannot make a constant out of {dtype}")


# Encode a potentially nested constant tuple as the raw row-major byte buffer according to MLIR's
# "DenseElementsAttr" non-splat format.
def _constant_tuple_to_bytes(value, dtype: DType, shape: tuple[int, ...]) -> bytes:
    # Note that MLIR requires bit packing for non-splat DenseElementsAttr<i1>
    if dtype == datatype.bool_ and isinstance(value, tuple):
        flat_bools = _flatten_bools(value)
        bits = 0
        for i, v in enumerate(flat_bools):
            if v:
                bits |= 1 << i
        return bits.to_bytes((len(flat_bools) + 7) // 8, "little")

    if len(shape) == 0:
        return _constant_to_bytes(value, dtype)

    assert len(value) == shape[0]
    return b"".join(_constant_tuple_to_bytes(c, dtype, shape[1:]) for c in value)


def _flatten_bools(value) -> tuple[bool]:
    if not isinstance(value, tuple):
        return (bool(value),)
    return sum((_flatten_bools(v) for v in value), start=())


def _get_type_conversion_encoder(from_dtype: Type, to_dtype: Type, *,
                                 rounding_mode: RoundingMode | None = None):

    def kind(t):
        if datatype.is_float(t):
            return 'f'
        if datatype.is_integral(t) or datatype.is_boolean(t):
            return 'si' if datatype.is_signed(t) else 'ui'
        raise TileInternalError(f'Unsupported dtype: {t}')

    from_kind, to_kind = kind(from_dtype), kind(to_dtype)

    if rounding_mode is not None:
        rounding_mode = rounding_mode_to_bytecode[rounding_mode]
    else:
        rounding_mode = bc.RoundingMode.NEAREST_EVEN

    round_to_float = rounding_mode_to_bytecode[get_default_rounding_mode()]
    partial = functools.partial
    match from_kind, to_kind:
        case 'f', 'f': return partial(bc.encode_FToFOp, rounding_mode=rounding_mode)
        case 'f', 'si': return partial(bc.encode_FToIOp,
                                       signedness=bc.Signedness.Signed,
                                       rounding_mode=bc.RoundingMode.NEAREST_INT_TO_ZERO,
                                       saturating=False)
        case 'f', 'ui': return partial(bc.encode_FToIOp,
                                       signedness=bc.Signedness.Unsigned,
                                       rounding_mode=bc.RoundingMode.NEAREST_INT_TO_ZERO,
                                       saturating=False)
        case 'si', 'f': return partial(bc.encode_IToFOp,
                                       signedness=bc.Signedness.Signed,
                                       rounding_mode=round_to_float)
        case 'ui', 'f': return partial(bc.encode_IToFOp,
                                       signedness=bc.Signedness.Unsigned,
                                       rounding_mode=round_to_float)

    if from_dtype.bitwidth < to_dtype.bitwidth or from_dtype is datatype.bool_:
        assert from_kind in ("si", "ui")
        return partial(bc.encode_ExtIOp, signedness=get_signedness(from_dtype))
    elif from_dtype.bitwidth > to_dtype.bitwidth:
        return partial(bc.encode_TruncIOp, overflow=bc.IntegerOverflow.NONE)
    elif from_kind in ("si", "ui") and to_kind in ("si", "ui"):
        # Signed-to-unsigned or unsigned-to-signed conversion without changing bitwidth is a no-op
        return lambda _builder, _type, val: val
    raise NotImplementedError(f"Type coversion from {from_dtype} to {to_dtype} not implemented")


def convert_dtype(ctx: "BytecodeContext", val: bc.Value, fromty: Type,
                  toty: Type, *, rounding_mode: RoundingMode | None = None) -> bc.Value:
    from_dtype = fromty.dtype if isinstance(fromty, TileTy) else fromty
    to_dtype = toty.dtype if isinstance(toty, TileTy) else toty
    toty_id = typeid(ctx.type_table, toty)
    if to_dtype == datatype.bool_ and datatype.is_integral(from_dtype):
        # TruncIOp is not doing pytorch style boolean casting (x != 0)
        # We have to use CmpIOp instead
        zero = ctx.constant(0, fromty)
        return bc.encode_CmpIOp(
            ctx.builder,
            result_type=toty_id,
            lhs=val,
            rhs=zero,
            comparison_predicate=bc.ComparisonPredicate.NOT_EQUAL,
            signedness=datatype.get_signedness(from_dtype))
    else:
        encoder = _get_type_conversion_encoder(from_dtype, to_dtype, rounding_mode=rounding_mode)
        return encoder(ctx.builder, toty_id, val)


def _broadcast_shape(ctx: "BytecodeContext",
                     val: bc.Value, fromty: TileTy, toty: TileTy):
    if len(fromty.shape) < len(toty.shape):
        # prepend 1s if input_shape have fewer dimensions
        diff = len(toty.shape) - len(fromty.shape)
        new_shape = (1,) * diff + fromty.shape
        reshaped_ty = TileTy(fromty.dtype, new_shape)
        reshaped_ty_id = typeid(ctx.type_table, reshaped_ty)
        val = bc.encode_ReshapeOp(ctx.builder, reshaped_ty_id, val)
        fromty = reshaped_ty

    if fromty.shape != toty.shape:
        broadcasted_ty = TileTy(fromty.dtype, toty.shape)
        broadcasted_ty_id = typeid(ctx.type_table, broadcasted_ty)
        val = bc.encode_BroadcastOp(ctx.builder, broadcasted_ty_id, val)
        fromty = broadcasted_ty
    return val, fromty


def _get_reduce_indices(
    ctx: "BytecodeContext", input_shape: Tuple[int, ...], output_ty: TileTy,
    normalized_axis: int,
) -> bc.Value:
    tt = ctx.type_table
    # iota
    indices_ty = TileTy(
        output_ty.dtype, (input_shape[normalized_axis],)
    )
    indices = bc.encode_IotaOp(ctx.builder, typeid(tt, indices_ty))

    # prepend and append 1 until normalized_axis is at the right dimension.
    new_shape = [1] * len(input_shape)
    new_shape[normalized_axis] = input_shape[normalized_axis]
    indices_ty = TileTy(
        output_ty.dtype, tuple(new_shape)
    )
    indices = bc.encode_ReshapeOp(ctx.builder, typeid(tt, indices_ty), indices)
    # broadcast to input_shape
    to_indices_ty = TileTy(output_ty.dtype, tuple(input_shape))
    res, _ = _broadcast_shape(ctx, indices, indices_ty, to_indices_ty)
    return res


def encode_comparison(builder: bc.CodeBuilder, fn: str, lhs: bc.Value, rhs: bc.Value,
                      dtype: DType, result_typeid: bc.TypeId) -> bc.Value:
    match fn:
        case "eq": pred = bc.ComparisonPredicate.EQUAL
        case "ne": pred = bc.ComparisonPredicate.NOT_EQUAL
        case "ge": pred = bc.ComparisonPredicate.GREATER_THAN_OR_EQUAL
        case "gt": pred = bc.ComparisonPredicate.GREATER_THAN
        case "le": pred = bc.ComparisonPredicate.LESS_THAN_OR_EQUAL
        case "lt": pred = bc.ComparisonPredicate.LESS_THAN

    if datatype.is_float(dtype):
        order = bc.ComparisonOrdering.UNORDERED if fn == 'ne' else bc.ComparisonOrdering.ORDERED
        return bc.encode_CmpFOp(builder,
                                result_type=result_typeid,
                                comparison_predicate=pred,
                                comparison_ordering=order,
                                lhs=lhs, rhs=rhs)
    elif datatype.is_integral(dtype) or datatype.is_boolean(dtype):
        return bc.encode_CmpIOp(builder,
                                result_type=result_typeid,
                                comparison_predicate=pred,
                                signedness=datatype.get_signedness(dtype),
                                lhs=lhs, rhs=rhs)
    else:
        raise TileInternalError(f'Unexpected dtype: {dtype}')


def create_synthetic_linkage_name(func_desc: FunctionDesc) -> str:
    # Build a synthetic linkage name for a helper function or lambda. Format:
    #
    #     <name>@<basename>:<line>:<column>_<specialization_id>
    #
    # By construction every FunctionDesc reaching this point has been
    # concretized by hir2ir — only the kernel entry skips this path (it uses
    # the externally-visible symbol instead), so the specialization_id is
    # required here.
    assert func_desc.specialization_id is not None, (
        f"create_synthetic_linkage_name called on a FunctionDesc without a "
        f"specialization_id: {func_desc}. hir2ir must concretize every "
        f"non-entry function before bytecode generation."
    )
    base_name = os.path.basename(func_desc.filename) or "unknown"
    stem = os.path.splitext(base_name)[0]
    # Convert any non-alphanumeric chars to _.
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem) or "anonymous"
    func_part = func_desc.name if func_desc.name is not None else "lambda"
    return (f"{func_part}@{stem}:{func_desc.line}:{func_desc.column}"
            f"_{func_desc.specialization_id}")


class DebugAttrMap:
    def __init__(self,
                 debug_attr_table: bc.DebugAttrTable,
                 entry_symbol: str,
                 anonymize: bool):
        self._subprogram_cache = {}
        self._debug_attr_table = debug_attr_table
        self._entry_symbol = entry_symbol
        self._anonymize = anonymize

    def _linkage_for(self, func_desc: FunctionDesc) -> str:
        # The kernel entry point keeps the externally-visible symbol so it can
        # be looked up by the loader. Every other function gets a per-function
        # artificial linkage name.
        if func_desc.is_entry:
            return self._entry_symbol
        return create_synthetic_linkage_name(func_desc)

    def get_subprogram(self, func_desc: FunctionDesc) -> bc.DebugAttrId:
        # Every FunctionDesc reaching DI emission must satisfy: a function has
        # no specialization_id iff it is the kernel entry. hir2ir leaves the
        # entry's abstract desc as-is and concretizes everyone else; anything
        # else is a bug.
        assert func_desc.is_entry == (func_desc.specialization_id is None), (
            f"FunctionDesc invariant violated: is_entry={func_desc.is_entry} "
            f"but specialization_id={func_desc.specialization_id!r}: {func_desc}"
        )
        try:
            return self._subprogram_cache[func_desc]
        except KeyError:
            pass

        func_dirname, func_basename = os.path.split(func_desc.filename)
        file_attr = self._debug_attr_table.file(func_basename, func_dirname)
        compile_unit_attr = self._debug_attr_table.compile_unit(file_attr)
        ret = self._debug_attr_table.subprogram(
            file=file_attr,
            line=func_desc.line,
            name="<lambda>" if func_desc.name is None else func_desc.name,
            linkage_name=self._linkage_for(func_desc),
            compile_unit=compile_unit_attr,
            scope_line=func_desc.line,
        )
        self._subprogram_cache[func_desc] = ret
        return ret

    def get_debugattr(self, loc: Loc) -> bc.DebugAttrId:
        if self._anonymize or loc.is_unknown():
            return bc.MISSING_DEBUG_ATTR_ID

        subprogram = self.get_subprogram(loc.function)
        attr = self._debug_attr_table.loc(subprogram, loc.filename, loc.line, loc.col)
        if loc.call_site is not None:
            caller_loc = self.get_debugattr(loc.call_site)
            attr = self._debug_attr_table.call_site(attr, caller_loc)
        return attr


class BytecodeContext:
    def __init__(self,
                 builder: bc.CodeBuilder,
                 type_table: bc.TypeTable,
                 debug_attr_map: DebugAttrMap,
                 global_section: bc.GlobalSection,
                 ir_ctx: IRContext,
                 dataflow_result: "DataflowResult",
                 sm_arch: str | None) -> None:
        self.builder = builder
        self.type_table = type_table
        self._debug_attr_map = debug_attr_map
        self.global_section = global_section
        self._typemap: Dict[str, Type] = ir_ctx.typemap
        self._constants: Dict[str, Any] = ir_ctx.constants
        self._dataflow_result = dataflow_result
        self._value_map: Dict[str, bc.Value] = {}
        self._array_base_ptr: Dict[str, bc.Value] = {}
        # Tensor-view TypeId computed by each array's MakeTensorView, keyed by the array var name.
        self._array_tv_id: Dict[str, bc.TypeId] = {}
        self._list_partition_views: Dict[str, bc.Value] = {}
        self.sm_arch = sm_arch
        self.innermost_loop = None

    @contextmanager
    def loc(self, loc: Loc):
        debug_attr_id = self._debug_attr_map.get_debugattr(loc)
        with loc, self.builder.debug_attr(debug_attr_id):
            yield

    @contextmanager
    def enter_loop(self, loop):
        old = self.innermost_loop
        self.innermost_loop = loop
        try:
            yield
        finally:
            self.innermost_loop = old

    def typeof(self, var: Var) -> Type:
        return self._typemap[var.name]

    def typeid_of(self, var: Var) -> bc.TypeId:
        return typeid(self.type_table, self.typeof(var))

    def is_constant(self, var: Var) -> bool:
        return var.name in self._constants

    def get_constant(self, var: Var):
        return self._constants[var.name]

    def get_constant_or_default(self, var: Var, default=None):
        return self._constants.get(var.name, default)

    def get_known_constant_value(self, var: Var) -> int | None:
        return self._dataflow_result.constant_value(var)

    def set_array_tv_id(self, var: Var, tv_id: bc.TypeId) -> None:
        self._array_tv_id[var.name] = tv_id

    def get_array_tv_id(self, var: Var) -> bc.TypeId:
        return self._array_tv_id[var.name]

    def get_value(self, var: Var) -> bc.Value:
        return self._value_map[var.name]

    def get_optional_value(self, var: Var) -> Optional[bc.Value]:
        if var.name in self._constants and self._constants[var.name] is None:
            return None
        else:
            return self.get_value(var)

    def set_value(self, var: Var, value: bc.Value) -> None:
        name = var.name
        if name in self._value_map:
            raise ValueError(f"Variable {name} is already in the value map")
        self._value_map[name] = value

    def cast(self, val: bc.Value, fromty: Type, toty: Type) -> bc.Value:
        assert isinstance(fromty, TileTy)
        assert isinstance(toty, TileTy)
        if fromty == toty:
            return val
        if fromty.shape != toty.shape:
            val, fromty = _broadcast_shape(self, val, fromty, toty)
        if fromty.dtype != toty.dtype:
            val = convert_dtype(self, val, fromty, toty)
        return val

    def bitcast(self, value: bc.Value, fromty: Type, toty: Type) -> bc.Value:
        assert isinstance(fromty, TileTy)
        assert isinstance(toty, TileTy)
        if fromty == toty:
            return value
        if fromty.shape != toty.shape:
            value, fromty = _broadcast_shape(self, value, fromty, toty)
        if fromty.dtype != toty.dtype:
            value = bc.encode_BitcastOp(self.builder, typeid(self.type_table, toty), value)
        return value

    def constant(self, value, ty: Type) -> bc.Value:
        if not isinstance(ty, TileTy):
            raise TypeError(f"Cannot encode a constant of type {ty}; expected a TileTy")

        def get_numel(v):
            if not isinstance(v, tuple):
                return 1
            return sum(get_numel(i) for i in v)

        if get_numel(value) == 1:
            while isinstance(value, tuple):
                value = value[0]
            data = _constant_to_bytes(value, ty.dtype)
        else:
            assert isinstance(value, tuple)
            data = _constant_tuple_to_bytes(value, ty.dtype, ty.shape)
        return bc.encode_ConstantOp(self.builder, typeid(self.type_table, ty), data)

    def index_tuple(self,
                    index: tuple[Var, ...], *, keep_i64: bool = False) -> Tuple[bc.Value, ...]:
        i32_tile_ty = self.type_table.tile(self.type_table.I32, ())
        item_types = tuple(x.get_type() for x in index)
        index_values = tuple(self.get_value(x) for x in index)
        if keep_i64:
            return index_values
        return tuple(
            bc.encode_TruncIOp(self.builder, i32_tile_ty, v, bc.IntegerOverflow.NONE)
            if (t.dtype if isinstance(t, TileTy) else t).bitwidth > 32 else v
            for v, t in zip(index_values, item_types, strict=True)
        )

    def load_store_hints(self,
                         latency: Optional[int],
                         allow_tma: Optional[bool]) -> Optional[bc.OptimizationHints]:
        if latency is None and allow_tma is None:
            return None
        if allow_tma is None:
            allow_tma = True
        load_store_hints = bc.LoadStoreHints(latency=latency, allow_tma=allow_tma)
        if self.builder.version < BytecodeVersion.V_13_3:
            assert self.sm_arch is not None
            return make_load_store_hints({self.sm_arch: load_store_hints})
        else:
            return make_load_store_hints({"default": load_store_hints})


def generate_bytecode_for_block(ctx: BytecodeContext, block: Block):
    for op in block.operations:
        with ctx.loc(op.loc):
            try:
                result_values = op.generate_bytecode(ctx)
                if isinstance(result_values, bc.Value):
                    result_values = (result_values,)

                for result_var, val in zip(op.result_vars, result_values, strict=True):
                    assert isinstance(val, bc.Value)
                    ctx.set_value(result_var, val)
            except TileError:
                raise
            except Exception as e:
                raise TileInternalError(f"Internal error: {e}") from e


def _resolve_num_worker_warps(num_worker_warps: Optional[int],
                              version: BytecodeVersion) -> Optional[int]:
    if num_worker_warps is not None and version < BytecodeVersion.V_13_3:
        warnings.warn(
            f"num_worker_warps is ignored: requires tileiras {BytecodeVersion.V_13_3.as_string()},"
            f" but current version is {version.as_string()}."
        )
        return None

    return num_worker_warps


def generate_bytecode_for_kernel(func_body: Block,
                                 dataflow_result: "DataflowResult",
                                 symbol: str,
                                 compiler_options: CompilerOptions,
                                 sm_arch: str | None,
                                 writer: bc.BytecodeWriter,
                                 anonymize_debug_attr: bool):
    version = writer.version
    hints_by_target = compiler_options.hints_by_target()
    if version < BytecodeVersion.V_13_3:
        if sm_arch is None:
            raise TileUnsupportedFeatureError(
                f"Architecture-independent TileIR bytecode generation requires version"
                f" {BytecodeVersion.V_13_3.as_string()} or later, but got {version.as_string()}."
                f" Specify a target architecture (gpu_code) for an earlier version.")
        specialized_hints = dict(hints_by_target.get("default", {}))
        specialized_hints.update(hints_by_target.get(sm_arch, {}))
        hints_by_target = {sm_arch: specialized_hints}

    hints = {
        target: bc.EntryHints(
            num_cta_in_cga=fields.get("num_ctas"),
            occupancy=fields.get("occupancy"),
            num_worker_warps_per_cta=_resolve_num_worker_warps(fields.get("num_worker_warps"),
                                                               version))
        for target, fields in hints_by_target.items()
    }

    param_type_ids = [typeid(writer.type_table, p.get_type()) for p in func_body.params]
    debug_attr_map = DebugAttrMap(writer.debug_attr_table, symbol,
                                  anonymize=anonymize_debug_attr)
    func_debug_attr = debug_attr_map.get_debugattr(func_body.loc)

    with writer.function(name=symbol,
                         parameter_types=param_type_ids,
                         result_types=(),
                         entry_point=True,
                         hints=hints,
                         debug_attr=func_debug_attr) as (builder, param_values):
        ctx = BytecodeContext(builder=builder,
                              type_table=writer.type_table,
                              debug_attr_map=debug_attr_map,
                              global_section=writer.global_section,
                              ir_ctx=func_body.ctx,
                              dataflow_result=dataflow_result,
                              sm_arch=sm_arch)

        for var, value in zip(func_body.params, param_values, strict=True):
            ctx.set_value(var, value)

        generate_bytecode_for_block(ctx, func_body)
