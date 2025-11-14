# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
import math
import operator
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Union, Any, List, Iterator, Callable

from typing_extensions import override

import cuda.tile._stub as ct
from cuda.tile import _datatype as datatype, TileValueError
from cuda.tile import RoundingMode, MemoryOrder, MemoryScope
from cuda.tile._mutex import tile_mutex
from cuda.tile._exception import (
    TileTypeError,
    ConstFoldNotImplementedError,
)
from cuda.tile._ir.ir import (
    AstOperation, Operation, Var, Loc, Block, TypeResult,
    has_side_effects, terminator, Mapper, add_operation, TypedOperation, RangeInfo, Builder
)
from cuda.tile._ir.load_store_impl import (
    check_load_store_hints,
    tile_load_generate_bytecode, tile_store_generate_bytecode,
    load_pointer_lowering, store_pointer_lowering,
)
from .op_impl import (
    impl, require_constant_int, require_constant_int_tuple,
    require_tile_type, normalize_axis, require_dtype_spec,
    require_tile_or_scalar_type, require_constant_bool, require_optional_constant_enum,
    require_constant_str, require_array_type, require_tuple_type, require_constant_slice,
    require_list_type, require_integer_dtype, require_scalar_or_0d_tile_type,
    require_index_or_index_tuple_type, require_constant_shape, require_constant_axis_order,
    require_constant_enum, require_optional_constant_int, require_optional_constant_bool,
    require_optional_constant_str, PrintfValidator, require_tile_or_scalar_maybe_loose_type,
    require_scalar_or_0d_tile_maybe_loose_type)
from .ops_utils import (
    BINOP_REGISTRY, UNARYOP_REGISTRY,
    check_dtype_autocast,
    check_rd_and_ftz, PaddingMode,
    rounding_mode_to_bytecode, get_dtype, change_dtype, memory_order_to_bytecode,
    memory_scope_to_bytecode, broadcast_shapes2, is_shape_broadcastable_to, BroadcastError,
    promote_types, promote_dtypes
)
from .typing_support import typeof_pyval, dtype_registry, loose_type_of_pyval
from .type import (
    TupleTy, TileTy, NoneType, BoundMethodTy, SizeTy, ArrayTy,
    ListTy, make_tile_ty, SliceType, DTypeConstructor, PointerTy, RangeIterType, Type,
    UNDEFINED, NONE, ModuleTy, TypeTy, LooselyTypedScalar, DTypeSpec, StringTy
)
from cuda.tile._datatype import (
    DType, is_integral, is_float, is_signed, is_boolean, is_restricted_float
)
from cuda.tile._ir2bytecode import (
    lower_reduce,
    lower_reduce_argmax_argmin, lower_scan,
    unflatten_values, BytecodeContext, typeid_tuple, typeid,
    generate_bytecode_for_block, convert_dtype, flatten_type, get_list_item_repr_size_in_words,
    get_list_partition_view_tile_size
)
from cuda.tile._passes.typeinfer import (
    TypingContext, infer_types_in_block, propagate_type
)
import cuda.tile._bytecode as bc


# ================================================
# Ast Parsing operations
# ================================================
class Store(AstOperation):
    def __init__(self, lhs_var_name: str, rhs: Var, loc: Loc):
        super().__init__("store",
                         operands={"rhs": rhs},
                         attributes={"lhs_var_name": lhs_var_name},
                         result_vars=[], loc=loc)


def store(lhs_var_name: str, rhs: Var, block: Block, loc: Loc) -> None:
    store_op = Store(lhs_var_name, rhs, loc)
    block.append(store_op)


class Load(AstOperation):
    def __init__(self, var_name: str, result_var: Var, loc: Loc):
        super().__init__("load", operands={},
                         attributes={"var_name": var_name},
                         result_vars=[result_var], loc=loc)


def load(var_name: str, block: Block, loc: Loc, res: Var) -> None:
    load_op = Load(var_name, res, loc)
    block.append(load_op)


class Call(Operation):
    def __init__(
        self,
        func: Var,
        args: Tuple[Var, ...],
        kwargs: Tuple[Tuple[str, Var], ...],
        result_var: Var,
        loc: Loc,
    ):
        kwarg_vars = tuple(v for _, v in kwargs)
        kwarg_names = tuple(k for k, _ in kwargs)
        super().__init__("call",
                         operands={"func": func, "args": args, "kwargs": kwarg_vars},
                         attributes={"kwarg_names": kwarg_names},
                         result_vars=[result_var], loc=loc)

    def kwarg_dict(self):
        return dict(zip(self.kwarg_names, self.kwargs))


def call(
    func: Var, args: Tuple[Var, ...], kwargs: Tuple[Tuple[str, Var], ...],
    block: Block, loc: Loc, res: Var
) -> None:
    call_op = Call(func, args, kwargs, res, loc)
    block.append(call_op)


# ================================================
# Control flow operations
# ================================================


# This is not a dataclass because we want object identity hashing
class CarriedVariables:
    names: Sequence[str]
    initial: Sequence[Var]
    body: Sequence[Var]
    results: Sequence[Var] = ()

    def __init__(self,
                 names: Sequence[str],
                 initial: Sequence[Var],
                 body: Sequence[Var],
                 results: Sequence[Var] = ()):
        self.names = names
        self.initial = initial
        self.body = body
        self.results = results

    def zipped_triplets(self) -> Iterator[Tuple[Var, Var, Var]]:
        return zip(self.initial, self.body, self.results)


@dataclass
class ForLoopInfo:
    induction_var: Var
    iterable: Var


class Loop(Operation):
    def __init__(
        self,
        body: Block,
        loc: Loc,
        for_loop: Optional[ForLoopInfo] = None,
        carried_vars: Optional[CarriedVariables] = None,
    ):
        super().__init__(
            "loop",
            operands={},
            result_vars=carried_vars.results if carried_vars else [],
            nested_blocks=[body],
            loc=loc,
        )
        self.for_loop: Optional[ForLoopInfo] = for_loop
        self.carried_vars = carried_vars

    @property
    def body(self):
        return self.nested_blocks[0]

    def clone(self, mapper: Mapper) -> Operation:
        if self.carried_vars is not None:
            new_carried_vars = CarriedVariables(
                names=self.carried_vars.names,
                initial=tuple(mapper.get_var(v) for v in self.carried_vars.initial),
                body=mapper.clone_vars(self.carried_vars.body),
                results=mapper.clone_vars(self.carried_vars.results),
            )
            mapper.set_object(self.carried_vars, new_carried_vars)
        else:
            new_carried_vars = None

        if self.for_loop is not None:
            new_induction_var = mapper.clone_var(self.for_loop.induction_var)
            new_for_loop = ForLoopInfo(new_induction_var, self.for_loop.iterable)
        else:
            new_for_loop = None

        res = self._clone_impl(mapper, [] if new_carried_vars is None else new_carried_vars.results)
        res.for_loop = new_for_loop
        res.new_cattied_vars = new_carried_vars
        return res

    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        if self.for_loop is not None:
            # Assign type to induction_var
            iterable_type = typing_context.get_type(self.for_loop.iterable)
            if not isinstance(iterable_type, RangeIterType):
                raise TypeError(f"Expected a range iterable for {self.for_loop.iterable.name},"
                                f"got {iterable_type}")
            typing_context.set_type(self.for_loop.induction_var, iterable_type.dtype)

            for initial_var, res_var in zip(self.carried_vars.initial, self.carried_vars.results):
                typing_context.phi_propagate_constant(initial_var, res_var)

        for initial_var, body_var in zip(self.carried_vars.initial, self.carried_vars.body):
            typing_context.set_type(body_var, typing_context.get_type(initial_var))

        # TODO: propagate constants from initial to body variables

        for nested_block in self.nested_blocks:
            infer_types_in_block(typing_context, nested_block)

        for var in self.carried_vars.results:
            typing_context.phi_finalize_constant(var)
        return [typing_context.get_type(body_var) for body_var in self.carried_vars.body]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[tuple[bc.Value, ...], ...]:
        result_types = tuple(ctx.typeof(x) for x in self.carried_vars.results)
        initial_values = [
            val
            for input_var, ty in zip(self.carried_vars.initial, result_types)
            for val in ctx.get_value_tuple_allow_undefined(input_var, ty)
        ]
        result_type_ids = [type_id
                           for ty in result_types
                           for type_id in typeid_tuple(ctx.type_table, ty)]

        assert len(result_type_ids) == len(initial_values)

        if self.for_loop is None:
            nested_builder = bc.encode_LoopOp(ctx.builder, result_type_ids, initial_values)
            block_arg_type_ids = result_type_ids
        else:
            start, stop, step = ctx.get_value_tuple(self.for_loop.iterable)
            nested_builder = bc.encode_ForOp(ctx.builder, result_type_ids, start, stop, step,
                                             initial_values)
            induction_var_type_id = ctx.typeid_of(self.for_loop.induction_var)
            block_arg_type_ids = (induction_var_type_id, *result_type_ids)

        with nested_builder.new_block(block_arg_type_ids) as block_args:
            block_args = iter(block_args)
            if self.for_loop is not None:
                ctx.set_values(self.for_loop.induction_var, (next(block_args),),)
            for var, value_tuple in zip(self.carried_vars.body,
                                        unflatten_values(block_args, result_types), strict=True):
                ctx.set_values(var, value_tuple)
            generate_bytecode_for_block(ctx, self.body)

        result_values = nested_builder.done()
        return unflatten_values(iter(result_values), result_types)

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["do"]

    @override
    def _to_string_rhs(self) -> str:
        def format_var(var):
            ty = var.try_get_type()
            if ty is None:
                return var.name
            return f"{var.name}: {ty}"

        if self.for_loop is not None:
            header_str = f"for {self.for_loop.induction_var.name} in {self.for_loop.iterable.name}"
        else:
            header_str = "loop"

        if self.carried_vars is not None:
            carried_vars_str = ", ".join(
                f"{format_var(b)} = {i.name}"
                for b, i in zip(self.carried_vars.body, self.carried_vars.initial)
                )
        else:
            carried_vars_str = "()"
        return f"{header_str} (with {carried_vars_str})"


@dataclass(eq=False)
class IfElseResults:
    names: Sequence[str]
    vars: Sequence[Var] = ()


class IfElse(Operation):
    def __init__(
        self,
        cond: Var,
        then_block: Block,
        else_block: Block,
        loc: Loc,
        results: Optional[IfElseResults] = None,
    ):
        super().__init__(
            "ifelse",
            operands={"cond": cond},
            result_vars=results.vars if results else [],
            nested_blocks=[then_block, else_block],
            loc=loc,
        )
        self.results = results

    @property
    def then_block(self):
        return self.nested_blocks[0]

    @property
    def else_block(self):
        return self.nested_blocks[1]

    def clone(self, mapper: Mapper):
        if self.results is None:
            new_result_vars = []
            new_results = None
        else:
            new_result_vars = mapper.clone_vars(self.results.vars)
            new_results = IfElseResults(self.results.names, new_result_vars)
            mapper.set_object(self.results, new_results)
        res = self._clone_impl(mapper, new_result_vars)
        res.results = new_results
        return res

    @override
    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        cond_type = typing_context.get_type(self.cond)
        if not datatype.is_boolean(cond_type):
            raise TileTypeError('condition must be a bool')

        for nested_block in self.nested_blocks:
            infer_types_in_block(typing_context, nested_block)

        for var in self.results.vars:
            typing_context.phi_finalize_constant(var)

        # end_branch should have inferred the types of the results.
        # Note the result types will be actually assigned twice:
        # once in the end_branch operation and once after this method.
        # TODO: We should do type assignment here.
        return [typing_context.get_type(var) for var in self.results.vars]

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[tuple[bc.Value, ...]]:
        cond_val = ctx.get_value(self.cond)
        result_types = tuple(ctx.typeof(v) for v in self.results.vars)
        result_type_id_tuples = tuple(typeid_tuple(ctx.type_table, t) for t in result_types)
        result_type_ids = sum(result_type_id_tuples, ())
        nested_builder = bc.encode_IfOp(ctx.builder, result_type_ids, cond_val)

        for block in (self.then_block, self.else_block):
            with nested_builder.new_block(()):
                generate_bytecode_for_block(ctx, block)

        results = nested_builder.done()
        return unflatten_values(iter(results), result_types)

    @override
    def _to_string_block_prefixes(self) -> List[str]:
        return ["then", "else"]

    @override
    def _to_string_rhs(self) -> str:
        return f"if(cond={self.cond})"


# Maps to ContinueOp in TileIR
@terminator
class Continue(Operation):
    def __init__(
        self,
        loc: Loc,
        next_vars: Tuple[Var, ...] = (),
        loop_vars: Optional[CarriedVariables] = None,
        is_for_loop_body: bool = False
    ):
        super().__init__("continue", operands={"next_vars": next_vars}, result_vars=[], loc=loc)
        self.loop_vars = loop_vars
        self.is_for_loop_body = is_for_loop_body

    def clone(self, mapper: Mapper) -> Operation:
        res = self._clone_impl(mapper, [])
        if self.loop_vars is not None:
            res.loop_vars = mapper.get_object(self.loop_vars)
        return res

    @override
    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        if self.loop_vars is not None:
            for var, body_var, result_var in zip(
                self.next_vars, self.loop_vars.body, self.loop_vars.results
            ):
                var_type = typing_context.get_type(var)
                for dest_var in (body_var, result_var):
                    if typing_context.typemap.get(
                        dest_var.name, UNDEFINED
                    ) is UNDEFINED:
                        typing_context.set_type(dest_var, var_type)

                # TODO: propagate constant values back to the body vars

                if self.is_for_loop_body:
                    typing_context.phi_propagate_constant(var, result_var)

        return []

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        next_values = [value
                       for var, res_var in zip(self.next_vars, self.loop_vars.results)
                       for value in ctx.get_value_tuple_allow_undefined(var, ctx.typeof(res_var))]
        bc.encode_ContinueOp(ctx.builder, next_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"continue {', '.join([x.name for x in self.next_vars])}"


# Maps to BreakOp
@terminator
class Break(Operation):
    def __init__(
        self,
        loc: Loc,
        output_vars: Tuple[Var, ...] = (),
        loop_vars: Optional[CarriedVariables] = None,
    ):
        super().__init__("break", operands={"output_vars": output_vars}, result_vars=[], loc=loc)
        self.loop_vars = loop_vars

    def clone(self, mapper: Mapper) -> Operation:
        res = self._clone_impl(mapper, [])
        if self.loop_vars is not None:
            res.loop_vars = mapper.get_object(self.loop_vars)
        return res

    @override
    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        if self.loop_vars is not None:
            for var, body_var, result_var in zip(
                self.output_vars, self.loop_vars.body, self.loop_vars.results
            ):
                var_type = typing_context.get_type(var)
                for dest_var in (body_var, result_var):
                    if typing_context.typemap.get(
                        dest_var.name, UNDEFINED
                    ) is UNDEFINED:
                        typing_context.set_type(dest_var, var_type)
                typing_context.phi_propagate_constant(var, result_var)

        return []

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        output_values = [value
                         for var, res_var in zip(self.output_vars, self.loop_vars.results)
                         for value in ctx.get_value_tuple_allow_undefined(var, ctx.typeof(res_var))]
        bc.encode_BreakOp(ctx.builder, output_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"break {', '.join([x.name for x in self.output_vars])}"


# Maps to YieldOp
@terminator
class EndBranch(Operation):
    def __init__(
        self,
        loc: Loc,
        outputs: Tuple[Var, ...] = (),
        ifelse_results: Optional[IfElseResults] = None,
    ):
        super().__init__("end_branch", operands={"outputs": outputs}, result_vars=[], loc=loc)
        self.ifelse_results = ifelse_results

    def clone(self, mapper: Mapper) -> Operation:
        res = self._clone_impl(mapper, [])
        if self.ifelse_results is not None:
            res.ifelse_results = mapper.get_object(self.ifelse_results)
        return res

    @override
    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        if self.ifelse_results is not None:
            for branch_res, ifelse_res in zip(self.outputs, self.ifelse_results.vars):
                propagate_type(branch_res, ifelse_res)
                typing_context.phi_propagate_constant(branch_res, ifelse_res)
        return []

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        output_values = [value
                         for var in self.outputs
                         for value in ctx.get_value_tuple(var)]
        bc.encode_YieldOp(ctx.builder, output_values)
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"yield {', '.join([x.name for x in self.outputs])}"


@terminator
class Return(Operation):
    def __init__(self, value: Var, loc: Loc):
        super().__init__("return", operands={"value": value}, result_vars=[], loc=loc)

    @override
    def infer_type(self, typing_context: TypingContext) -> TypeResult:
        value_type = typing_context.get_type(self.value)
        if value_type is not NONE:
            raise TileTypeError("Tile kernels cannot return values")
        return []

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        bc.encode_ReturnOp(ctx.builder, ())
        return ()

    @override
    def _to_string_rhs(self) -> str:
        return f"return {self.value.name}"


def _check_value_numeric_type(value: Any, dtype: DType) -> None:
    value_type = typeof_pyval(value)
    if datatype.is_arithmetic(value_type):
        if not datatype.is_arithmetic(dtype):
            raise TileTypeError(f"Expect \"value\" to be a non-numeric dtype {dtype}, "
                                f"got numeric dtype {value_type}")
        # TODO: Both are numeric types, check the data range after ir dtype supports it.
    else:
        if value_type != dtype:
            raise TileTypeError(f"Expect \"value\" to be a {dtype}, got {value_type}")


class Const(Operation):
    def __init__(
        self,
        value: Any,
        result_var: Var,
        loc: Loc,
    ):
        super().__init__(
            "const",
            operands={},
            result_vars=[result_var],
            attributes={"value": value},
            loc=loc,
        )


def const(value: Any, block: Block, loc: Loc, res: Var) -> None:
    block.append(Const(value, res, loc))


class TypedConst(TypedOperation):
    def __init__(self, value: Any, result_var: Var, loc: Loc):
        super().__init__("typed_const", operands={}, result_vars=[result_var],
                         attributes={"value": value}, loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> list[tuple[bc.Value, ...]]:
        try:
            return [ctx.constant_tuple(self.value, ctx.typeof(self.result_var))]
        except ConstFoldNotImplementedError:
            # FIXME: this is a workaround. Ideally, we should remove these Const ops
            #        with DCE. But this will only be possible when we move parameters
            #        like `order` (which is a string) from operands to attributes.
            return [()]


def loosely_typed_const(value: Any,
                        ty: Optional[Type] = None,
                        loose_ty: Optional[Type] = None) -> Var:
    if ty is None:
        ty = typeof_pyval(value)
    ret = strictly_typed_const(value, ty)
    if loose_ty is None:
        loose_ty = loose_type_of_pyval(value)
    ret.set_loose_type(loose_ty)
    return ret


def strictly_typed_const(value: Any, ty: Type) -> Var:
    ret = add_operation(TypedConst, ty, value=value)
    if not isinstance(ty, TileTy):
        # We currently don't have a way to represent a tile constant
        ret.set_constant(value)
    return ret


# Computes lhs*rhs + acc.  Also known as FMA.
class FusedMulAddOperation(Operation):
    def __init__(self, lhs: Var, rhs: Var, acc: Var,
                 rounding_mode: RoundingMode, flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "fma",
            operands={"lhs": lhs, "rhs": rhs, "acc": acc},
            attributes={"rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_type = ctx.typeof(self.result_var)
        lhs = ctx.cast(ctx.get_value(self.lhs), ctx.typeof(self.lhs), result_type)
        rhs = ctx.cast(ctx.get_value(self.rhs), ctx.typeof(self.rhs), result_type)
        acc = ctx.cast(ctx.get_value(self.acc), ctx.typeof(self.acc), result_type)
        return bc.encode_FmaOp(ctx.builder,
                               ctx.typeid_of(self.result_var),
                               lhs, rhs, acc,
                               rounding_mode_to_bytecode[self.rounding_mode],
                               self.flush_to_zero)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawComparisonOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_cmp",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        from .._ir2bytecode import encode_comparison
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        dtype = get_dtype(self.lhs.get_type())
        result_typeid = ctx.typeid_of(self.result_var)
        return encode_comparison(ctx.builder, self.fn, lhs, rhs, dtype, result_typeid)


def raw_comparison(fn: str, x: Var, y: Var) -> Var:
    if fn == 'is':
        raise TileTypeError("\"is\" only supports constants")

    ty = x.get_type()
    assert ty == y.get_type()
    res_ty = change_dtype(ty, datatype.bool_)
    return add_operation(RawComparisonOperation, res_ty, fn=fn, lhs=x, rhs=y)


@contextmanager
def _reraise_tile_exception():
    try:
        yield
    except (ZeroDivisionError, ValueError) as e:
        raise TileValueError(str(e))
    except TypeError as e:
        raise TileTypeError(str(e))


def _binop_propagate_constant(fn: str, x: Any, y: Any, type: Optional[Type]) -> Var:
    impl = BINOP_REGISTRY[fn].impl
    with _reraise_tile_exception():
        res = impl(x, y)

    if type is None:
        return loosely_typed_const(res)
    else:
        return strictly_typed_const(res, type)


@impl(ct.equal, fixed_args=["eq"])
@impl(ct.greater, fixed_args=["gt"])
@impl(ct.not_equal, fixed_args=["ne"])
@impl(ct.greater_equal, fixed_args=["ge"])
@impl(ct.less, fixed_args=["lt"])
@impl(ct.less_equal, fixed_args=["le"])
def comparison(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        res_ty = change_dtype(common_ty, datatype.bool_)
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), res_ty)

    return raw_comparison(fn, x, y)


@impl(operator.is_)
def operator_is_impl(x: Var, y: Var):
    x_ty = x.get_type()
    y_ty = y.get_type()
    if x_ty is NONE:
        return loosely_typed_const(y_ty is NONE)
    elif y_ty is NONE:
        return loosely_typed_const(x_ty is NONE)
    else:
        raise TileTypeError("Operator 'is' expects one of the operands to be None")


@impl(operator.eq, fixed_args=["eq"])
@impl(operator.ne, fixed_args=["ne"])
@impl(operator.lt, fixed_args=["lt"])
@impl(operator.le, fixed_args=["le"])
@impl(operator.gt, fixed_args=["gt"])
@impl(operator.ge, fixed_args=["ge"])
def comparison_operator_impl(fn: str, x: Var, y: Var) -> Var:
    x_ty = x.get_type()
    y_ty = y.get_type()

    match x_ty, y_ty:
        case DTypeSpec(), DTypeSpec():
            return _binop_propagate_constant(fn, x_ty.dtype, y_ty.dtype, None)
        case StringTy(), StringTy():
            return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)
        case _, _:
            return comparison(fn, x, y)


def _promote_and_broadcast_to(x: Var, ty: TileTy | DType) -> Var:
    is_tile = isinstance(ty, TileTy)
    shape = ty.shape_value if is_tile else ()
    return broadcast_to(astype(x, get_dtype(ty)), shape, keep_scalar=not is_tile)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBinaryBitwiseOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_binary_bitwise",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        res_typeid = ctx.typeid_of(self.result_var)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "and_": return bc.encode_AndIOp(ctx.builder, res_typeid, lhs, rhs)
            case "or_": return bc.encode_OrIOp(ctx.builder, res_typeid, lhs, rhs)
            case "xor": return bc.encode_XOrIOp(ctx.builder, res_typeid, lhs, rhs)
            case _:
                raise NotImplementedError(f"Missing binary bitwise implementation for {self.fn}")


def raw_binary_bitwise(fn: str, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBinaryBitwiseOperation, ty, fn=fn, lhs=x, rhs=y)


@impl(ct.bitwise_and, fixed_args=["and_"])
@impl(ct.bitwise_or, fixed_args=["or_"])
@impl(ct.bitwise_xor, fixed_args=["xor"])
@impl(operator.and_, fixed_args=["and_"])
@impl(operator.or_, fixed_args=["or_"])
@impl(operator.xor, fixed_args=["xor"])
def binary_bitwise(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = get_dtype(x_ty)
    rhs_dtype = get_dtype(y_ty)

    if not (datatype.is_integral(lhs_dtype) or datatype.is_boolean(lhs_dtype)) \
            or not (datatype.is_integral(rhs_dtype) or datatype.is_boolean(rhs_dtype)):
        raise TileTypeError("Bitwise operations require integers or booleans."
                            " Use an explicit cuda.tile.bitcast() for non-integer operands.")

    x_loose = isinstance(x_ty, LooselyTypedScalar)
    y_loose = isinstance(y_ty, LooselyTypedScalar)
    if x_loose == y_loose and lhs_dtype != rhs_dtype:
        msg = "Bitwise operands must have same data type, got:"
        msg += f" {lhs_dtype} and {rhs_dtype}"
        raise TileTypeError(msg)

    if {lhs_dtype, rhs_dtype} == {datatype.bool_, datatype.int8}:
        raise TileTypeError("Bitwise op does not support bool and int8")

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_binary_bitwise(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBitwiseShiftOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var, result_var: Var, loc: Loc):
        super().__init__(
            "raw_bitwise_shift",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_ty = self.result_var.get_type()
        res_type_id = typeid(ctx.type_table, res_ty)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "lshift":
                return bc.encode_ShLIOp(ctx.builder, res_type_id, lhs, rhs, bc.IntegerOverflow.NONE)
            case "rshift":
                return bc.encode_ShRIOp(ctx.builder, res_type_id, lhs, rhs,
                                        datatype.get_signedness(get_dtype(res_ty)))
            case _: raise NotImplementedError()


def raw_bitwise_shift(fn: str, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBitwiseShiftOperation, ty, fn=fn, lhs=x, rhs=y)


@impl(ct.bitwise_lshift, fixed_args=["lshift"])
@impl(ct.bitwise_rshift, fixed_args=["rshift"])
@impl(operator.lshift, fixed_args=["lshift"])
@impl(operator.rshift, fixed_args=["rshift"])
def bitwise_shift(fn: str, x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = get_dtype(x_ty)
    if not datatype.is_integral(lhs_dtype):
        msg = f'Bitwise shift requires an integer for left-hand side, got: {lhs_dtype}'
        raise TileTypeError(msg)

    rhs_dtype = get_dtype(y_ty)
    if not datatype.is_integral(rhs_dtype):
        msg = f'Bitwise shift requires an integer for right-hand side, got: {rhs_dtype}'
        raise TileTypeError(msg)

    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_bitwise_shift(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
class RawBinaryArithmeticOperation(TypedOperation):
    def __init__(self, fn: str, lhs: Var, rhs: Var,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "raw_binary_arith",
            operands={"lhs": lhs, "rhs": rhs},
            attributes={"fn": fn, "rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_ty = self.result_var.get_type()
        dtype = get_dtype(result_ty)
        kind = "float" if datatype.is_float(dtype) else "int"
        res_typeid = typeid(ctx.type_table, result_ty)
        rounding_mode = rounding_mode_to_bytecode[self.rounding_mode]
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)

        match self.fn, kind:
            case "add", "int":
                return bc.encode_AddIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "add", "float":
                return bc.encode_AddFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "sub", "int":
                return bc.encode_SubIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "sub", "float":
                return bc.encode_SubFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "mul", "int":
                return bc.encode_MulIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "mul", "float":
                return bc.encode_MulFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "floordiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype),
                                        rounding=bc.RoundingMode.NEGATIVE_INF)
            case "cdiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype),
                                        rounding=bc.RoundingMode.POSITIVE_INF)
            case "truediv", "float":
                return bc.encode_DivFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "pow", "float":
                return bc.encode_PowOp(ctx.builder, res_typeid, lhs, rhs)
            case "min", "int":
                return bc.encode_MinIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case "min", "float":
                return bc.encode_MinFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "max", "int":
                return bc.encode_MaxIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case "max", "float":
                return bc.encode_MaxFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "c_mod", "float":
                # C-style modulo
                return bc.encode_RemFOp(ctx.builder, res_typeid, lhs, rhs)
            case "c_mod", "int":
                # C-style modulo
                return bc.encode_RemIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=datatype.get_signedness(dtype))
            case _:
                raise NotImplementedError(f"Missing binary arithmetic implementation"
                                          f" for {self.fn}, {kind}")


def raw_binary_arithmetic(fn: str, x: Var, y: Var, rounding_mode: Optional[RoundingMode] = None,
                          flush_to_zero: bool = False) -> Var:
    ty = x.get_type()
    assert ty == y.get_type(), f"{ty} != {y.get_type()}"
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, get_dtype(ty))
    return add_operation(RawBinaryArithmeticOperation, ty, fn=fn, lhs=x, rhs=y,
                         rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def binary_arithmetic(fn: str, x: Var, y: Var, rounding_mode: Optional[RoundingMode] = None,
                      flush_to_zero: bool = False) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    if get_dtype(x_ty) == get_dtype(y_ty) == datatype.bool_:
        raise TileTypeError(f'Binary arithmetic op `{fn}` does not support bool, '
                            f'please cast bool to int')

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return _binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    force_float = (fn == "truediv")
    common_ty = promote_types(x_ty, y_ty, force_float=force_float)

    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return _binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return raw_binary_arithmetic(fn, x, y, rounding_mode, flush_to_zero)


@impl(ct.floordiv, fixed_args=["floordiv"])
@impl(ct.cdiv, fixed_args=["cdiv"])
@impl(ct.pow, fixed_args=["pow"])
@impl(operator.add, fixed_args=["add"])
@impl(operator.sub, fixed_args=["sub"])
@impl(operator.mul, fixed_args=["mul"])
@impl(operator.floordiv, fixed_args=["floordiv"])
@impl(operator.truediv, fixed_args=["truediv"])
@impl(operator.pow, fixed_args=["pow"])
@impl(min, fixed_args=["min"])
@impl(max, fixed_args=["max"])
def binary_arithmetic_impl(fn: str, x: Var, y: Var) -> Var:
    return binary_arithmetic(fn, x, y)


@impl(ct.minimum, fixed_args=["min"])
@impl(ct.maximum, fixed_args=["max"])
def binary_arithmetic_impl_with_ftz(fn: str, x: Var, y: Var, flush_to_zero: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    return binary_arithmetic(fn, x, y, flush_to_zero=flush_to_zero)


@impl(ct.add, fixed_args=["add"])
@impl(ct.sub, fixed_args=["sub"])
@impl(ct.mul, fixed_args=["mul"])
@impl(ct.truediv, fixed_args=["truediv"])
def binary_arithmetic_impl_with_rd_and_ftz(fn: str, x: Var, y: Var,
                                           rounding_mode: Var, flush_to_zero: Var) -> Var:
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return binary_arithmetic(fn, x, y, rounding_mode, flush_to_zero)


@impl(operator.mod)
@impl(ct.mod)
def mod(x: Var, y: Var) -> Var:
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)
    if get_dtype(x_ty) == get_dtype(y_ty) == datatype.bool_:
        raise TileTypeError('Modulo operation does not support bool')

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        with _reraise_tile_exception():
            res = x_ty.value % y_ty.value
        return loosely_typed_const(res)

    # Usual promote & broadcast logic
    common_ty = promote_types(x_ty, y_ty)
    x = _promote_and_broadcast_to(x, common_ty)
    y = _promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        with _reraise_tile_exception():
            res = x.get_constant() % y.get_constant()
        return strictly_typed_const(res, common_ty)

    # TileOR rem follows the C behavior while Python's mod behavior differs.
    # So we generate the C-style mod first and then apply a correction.
    value = raw_binary_arithmetic("c_mod", x, y)

    # If the sign of `value` does not match the sign of `y`, apply a correction.
    zero = strictly_typed_const(0, common_ty)
    value_sign = comparison("lt", value, zero)
    y_sign = comparison("lt", y, zero)

    # need_fix = (value_sign ^ y_sign) & (value != 0)
    sign_mismatch = binary_bitwise("xor", value_sign, y_sign)
    value_not_zero = comparison("ne", value, zero)
    need_fix = binary_bitwise("and_", sign_mismatch, value_not_zero)

    fixed_value = binary_arithmetic("add", value, y)
    return where(need_fix, fixed_value, value)


@impl(slice)
def slice_impl(start: Var, stop: Var, step: Var) -> Var:
    if not (start.is_constant() and stop.is_constant() and step.is_constant()):
        raise TileTypeError("Non-constant slices are not supported")
    return loosely_typed_const(
        slice(start.get_constant(), stop.get_constant(), step.get_constant()))


class TupleItemOperation(TypedOperation):
    def __init__(self, x: Var, index: int, result_var: Var, loc: Loc):
        super().__init__("tuple_item",
                         operands={"x": x},
                         attributes={"index": index},
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        tuple_ty = ctx.typeof(self.x)
        flat_index = 0
        for item_ty in tuple_ty.value_types[:self.index]:
            flat_index += len(flatten_type(item_ty))
        items = ctx.get_value_tuple(self.x)
        item_len = len(flatten_type(tuple_ty.value_types[self.index]))
        return [tuple(items[flat_index:flat_index + item_len])]


def tuple_item(x: Var, index: int) -> Var:
    tuple_ty = require_tuple_type(x)
    try:
        item_ty = tuple_ty[index]
    except IndexError:
        raise TileTypeError(f"Index {index} is out of range for a tuple of length {len(tuple_ty)}")

    loose_tuple_ty = x.get_loose_type()
    assert isinstance(loose_tuple_ty, TupleTy) and len(loose_tuple_ty) == len(tuple_ty)
    loose_item_ty = loose_tuple_ty[index]

    if index < 0:
        index += len(tuple_ty)

    if x.is_constant():
        item_const = x.get_constant()[index]
        return loosely_typed_const(item_const)

    if isinstance(loose_item_ty, LooselyTypedScalar):
        return loosely_typed_const(loose_item_ty.value)

    res = add_operation(TupleItemOperation, item_ty, x=x, index=index)
    res.set_loose_type(loose_item_ty)
    return res


class TupleSliceOperation(TypedOperation):
    def __init__(self, x: Var, slc: slice, result_var: Var, loc: Loc):
        super().__init__("tuple_slice",
                         operands={"x": x},
                         attributes={"slc": slc},
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        tuple_ty = ctx.typeof(self.x)
        flat_index = 0
        flat_indices = []
        for item_ty in tuple_ty.value_types:
            item_len = len(flatten_type(item_ty))
            flat_indices.append(slice(flat_index, flat_index + item_len))
            flat_index += item_len

        items = ctx.get_value_tuple(self.x)
        ret_items = []
        for sub_slice in flat_indices[self.slc]:
            ret_items.extend(items[sub_slice])

        return [tuple(ret_items)]


def tuple_slice(x: Var, slc: slice) -> Var:
    assert isinstance(slc, slice)
    tuple_ty = require_tuple_type(x)
    item_types = tuple_ty.value_types[slc]
    subtuple_ty = TupleTy(item_types)

    loose_tuple_ty = x.get_loose_type()
    assert isinstance(loose_tuple_ty, TupleTy) and len(loose_tuple_ty) == len(tuple_ty)
    loose_item_types = loose_tuple_ty.value_types[slc]
    loose_subtuple_ty = TupleTy(loose_item_types)

    if x.is_constant():
        subtuple_constant = tuple(x.get_constant()[slc])
        return loosely_typed_const(subtuple_constant, subtuple_ty, loose_subtuple_ty)

    if all(isinstance(item_ty, LooselyTypedScalar) for item_ty in loose_subtuple_ty):
        subtuple_constant = tuple(item_ty.value for item_ty in loose_subtuple_ty)
        return loosely_typed_const(subtuple_constant, subtuple_ty, loose_subtuple_ty)

    ret = add_operation(TupleSliceOperation, subtuple_ty, x=x, slc=slc)
    ret.set_loose_type(loose_subtuple_ty)
    return ret


def tuple_getitem(x: Var, key: Var) -> Var:
    key_ty = key.get_type()
    if isinstance(key_ty, SliceType):
        slc = require_constant_slice(key)
        return tuple_slice(x, slc)
    idx = require_constant_int(key)
    return tuple_item(x, idx)


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
    new_shape = list(x_type.shape_value)
    for axis in axes:
        normalized_axis = axis + new_rank if axis < 0 else axis
        new_shape.insert(normalized_axis, 1)
    return reshape(x, tuple(new_shape))


def tile_getitem(x: Var, key: Var) -> Var:
    key_ty = key.get_type()
    if isinstance(key_ty, NoneType):
        return tile_expand_dims(x, (None,))
    elif isinstance(key_ty, TupleTy):
        if not key.is_constant():
            raise TileTypeError("Tile subscript must be a constant tuple")
        return tile_expand_dims(x, key.get_constant())
    else:
        raise TileTypeError("Directly indexing a tile is not supported; "
                            "use `extract()` or `item()` instead.")


class ListItemOperation(TypedOperation):
    def __init__(self, x: Var, index: Var, result_var: Var, loc: Loc):
        super().__init__("list_item",
                         operands={"x": x, "index": index},
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        list_ty = ctx.typeof(self.x)
        assert isinstance(list_ty, ListTy)

        # First, load a (1 x item_tile_size) tile that represents the item
        partition_view = ctx.get_list_partition_view(self.x)
        item_size = get_list_item_repr_size_in_words(list_ty.item_type)
        item_tile_size = get_list_partition_view_tile_size(item_size)
        pv_tile_type_id = ctx.type_table.tile(ctx.type_table.I64, (1, item_tile_size))
        index = ctx.get_value(self.index)
        index_i32 = ctx.cast(index, ctx.typeof(self.index), datatype.int32)

        zero_i32 = ctx.constant(0, datatype.int32)

        loaded_tile, _token = bc.encode_LoadViewTkoOp(
            ctx.builder,
            tile_type=pv_tile_type_id,
            result_token_type=ctx.type_table.Token,
            view=partition_view,
            index=(index_i32, zero_i32),
            token=None,
            memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
            memory_scope=None,
            optimization_hints=None
        )

        item_typeid_tuple = typeid_tuple(ctx.type_table, list_ty.item_type)

        # Next, unpack the tile into individual values that represent the item
        if isinstance(list_ty.item_type, ArrayTy):
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
                                        (zero_i32, ctx.constant(i, datatype.int32)),),
                )
                for i in range(item_size)
            )

            # Cast each of the i64 words to appropriate types
            return [(
                # Cast the first word to data pointer
                bc.encode_IntToPtrOp(ctx.builder, item_typeid_tuple[0], extracted_words[0]),
                # Cast the remaining words to i32 shape/strides
                *(bc.encode_TruncIOp(ctx.builder, ty, w, bc.IntegerOverflow.NONE)
                  for ty, w in zip(item_typeid_tuple[1:], extracted_words[1:], strict=True))
            )]
        else:
            raise NotImplementedError(f"Indexing a list of {list_ty.item_type} is not implemented")


def list_item(x: Var, index: Var) -> Var:
    list_ty = require_list_type(x)
    require_integer_dtype(index)
    item_ty = list_ty.item_type
    return add_operation(ListItemOperation, item_ty, x=x, index=index)


@impl(operator.getitem)
def getitem(object: Var, key: Var) -> Var:
    object_ty = object.get_type()
    match object_ty:
        case TupleTy(): return tuple_getitem(object, key)
        case TileTy(): return tile_getitem(object, key)
        case ListTy(): return list_item(object, key)
    raise TileTypeError(f'Indexing an object of type {object_ty} is not supported')


class ListLen(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__("list_len", operands={"x": x}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.get_value_tuple(self.x)[1]


@impl(len)
def len_impl(x: Var) -> Var:
    x_type = x.get_type()
    if isinstance(x_type, TupleTy):
        return loosely_typed_const(len(x_type))
    require_list_type(x)
    return add_operation(ListLen, datatype.int32, x=x)


class BuildTuple(TypedOperation):
    def __init__(self, items: tuple[Var, ...], result_var: Var, loc: Loc):
        super().__init__("build_tuple",
                         operands={"items": items},
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        return [sum((ctx.get_value_tuple(x) for x in self.items), ())]


@impl(ct._build_tuple)
def build_tuple(items: tuple[Var, ...]) -> Var:
    ty = TupleTy(tuple(x.get_type() for x in items))
    loose_ty = TupleTy(tuple(x.get_loose_type() for x in items))

    if all(x.is_constant() for x in items):
        return loosely_typed_const(tuple(x.get_constant() for x in items), ty=ty, loose_ty=loose_ty)

    ret = add_operation(BuildTuple, ty, items=items)
    ret.set_loose_type(loose_ty)
    return ret


class Unary(TypedOperation):
    def __init__(self, fn: str, operand: Var,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__("unaryop",
                         operands={"operand": operand},
                         attributes={
                             "fn": fn,
                             "rounding_mode": rounding_mode,
                             "flush_to_zero": flush_to_zero,
                         },
                         result_vars=[result_var],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x = ctx.get_value(self.operand)
        rounding_mode = rounding_mode_to_bytecode[self.rounding_mode]
        flush_to_zero = self.flush_to_zero
        input_type = ctx.typeof(self.operand)
        input_dtype = get_dtype(input_type)
        flt = is_float(input_dtype) or is_restricted_float(input_dtype)
        res_type_id = ctx.typeid_of(self.result_var)

        match self.fn, flt:
            case "abs", True: return bc.encode_AbsFOp(ctx.builder, res_type_id, x)
            case "abs", False: return bc.encode_AbsIOp(ctx.builder, res_type_id, x)
            case "neg", True: return bc.encode_NegFOp(ctx.builder, res_type_id, x)
            case "neg", False: return bc.encode_NegIOp(ctx.builder, res_type_id, x)
            case "exp", True: return bc.encode_ExpOp(ctx.builder, res_type_id, x)
            case "exp2", True: return bc.encode_Exp2Op(ctx.builder, res_type_id, x,
                                                       flush_to_zero=flush_to_zero)
            case "sin", True: return bc.encode_SinOp(ctx.builder, res_type_id, x)
            case "cos", True: return bc.encode_CosOp(ctx.builder, res_type_id, x)
            case "sinh", True: return bc.encode_SinHOp(ctx.builder, res_type_id, x)
            case "cosh", True: return bc.encode_CosHOp(ctx.builder, res_type_id, x)
            case "tan", True: return bc.encode_TanOp(ctx.builder, res_type_id, x)
            case "tanh", True: return bc.encode_TanHOp(ctx.builder, res_type_id, x)
            case "log", True: return bc.encode_LogOp(ctx.builder, res_type_id, x)
            case "log2", True: return bc.encode_Log2Op(ctx.builder, res_type_id, x)
            case "sqrt", True: return bc.encode_SqrtOp(ctx.builder, res_type_id, x,
                                                       rounding_mode=rounding_mode,
                                                       flush_to_zero=flush_to_zero)
            case "rsqrt", True: return bc.encode_RsqrtOp(ctx.builder, res_type_id, x,
                                                         flush_to_zero=flush_to_zero)
            case "floor", True: return bc.encode_FloorOp(ctx.builder, res_type_id, x)
            case "ceil", True: return bc.encode_CeilOp(ctx.builder, res_type_id, x)
            case "invert", False:
                # ~tx == tx ^ ~0
                all_ones = (-1 if datatype.is_signed(input_dtype)
                            else ~(-1 << input_dtype.bitwidth))
                all_ones_tile = ctx.constant(all_ones, input_type)
                return bc.encode_XOrIOp(ctx.builder, res_type_id, x, all_ones_tile)
            case _:
                raise NotImplementedError(f"Missing implementation for unary op: {self.fn}")


def _unary_promote_to_int(x):
    return astype(x, datatype.default_int_type)


def _unary_promote_to_float(x):
    return astype(x, datatype.default_float_type)


def _unary_preserve(x):
    return x


@dataclass
class _UnaryBehavior:
    bool_handler: Optional[Callable[[Var], Var]]
    int_handler: Optional[Callable[[Var], Var]]
    float_handler: Optional[Callable[[Var], Var]]


def _unary_propagate_constant(fn: str, arg: Any) -> Any:
    impl = UNARYOP_REGISTRY[fn].impl
    with _reraise_tile_exception():
        return impl(arg)


def unary(fn: str, behavior: _UnaryBehavior, x: Var,
          rounding_mode: Optional[RoundingMode] = None, flush_to_zero: bool = False) -> Var:
    x_type = require_tile_or_scalar_maybe_loose_type(x)
    if isinstance(x_type, LooselyTypedScalar):
        res = _unary_propagate_constant(fn, x_type.value)
        return loosely_typed_const(res)

    input_dtype = get_dtype(x_type)
    if is_boolean(input_dtype):
        if behavior.bool_handler is None:
            raise TileTypeError("Boolean inputs are not supported")
        x = behavior.bool_handler(x)
    elif is_integral(input_dtype):
        if behavior.int_handler is None:
            raise TileTypeError("Integer inputs are not supported")
        x = behavior.int_handler(x)
    elif is_float(input_dtype) or is_restricted_float(input_dtype):
        if behavior.float_handler is None:
            raise TileTypeError("Float inputs are not supported")
        x = behavior.float_handler(x)
    else:
        raise TileTypeError(f"Unexpected input dtype {input_dtype}")

    ty = x.get_type()
    if x.is_constant():
        res = _unary_propagate_constant(fn, x.get_constant())
        return strictly_typed_const(res, ty)

    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, get_dtype(ty))
    return add_operation(Unary, ty, fn=fn, operand=x,
                         rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


_UNARY_FLOAT = _UnaryBehavior(_unary_promote_to_float, _unary_promote_to_float, _unary_preserve)
_UNARY_STRICT_FLOAT = _UnaryBehavior(None, None, _unary_preserve)
_UNARY_INT_FLOAT = _UnaryBehavior(_unary_promote_to_int, _unary_preserve, _unary_preserve)
_UNARY_BOOL_INT = _UnaryBehavior(_unary_preserve, _unary_preserve, None)
_UNARY_ANYTHING = _UnaryBehavior(_unary_preserve, _unary_preserve, _unary_preserve)


@impl(operator.not_)
def logical_not_impl(x: Var) -> Var:
    ty = require_scalar_or_0d_tile_maybe_loose_type(x)

    if isinstance(ty, LooselyTypedScalar):
        return loosely_typed_const(not ty.value)

    x = astype(x, datatype.bool_)
    if x.is_constant():
        return strictly_typed_const(not x.get_constant(), x.get_type())

    return add_operation(Unary, x.get_type(), fn="invert", operand=x,
                         rounding_mode=None, flush_to_zero=False)


@impl(operator.pos)
def pos_impl(x: Var):
    ty = require_tile_or_scalar_maybe_loose_type(x)

    if isinstance(ty, LooselyTypedScalar):
        return loosely_typed_const(+ty.value)

    if get_dtype(ty) == datatype.bool_:
        return astype(x, datatype.default_int_type)
    else:
        return x


@impl(ct.log, fixed_args=["log", _UNARY_FLOAT])
@impl(ct.log2, fixed_args=["log2", _UNARY_FLOAT])
@impl(ct.tan, fixed_args=["tan", _UNARY_FLOAT])
@impl(ct.tanh, fixed_args=["tanh", _UNARY_FLOAT])
@impl(ct.sin, fixed_args=["sin", _UNARY_FLOAT])
@impl(ct.sinh, fixed_args=["sinh", _UNARY_FLOAT])
@impl(ct.cos, fixed_args=["cos", _UNARY_FLOAT])
@impl(ct.cosh, fixed_args=["cosh", _UNARY_FLOAT])
@impl(ct.exp, fixed_args=["exp", _UNARY_FLOAT])
@impl(ct.bitwise_not, fixed_args=["invert", _UNARY_BOOL_INT])
@impl(ct.floor, fixed_args=["floor", _UNARY_STRICT_FLOAT])
@impl(ct.ceil, fixed_args=["ceil", _UNARY_STRICT_FLOAT])
@impl(ct.negative, fixed_args=["neg", _UNARY_INT_FLOAT])
@impl(abs, fixed_args=["abs", _UNARY_ANYTHING])
@impl(operator.invert, fixed_args=["invert", _UNARY_BOOL_INT])
@impl(operator.neg, fixed_args=["neg", _UNARY_INT_FLOAT])
def unary_impl(fn: str, behavior: _UnaryBehavior, x: Var) -> Var:
    return unary(fn, behavior, x)


@impl(ct.rsqrt, fixed_args=["rsqrt", _UNARY_FLOAT])
@impl(ct.exp2, fixed_args=["exp2", _UNARY_FLOAT])
def unary_impl_with_ftz(fn: str, behavior: _UnaryBehavior, x: Var, flush_to_zero: Var) -> Var:
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, x, flush_to_zero=flush_to_zero)


@impl(ct.sqrt, fixed_args=["sqrt", _UNARY_FLOAT])
def unary_impl_with_rd_and_ftz(fn: str, behavior: _UnaryBehavior,
                               x: Var, rounding_mode: Var, flush_to_zero: Var) -> Var:
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return unary(fn, behavior, x, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(getattr)
def getattr_impl(object: Var, name: Var) -> Var:
    ty = object.get_type()
    attr_name = require_constant_str(name)
    match ty, attr_name:
        case ArrayTy(), "dtype": return loosely_typed_const(ty.dtype)
        case ArrayTy(), "ndim": return loosely_typed_const(ty.ndim)
        case ArrayTy(), "shape": return get_array_shape(object)

        case TileTy(), "dtype": return loosely_typed_const(ty.dtype)
        case TileTy(), "shape": return loosely_typed_const(ty.shape_value)
        case TileTy(), "ndim": return loosely_typed_const(ty.ndim)

        case TileTy(), "extract": return bind_method(object, ct.extract)
        case TileTy(), "reshape": return bind_method(object, ct.reshape)
        case TileTy(), "astype": return bind_method(object, ct.astype)
        case TileTy(), "permute": return bind_method(object, ct.permute)
        case TileTy(), "transpose": return bind_method(object, ct.transpose)
        case TileTy(), "item": return bind_method(object, ct._m_tile_item)

        case ModuleTy(), _:
            try:
                return loosely_typed_const(getattr(ty.py_mod, attr_name))
            except AttributeError:
                pass

        case TypeTy(), _:
            try:
                return loosely_typed_const(getattr(ty.ty, attr_name))
            except AttributeError:
                pass

        case _: pass

    raise TileTypeError(f"No such attribute '{attr_name}' for object of type {ty}")


class GetArrayBasePtr(TypedOperation):
    def __init__(self, array: Var, result_var: Var, loc: Loc):
        super().__init__("get_array_base_ptr", operands={"array": array},
                         result_vars=[result_var], loc=loc)

    def generate_bytecode(self, ctx: "BytecodeContext"):
        arr_tuple = ctx.get_value_tuple(self.array)
        return arr_tuple[0]


def get_array_base_ptr(array: Var) -> Var:
    array_ty = require_array_type(array)
    res_ty = PointerTy(array_ty.dtype)
    return add_operation(GetArrayBasePtr, res_ty, array=array)


class GetArrayShape(TypedOperation):
    def __init__(self, value: Var, result_var: Var, loc: Loc):
        super().__init__("get_array_shape", operands={"value": value},
                         result_vars=[result_var], loc=loc)

    def generate_bytecode(self, ctx: BytecodeContext):
        ndim = ctx.typeof(self.value).ndim
        arr_tuple = ctx.get_value_tuple(self.value)
        return [arr_tuple[1:1 + ndim]]


def get_array_shape(array: Var) -> Var:
    array_ty = require_array_type(array)
    shape_ty = TupleTy((datatype.default_int_type,) * array_ty.ndim)
    loose_shape_ty = TupleTy(tuple(datatype.default_int_type
                                   if sz.maybe_value is None else LooselyTypedScalar(sz.value)
                                   for sz in array_ty.shape))
    ret = add_operation(GetArrayShape, shape_ty, value=array)
    ret.set_loose_type(loose_shape_ty)
    return ret


class GetArrayStrides(TypedOperation):
    def __init__(self, array: Var, result_var: Var, loc: Loc):
        super().__init__("get_array_strides", operands={"array": array},
                         result_vars=[result_var], loc=loc)

    def generate_bytecode(self, ctx: BytecodeContext):
        ndim = ctx.typeof(self.array).ndim
        arr_tuple = ctx.get_value_tuple(self.array)
        return [arr_tuple[1 + ndim:]]


def get_array_strides(array: Var) -> Var:
    array_ty = require_array_type(array)
    res_ty = TupleTy((datatype.default_int_type,) * array_ty.ndim)
    return add_operation(GetArrayStrides, res_ty, array=array)


class BindMethod(TypedOperation):
    def __init__(self, object: Var, func: types.FunctionType, result_var: Var, loc: Loc):
        super().__init__("bind_method", operands={"object": object},
                         attributes={"func": func}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        return [ctx.get_value_tuple(self.object)]


def bind_method(object: Var, func) -> Var:
    res_ty = BoundMethodTy(object.get_type(), func)
    return add_operation(BindMethod, res_ty, object=object, func=func)


class GetBoundSelf(TypedOperation):
    def __init__(self, bound_method: Var, result_var: Var, loc: Loc):
        super().__init__("get_bound_self", operands={"bound_method": bound_method},
                         result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        return [ctx.get_value_tuple(self.bound_method)]


def get_bound_self(bound_method: Var) -> Var:
    return add_operation(GetBoundSelf, bound_method.get_type().self_ty, bound_method=bound_method)


class Assign(TypedOperation):
    def __init__(self, value: Var, result_var: Var, loc: Loc):
        super().__init__(
            "assign", operands={"value": value}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> list[tuple[bc.Value, ...]]:
        return [ctx.get_value_tuple(self.value)]

    @override
    def _to_string_rhs(self) -> str:
        return f"{self.value.name}"


def assign(value: Var, res: Var) -> None:
    Builder.get_current().append_verbatim(Assign(value, res, res.loc))
    res.ctx.copy_type_information(value, res)
    if value.is_undefined():
        res.set_undefined()


def assign_untyped(value: Var, res: Var):
    identity_var = add_operation(Const, None, value=ct._identity)
    Builder.get_current().append_verbatim(Call(identity_var, (value,), (), res, res.loc))


@impl(ct._identity)
def identity_impl(x: Var) -> Var:
    if x.is_constant():
        return loosely_typed_const(x.get_constant(), x.get_type(), x.get_loose_type())
    else:
        return x


class Range(TypedOperation):
    def __init__(self, start: Var, stop: Var, step: Var, result_var: Var, loc: Loc):
        super().__init__("range",
                         operands={"start": start, "stop": stop, "step": step},
                         result_vars=[result_var],
                         loc=loc)
        if step.is_constant():
            result_var.set_range_info(RangeInfo(known_step=step.get_constant()))

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[tuple[bc.Value, ...]]:
        start, stop, step = (ctx.get_value_tuple(v)
                             for v in (self.start, self.stop, self.step))
        return (start + stop + step),


@impl(range)
def range_(args: Tuple[Var, ...]) -> Var:
    if not 1 <= len(args) <= 3:
        raise TileTypeError(f"Invalid number of arguments: {len(args)}")
    for arg in args:
        arg_ty = require_scalar_or_0d_tile_type(arg)
        if isinstance(arg_ty, TileTy):
            arg_ty = arg_ty.dtype
        if not datatype.is_integral(arg_ty) or not datatype.is_signed(arg_ty):
            raise TileTypeError(f"Expected a signed integer, got {arg_ty}")

    if len(args) == 1:
        start = strictly_typed_const(0, datatype.default_int_type)
        stop = args[0]
        step = strictly_typed_const(1, datatype.default_int_type)
    elif len(args) == 2:
        start, stop = args[0], args[1]
        step = strictly_typed_const(1, datatype.default_int_type)
    else:
        start, stop, step = args[0], args[1], args[2]
        # FIXME(Issue 314): Support negative step.
        # Error out if step is constant and not positive.
        if step.is_constant() and step.get_constant() <= 0:
            raise TileTypeError(f"Step must be positive, got {step.get_constant()}")

    return add_operation(
        Range,
        RangeIterType(datatype.default_int_type),
        start=start, stop=stop, step=step)


class UnpackRange(Operation):
    def __init__(self, range_object: Var, result_vars: Tuple[Var, Var, Var], loc: Loc):
        assert len(result_vars) == 3
        super().__init__("unpack_range",
                         operands={"range_object": range_object},
                         result_vars=list(result_vars),
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: "BytecodeContext"):
        start, stop, step = ctx.get_value_tuple(self.range_object)
        return (start,), (stop,), (step,)


def _register_dtype_constructors(f):
    """
    Helper decorator to register all DType constructors automatically.
    """
    for obj, ty in dtype_registry.items():
        if isinstance(ty, DTypeConstructor):
            f = impl(obj, fixed_args=[ty.dtype])(f)
    return f


@_register_dtype_constructors
def dtype_constructor_impl(new_dtype: DType, x: Var) -> Var:
    if x.is_constant():
        try:
            const_value = new_dtype._py_type(x.get_constant())
        except (ValueError, TypeError):
            raise TileTypeError(f"Invalid argument type for {new_dtype}")
        return strictly_typed_const(const_value, ty=new_dtype)

    x_ty = require_scalar_or_0d_tile_type(x)
    if isinstance(x_ty, TileTy):
        x = tile_item(x)

    return astype(x, new_dtype)


# ================================================
# Tile specific operations
# ================================================

class TileBid(TypedOperation):
    def __init__(self, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_bid", operands={}, attributes={"axis": axis}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        axis = self.axis
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_GetTileBlockIdOp(ctx.builder, res_typeid, res_typeid, res_typeid)[axis]


def bid(axis: int) -> Var:
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileBid, datatype.default_int_type, axis=axis)


@impl(ct.bid)
def bid_impl(axis: Var) -> Var:
    axis = require_constant_int(axis)
    return bid(axis)


class TileNumBlocks(TypedOperation):
    def __init__(self, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_num_blocks",
            operands={},
            attributes={"axis": axis},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        t = ctx.typeid_of(self.result_var)
        return bc.encode_GetNumTileBlocksOp(ctx.builder, t, t, t)[self.axis]


@impl(ct.num_blocks)
def num_blocks(axis: Var) -> Var:
    axis = require_constant_int(axis)
    if axis not in (0, 1, 2):
        raise TileTypeError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    return add_operation(TileNumBlocks, datatype.default_int_type, axis=axis)


class TileLoad(TypedOperation):
    def __init__(
        self, array: Var, index: Var, order: Sequence[int],
        padding_mode: PaddingMode, latency: Optional[int], allow_tma: Optional[bool],
        result_var: Var, loc: Loc
    ):
        super().__init__(
            "tile_load",
            operands={"array": array, "index": index},
            attributes={"order": tuple(order), "padding_mode": padding_mode, "latency": latency,
                        "allow_tma": allow_tma},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        (res,), _ = tile_load_generate_bytecode(self, ctx)
        return res


def tile_load(array: Var, index: Var, shape: Sequence[int], order: Sequence[int],
              padding_mode: PaddingMode, latency: Optional[int], allow_tma: Optional[bool]) -> Var:
    res_ty = make_tile_ty(array.get_type().dtype, shape)
    return add_operation(TileLoad, res_ty,
                         array=array, index=index, order=order,
                         padding_mode=padding_mode, latency=latency, allow_tma=allow_tma)


@impl(ct.load)
def tile_load_impl(array: Var, index: Var, shape: Var, order: Var,
                   padding_mode: Var, latency: Var, allow_tma: Var) -> Var:
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_len = len(index_ty) if isinstance(index_ty, TupleTy) else 1
    if array_ty.ndim != index_len:
        raise TileTypeError(f"Index size {index_len} does not match the array rank {array_ty.ndim}")

    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    broadcasted_shape = (1,) * array_ty.ndim if len(shape) == 0 else shape
    order = require_constant_axis_order(order, array_ty.ndim)
    padding_mode = require_constant_enum(padding_mode, PaddingMode)
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    check_load_store_hints(latency, allow_tma)
    result = tile_load(array, index, broadcasted_shape, order, padding_mode, latency, allow_tma)
    return reshape(result, shape)


class TileLoadTokenOrdered(TypedOperation):
    def __init__(
        self, array: Var, index: Var, order: Sequence[int],
            padding_mode: PaddingMode, token: Var, latency: Optional[int],
            allow_tma: Optional[bool],
            result_var: Var, result_token: Var, loc: Loc
    ):
        super().__init__(
            "tile_load_token_ordered",
            operands={"array": array, "index": index,
                      "token": token},
            attributes={"order": order, "padding_mode": padding_mode,
                        "latency": latency, "allow_tma": allow_tma},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[tuple[bc.Value], tuple[bc.Value]]:
        return tile_load_generate_bytecode(self, ctx)


@has_side_effects
class TileStore(TypedOperation):
    def __init__(self, array: Var, index: Var, tile: Var, order: Sequence[int],
                 latency: Optional[int], allow_tma: Optional[bool], loc: Loc):
        super().__init__(
            "tile_store",
            operands={"array": array, "index": index, "tile": tile},
            attributes={"order": order, "latency": latency, "allow_tma": allow_tma},
            result_vars=[],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[()]:
        tile_store_generate_bytecode(self, ctx)
        return ()


def tile_store(array: Var, index: Var, tile: Var, order: Sequence[int],
               latency: Optional[int], allow_tma: Optional[bool]):
    array_ty = array.get_type()
    ty = require_tile_or_scalar_type(tile)
    tile = astype(tile, array_ty.dtype)
    if isinstance(ty, DType) or ty.ndim == 0:
        tile = reshape(tile, (1,) * array_ty.ndim)
    add_operation(TileStore, (), array=array, index=index, tile=tile, order=order,
                  latency=latency, allow_tma=allow_tma)


@impl(ct.store)
def tile_store_impl(array: Var, index: Var, tile: Var, order: Var,
                    latency: Var, allow_tma: Var):
    array_ty = require_array_type(array)
    index_ty = require_index_or_index_tuple_type(index)
    index_len = len(index_ty) if isinstance(index_ty, TupleTy) else 1
    if array_ty.ndim != index_len:
        raise TileTypeError(f"Index size {index_len} does not match the array rank {array_ty.ndim}")

    src_ty = require_tile_or_scalar_type(tile)
    src_dtype = get_dtype(src_ty)
    check_dtype_autocast(src_dtype, array_ty.dtype)

    order = require_constant_axis_order(order, array_ty.ndim)
    latency = require_optional_constant_int(latency)
    allow_tma = require_optional_constant_bool(allow_tma)
    check_load_store_hints(latency, allow_tma)

    tile_store(array, index, tile, order, latency, allow_tma)


@has_side_effects
class TileStoreTokenOrdered(TypedOperation):
    def __init__(self, array: Var, index: Var, tile: Var, order: Sequence[int], token: Var,
                 latency: Optional[int], allow_tma: Optional[bool], result_token: Var, loc: Loc):
        super().__init__(
            "tile_store_token_ordered",
            operands={"array": array, "index": index, "tile": tile, "token": token},
            attributes={"order": order, "latency": latency, "allow_tma": allow_tma},
            result_vars=[result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return tile_store_generate_bytecode(self, ctx)


class LoadPointer(TypedOperation):
    def __init__(
        self,
        pointer: Var,
        mask: Optional[Var],
        padding_value: Optional[Var],
        latency: Optional[int],
        result_var: Var,
        loc: Loc,
    ):
        super().__init__(
            "load_pointer",
            operands={"pointer": pointer, "mask": mask, "padding_value": padding_value},
            attributes={"latency": latency},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        lowered_res, _ = load_pointer_lowering(self, ctx)
        return lowered_res


class LoadPointerTokenOrdered(Operation):
    def __init__(
            self,
            pointer: Var,
            mask: Optional[Var],
            padding_value: Optional[Var],
            latency: Optional[int],
            token: Var,
            result_var: Var,
            result_token: Var,
            loc: Loc,
    ):
        super().__init__(
            "load_pointer_tko",
            operands={"pointer": pointer, "mask": mask, "padding_value": padding_value,
                      "token": token},
            attributes={"latency": latency},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> tuple[tuple[bc.Value], tuple[bc.Value]]:
        lowered_res, lowered_tok = load_pointer_lowering(self, ctx)
        return (lowered_res,), (lowered_tok,)


def load_pointer(pointer: Var, mask: Optional[Var], padding_value: Optional[Var],
                 latency: Optional[int]) -> Var:
    pointer_ty = pointer.get_type()
    shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()
    result_ty = make_tile_ty(get_dtype(pointer_ty).pointee_type, shape)
    return add_operation(LoadPointer, result_ty,
                         pointer=pointer, mask=mask, padding_value=padding_value, latency=latency)


@has_side_effects
class StorePointer(TypedOperation):
    def __init__(
            self,
            pointer: Var,
            value: Var,
            mask: Optional[Var],
            latency: Optional[int],
            loc: Loc,
    ):
        super().__init__(
            "store_pointer",
            operands={"pointer": pointer, "value": value, "mask": mask},
            attributes={"latency": latency},
            result_vars=[],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        store_pointer_lowering(self, ctx)
        return ()


@has_side_effects
class StorePointerTokenOrdered(Operation):
    def __init__(
            self,
            pointer: Var,
            value: Var,
            mask: Optional[Var],
            latency: Optional[int],
            token: Var,
            result_token: Var,
            loc: Loc,
    ):
        super().__init__(
            "store_pointer_tko",
            operands={"pointer": pointer, "value": value, "mask": mask, "token": token},
            attributes={"latency": latency},
            result_vars=[result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return store_pointer_lowering(self, ctx)


def store_pointer(pointer: Var, value: Var, mask: Optional[Var], latency: Optional[int]):
    add_operation(StorePointer, (),
                  pointer=pointer, value=value, mask=mask, latency=latency)


class PointerOffset(TypedOperation):
    def __init__(self, pointer: Var, offset: Var, result_var: Var, loc: Loc):
        super().__init__(
            "pointer_offset",
            operands={"pointer": pointer, "offset": offset},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: "BytecodeContext"):
        res_typeid = ctx.typeid_of(self.result_var)
        pointer = ctx.get_value(self.pointer)
        offset = ctx.get_value(self.offset)
        return bc.encode_OffsetOp(ctx.builder, res_typeid, pointer, offset)


def pointer_offset(pointer: Var, offset: Var) -> Var:
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    offset_ty = offset.get_type()
    offset_shape = offset_ty.shape_value if isinstance(offset_ty, TileTy) else ()

    common_shape = broadcast_shapes2(pointer_shape, offset_shape)
    pointer = broadcast_to(pointer, common_shape)
    offset = broadcast_to(offset, common_shape)
    result_ty = make_tile_ty(get_dtype(pointer_ty), common_shape)
    return add_operation(PointerOffset, result_ty, pointer=pointer, offset=offset)


@impl(ct.gather)
def gather_impl(array: Var, indices: Var, padding_value: Var, check_bounds: Var,
                latency: Var) -> Var:
    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the padding value
    padding_ty = require_tile_or_scalar_type(padding_value)
    padding_shape = padding_ty.shape_value if isinstance(padding_ty, TileTy) else ()
    if not is_shape_broadcastable_to(padding_shape, pointer_shape):
        raise TileTypeError(f"Padding shape {padding_shape} is not broadcastable to the"
                            f" index shape {pointer_ty}")
    array_dtype = array.get_type().dtype

    # HACK: The default value 0 is inferred as int32, which would trigger a type error
    #       below if the array has a narrower type. So we skip the implicit cast check
    #       when the value equals the default. Ideally, we'd treat literals in a special way.
    have_default_padding_value = (padding_value.is_constant()
                                  and type(padding_const := padding_value.get_constant()) is int
                                  and padding_const == 0)

    if not have_default_padding_value:
        padding_dtype = get_dtype(padding_ty)
        if not datatype.can_autocast_dtypes(padding_dtype, array_dtype):
            raise TileTypeError(f"Padding value dtype {padding_dtype}"
                                f" cannot be implicitly cast to the array dtype {array_dtype}")
    padding_value = astype(padding_value, array_dtype)
    padding_value = broadcast_to(padding_value, pointer_shape)

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    check_load_store_hints(latency)
    return load_pointer(pointer, mask, padding_value, latency)


@impl(ct.scatter)
def scatter_impl(array: Var, indices: Var, value: Var, check_bounds: Var, latency: Var):
    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the `value`
    array_dtype = array.get_type().dtype
    value = _get_scatter_value(value, pointer_shape, array_dtype, "Value")

    # Handle the latency hint
    latency = require_optional_constant_int(latency)
    check_load_store_hints(latency)

    store_pointer(pointer, value, mask, latency)


def _get_scatter_value(value: Var, pointer_shape: Tuple[int, ...], array_dtype: DType,
                       value_name: str, cast_dtype: bool = True) -> Var:
    value_ty = require_tile_or_scalar_type(value)
    value_shape = value_ty.shape_value if isinstance(value_ty, TileTy) else ()

    if not is_shape_broadcastable_to(value_shape, pointer_shape):
        raise TileTypeError(f"{value_name} shape {value_shape} is not broadcastable"
                            f" to the index shape {pointer_shape}")

    if cast_dtype:
        value_dtype = get_dtype(value_ty)
        if not datatype.can_autocast_dtypes(value_dtype, array_dtype):
            raise TileTypeError(f"{value_name}'s dtype {value_dtype}"
                                f" cannot be implicitly cast to the array dtype {array_dtype}")
        value = astype(value, array_dtype)
    return broadcast_to(value, pointer_shape)


def _gather_scatter_pointer_and_mask(array: Var,
                                     indices: Var,
                                     check_bounds: Var) -> Tuple[Var, Optional[Var]]:
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
    index_shapes = [indty.shape_value if isinstance(indty, TileTy) else ()
                    for indty in index_types]
    common_shape = ()
    for shape in index_shapes:
        try:
            common_shape = broadcast_shapes2(common_shape, shape)
        except BroadcastError:
            all_shapes = ", ".join(str(s) for s in index_shapes)
            raise TileTypeError(f"Index shapes {all_shapes}"
                                f" are not broadcastable to a common shape")

    # Calculate offset from indices (and the mask, if check_bounds is True)
    array_shape = get_array_shape(array)
    array_stride = get_array_strides(array)
    offset = None
    mask = None
    for dim in range(len(index_types)):
        if isinstance(indices_ty, TupleTy):
            ind = tuple_item(indices, dim)
        else:
            ind = indices

        ind = astype(ind, datatype.uint64)
        ind = broadcast_to(ind, common_shape)

        if check_bounds:
            array_size = tuple_item(array_shape, dim)
            array_size = astype(array_size, datatype.uint64)
            dim_mask = comparison("lt", ind, array_size)
            if mask is None:
                mask = dim_mask
            else:
                mask = binary_bitwise("and_", mask, dim_mask)

        static_stride = array_ty.strides[dim].maybe_value
        if static_stride == 1:
            offset_delta = ind
        else:
            if static_stride is None:
                stride = astype(tuple_item(array_stride, dim), datatype.uint64)
            else:
                stride = loosely_typed_const(static_stride)
            offset_delta = binary_arithmetic("mul", ind, stride)

        if offset is None:
            offset = offset_delta
        else:
            offset = binary_arithmetic("add", offset, offset_delta)

    # Offset the base pointer
    base_ptr = get_array_base_ptr(array)
    if offset is None:
        # 0-D array case
        return base_ptr, None
    else:
        pointer = pointer_offset(base_ptr, offset)
        return pointer, mask


@has_side_effects
class TileAtomicCAS(TypedOperation):
    def __init__(self, pointer: Var, expected: Var, desired: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_atomic_cas",
            operands={"pointer": pointer, "expected": expected, "desired": desired, "mask": mask},
            attributes={"memory_order": memory_order, "memory_scope": memory_scope},
            result_vars=[result_var],
            loc=loc,
        )

    def generate_bytecode(self, ctx: BytecodeContext):
        result_value, _ = _atomic_cas_generate_bytecode(self, ctx)
        return result_value


@has_side_effects
class TileAtomicCASTokenOrdered(TypedOperation):
    def __init__(self, pointer: Var, expected: Var, desired: Var,
                 mask: Optional[Var], memory_order: Var, memory_scope: Var, token: Var,
                 result_var: Var, result_token: Var, loc: Loc):
        super().__init__(
            "tile_atomic_cas_token_ordered",
            operands={"pointer": pointer, "expected": expected, "desired": desired,
                      "mask": mask, "token": token},
            attributes={"memory_order": memory_order, "memory_scope": memory_scope},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        val, token = _atomic_cas_generate_bytecode(self, ctx)
        return (val,), (token,)


def _atomic_cas_generate_bytecode(op: Union[TileAtomicCAS, TileAtomicCASTokenOrdered],
                                  ctx: BytecodeContext) -> tuple[bc.Value, bc.Value]:
    return bc.encode_AtomicCASTkoOp(
        ctx.builder,
        result_type=ctx.typeid_of(op.result_vars[0]),
        result_token_type=ctx.type_table.Token,
        pointers=ctx.get_value(op.pointer),
        cmp=ctx.get_value(op.expected),
        val=ctx.get_value(op.desired),
        mask=None if op.mask is None else ctx.get_value(op.mask),
        token=ctx.get_value(op.token) if isinstance(op, TileAtomicCASTokenOrdered) else None,
        memory_ordering_semantics=memory_order_to_bytecode[op.memory_order],
        memory_scope=memory_scope_to_bytecode[op.memory_scope],
    )


@impl(ct.atomic_cas)
def atomic_cas_impl(array: Var, indices: Var, expected: Var, desired: Var, check_bounds: Var,
                    memory_order: Var, memory_scope: Var) -> Var:
    array_dtype = array.get_type().dtype
    if array_dtype not in int_float_32_64_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    # Handle the `expected` and `desired` values
    expected = _get_scatter_value(expected, pointer_shape, array_dtype, "Expected value")
    desired = _get_scatter_value(desired, pointer_shape, array_dtype, "Desired value")

    # Handle `memory_order` and `memory_scope`
    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)

    result_ty = make_tile_ty(array_dtype, pointer_shape)
    return add_operation(TileAtomicCAS, result_ty,
                         pointer=pointer, expected=expected, desired=desired, mask=mask,
                         memory_order=memory_order, memory_scope=memory_scope)


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


@has_side_effects
class TileAtomicRMW(TypedOperation):
    def __init__(self, mode: AtomicRMWMode, pointer: Var, update: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_atomic_rmw",
            attributes={"mode": mode, "memory_order": memory_order, "memory_scope": memory_scope},
            operands={"pointer": pointer, "update": update, "mask": mask, },
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        result_value, _ = _atomic_rmw_generate_bytecode(self, ctx)
        return result_value


@has_side_effects
class TileAtomicRMWTokenOrdered(TypedOperation):
    def __init__(self, mode: AtomicRMWMode, pointer: Var, update: Var,
                 mask: Optional[Var], memory_order: MemoryOrder, memory_scope: MemoryScope,
                 token: Var, result_var: Var, result_token: Var, loc: Loc):
        super().__init__(
            "tile_atomic_rmw_token_ordered",
            attributes={"mode": mode, "memory_order": memory_order, "memory_scope": memory_scope},
            operands={"pointer": pointer, "update": update, "mask": mask, "token": token},
            result_vars=[result_var, result_token],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        res, tok = _atomic_rmw_generate_bytecode(self, ctx)
        return (res,), (tok,)


def _atomic_rmw_generate_bytecode(op: Union[TileAtomicRMW, TileAtomicRMWTokenOrdered],
                                  ctx: BytecodeContext) -> Tuple[bc.Value, bc.Value]:
    return bc.encode_AtomicRMWTkoOp(
        ctx.builder,
        result_type=ctx.typeid_of(op.result_vars[0]),
        result_token_type=ctx.type_table.Token,
        pointers=ctx.get_value(op.pointer),
        arg=ctx.get_value(op.update),
        mask=None if op.mask is None else ctx.get_value(op.mask),
        token=ctx.get_value(op.token) if isinstance(op, TileAtomicRMWTokenOrdered) else None,
        memory_ordering_semantics=memory_order_to_bytecode[op.memory_order],
        memory_scope=memory_scope_to_bytecode[op.memory_scope],
        mode=op.mode._value_
    )


int_32_64_dtypes = (datatype.int32, datatype.int64, datatype.uint32, datatype.uint64)
int_float_32_64_dtypes = (*int_32_64_dtypes, datatype.float32, datatype.float64)


@impl(ct.atomic_xchg, fixed_args=[
    AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE, AtomicRMWMode.EXCHANGE,
    False, int_float_32_64_dtypes])
@impl(ct.atomic_add, fixed_args=[
    AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_INT, AtomicRMWMode.ADD_FLOAT,
    False, (*int_float_32_64_dtypes, datatype.float16)])
@impl(ct.atomic_min, fixed_args=[
    AtomicRMWMode.MIN_SIGNED_INT, AtomicRMWMode.MIN_UNSIGNED_INT, None,
    False, int_32_64_dtypes])
@impl(ct.atomic_max, fixed_args=[
    AtomicRMWMode.MAX_SIGNED_INT, AtomicRMWMode.MAX_UNSIGNED_INT, None,
    False, int_32_64_dtypes])
@impl(ct.atomic_and, fixed_args=[
    AtomicRMWMode.BITWISE_AND, AtomicRMWMode.BITWISE_AND, None,
    True, int_32_64_dtypes])
@impl(ct.atomic_or, fixed_args=[
    AtomicRMWMode.BITWISE_OR, AtomicRMWMode.BITWISE_OR, None,
    True, int_32_64_dtypes])
@impl(ct.atomic_xor, fixed_args=[
    AtomicRMWMode.BITWISE_XOR, AtomicRMWMode.BITWISE_XOR, None,
    True, int_32_64_dtypes])
def atomic_rmw_impl(int_mode: Optional[AtomicRMWMode],
                    uint_mode: Optional[AtomicRMWMode],
                    float_mode: Optional[AtomicRMWMode],
                    bitwise: bool,
                    supported_dtypes: Sequence[DType],
                    # --- end of fixed args ---
                    array: Var, indices: Var, update: Var,
                    check_bounds: Var, memory_order: Var, memory_scope: Var):
    array_dtype = array.get_type().dtype
    if array_dtype not in supported_dtypes:
        raise TileTypeError(f"Unsupported array dtype: {array_dtype}")

    if is_float(array_dtype):
        mode = float_mode
    elif is_integral(array_dtype):
        mode = int_mode if is_signed(array_dtype) else uint_mode
    else:
        mode = None
    assert mode is not None

    pointer, mask = _gather_scatter_pointer_and_mask(array, indices, check_bounds)
    pointer_ty = pointer.get_type()
    pointer_shape = pointer_ty.shape_value if isinstance(pointer_ty, TileTy) else ()

    update = _get_scatter_value(update, pointer_shape, array_dtype, "Update",
                                cast_dtype=not bitwise)
    if bitwise:
        update_dtype = get_dtype(update.get_type())
        if update_dtype != array_dtype:
            raise TileTypeError("Bitwise atomic read-modify-write operations require"
                                f" that the update dtype ({update_dtype}) exactly matches"
                                f" the array dtype ({array_dtype})")

    memory_order = require_constant_enum(memory_order, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope, MemoryScope)

    result_ty = make_tile_ty(array_dtype, pointer_shape)
    return add_operation(TileAtomicRMW, result_ty, mode=mode, pointer=pointer, update=update,
                         mask=mask, memory_order=memory_order, memory_scope=memory_scope)


class MakeToken(Operation):
    def __init__(self, result_var: Var, loc: Loc):
        super().__init__(
            "make_token",
            operands={},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return bc.encode_MakeTokenOp(ctx.builder, ctx.type_table.Token)


def make_token(*, block: Block, res: Var, loc: Loc) -> None:
    make_token_op = MakeToken(res, loc)
    block.append(make_token_op)


class JoinTokens(Operation):
    def __init__(self, tokens: Tuple[Var, ...], result_var: Var, loc: Loc):
        super().__init__(
            "join_tokens",
            operands={"tokens": tokens},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        tokens = tuple(ctx.get_value(x) for x in self.tokens)
        return bc.encode_JoinTokensOp(ctx.builder, ctx.type_table.Token, tokens)


def join_tokens(tokens: Tuple[Var, ...], *, block: Block, res: Var, loc: Loc) -> None:
    join_tokens_op = JoinTokens(tokens, res, loc)
    block.append(join_tokens_op)


class NumTiles(TypedOperation):
    def __init__(
        self, array: Var, axis: int, shape: Sequence[int], order: Sequence[int],
            result_var: Var, loc: Loc
    ):
        super().__init__(
            "num_tiles",
            operands={"array": array},
            attributes={"axis": axis, "shape": tuple(shape), "order": tuple(order)},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        pv = ctx.make_partition_view(self.array, self.order, self.shape,
                                     padding_mode=PaddingMode.UNDEFINED)
        result_types = [ctx.type_table.tile(ctx.type_table.I32, ())] * len(self.shape)
        values = bc.encode_GetIndexSpaceShapeOp(ctx.builder, result_types, pv)
        return values[self.axis]


def num_tiles(array: Var, axis: int, shape: Sequence[int], order: Sequence[int]) -> Var:
    return add_operation(NumTiles, datatype.default_int_type, array=array,
                         axis=axis, shape=shape, order=order)


@impl(ct.num_tiles)
def num_tiles_impl(array: Var, axis: Var, shape: Var, order: Var) -> Var:
    array_ty = require_array_type(array)
    axis = require_constant_int(axis)
    axis = normalize_axis(axis, array_ty.ndim)
    shape = require_constant_shape(shape, allow_single_int=True, expected_rank=array_ty.ndim,
                                   allow_0d_shape=True)
    if len(shape) == 0:
        shape = (1,) * array_ty.ndim
    order = require_constant_axis_order(order, array_ty.ndim)
    return num_tiles(array, axis, shape, order)


def full_const(shape: Sequence[int], fill_value: int | float, dtype: DType) -> Var:
    res_ty = make_tile_ty(dtype, shape)
    return strictly_typed_const(fill_value, res_ty)


def full(shape: Sequence[int], fill_value: Var, dtype: DType) -> Var:
    if fill_value.is_constant():
        return full_const(shape, fill_value.get_constant(), dtype)
    fill_value = astype(fill_value, dtype)
    return broadcast_to(fill_value, shape)


@impl(ct.full)
def full_impl(shape: Var, fill_value: Var, dtype: Var) -> Var:
    require_scalar_or_0d_tile_type(fill_value)
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full(shape, fill_value, dtype)


@impl(ct.ones)
def ones_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full_const(shape, 1, dtype)


@impl(ct.zeros)
def zeros_impl(shape: Var, dtype: Var) -> Var:
    shape = require_constant_shape(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    return full_const(shape, 0, dtype)


def _matmul_broadcast_shape(x_shape: TupleTy, y_shape: TupleTy) -> \
        Tuple[TupleTy, TupleTy, TupleTy, TupleTy]:
    x_orig_ndim = len(x_shape)
    y_orig_ndim = len(y_shape)

    # Promote 1D tensors to 2D for matmul
    if x_orig_ndim == 1:
        x_shape = TupleTy((SizeTy(1),) + x_shape.value_types)

    if y_orig_ndim == 1:
        y_shape = TupleTy(y_shape.value_types + (SizeTy(1),))

    if x_shape[-1] != y_shape[-2]:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    # Compute result matrix shape
    try:
        batch_shape = datatype.broadcast_shapes(x_shape[:-2], y_shape[:-2])
    except TypeError:
        raise TileTypeError(f"Incompatible shapes for matrix mul on tiles: {x_shape}, {y_shape}.")

    x_shape = TupleTy(batch_shape.value_types + x_shape.value_types[-2:])
    y_shape = TupleTy(batch_shape.value_types + y_shape.value_types[-2:])
    acc_shape = TupleTy(batch_shape.value_types + (x_shape[-2],) + (y_shape[-1],))

    output_shape = acc_shape
    # If x was 1D, squeeze the leading dim
    if x_orig_ndim == 1:
        output_shape = TupleTy(output_shape.value_types[:-2]
                               + output_shape.value_types[-1:])
    # If y was 1D, squeeze the trailing dim
    if y_orig_ndim == 1:
        output_shape = TupleTy(output_shape.value_types[:-1])

    return (x_shape, y_shape, acc_shape, output_shape)


class TileMma(TypedOperation):
    def __init__(self, x: Var, y: Var, acc: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_mma", operands={"x": x, "y": y, "acc": acc}, result_vars=[result_var], loc=loc
        )

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
                                    acc_value)


@impl(ct.mma)
def mma(x: Var, y: Var, acc: Var) -> Var:
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
    datatype._resolve_mma_supported_dtype(x_tile_type.dtype, y_tile_type.dtype, acc_tile_type.dtype)
    x = _promote_and_broadcast_to(x, TileTy(x_tile_type.dtype, x_shape))
    y = _promote_and_broadcast_to(y, TileTy(y_tile_type.dtype, y_shape))
    return add_operation(TileMma, acc_tile_type, x=x, y=y, acc=acc)


@impl(ct.matmul)
@impl(operator.matmul)
def matmul(x: Var, y: Var) -> Var:
    x_tile_type = require_tile_type(x)
    y_tile_type = require_tile_type(y)
    x_shape_orig = x_tile_type.shape
    y_shape_orig = y_tile_type.shape
    x_shape, y_shape, acc_shape, output_shape = _matmul_broadcast_shape(x_shape_orig, y_shape_orig)
    common_dtype = promote_dtypes(x_tile_type.dtype, y_tile_type.dtype)
    acc_dtype = datatype._resolve_mma_supported_dtype(common_dtype, common_dtype, None)
    x = _promote_and_broadcast_to(x, TileTy(common_dtype, x_shape))
    if len(y_shape_orig) == 1:
        # When y is 1d, we cannot directly use cast for reshape + broadcast
        # because y is first reshaped to 2d by appending 1.
        # Therefore, we need to first reshape y from (k,) to (k, 1) and then
        # apply the reshape+broadcast rule for batch dims
        y_shape_2d = (y_shape_orig[0].value, 1)
        y = reshape(y, y_shape_2d)
    y = _promote_and_broadcast_to(y, TileTy(common_dtype, y_shape))
    acc_ty = TileTy(acc_dtype, acc_shape)
    acc_value = strictly_typed_const(0, acc_ty)
    matmul_result = add_operation(TileMma, acc_ty, x=x, y=y, acc=acc_value)
    matmul_result = astype(matmul_result, common_dtype)
    ret = reshape(matmul_result, tuple(dim.value for dim in output_shape))
    return ret


class TileReduce(TypedOperation):
    def __init__(self, fn: str, x: Var, axis: int,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_reduce",
            operands={"x": x},
            attributes={
                "fn": fn, "axis": axis,
                "rounding_mode": rounding_mode,
                "flush_to_zero": flush_to_zero,
            },
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_type = ctx.typeof(self.x)
        x_value = ctx.get_value(self.x)
        res_type = ctx.typeof(self.result_var)
        return lower_reduce(
            ctx, x_value, x_type, self.axis, res_type, self.fn,
            self.rounding_mode, self.flush_to_zero
        )


def _get_reduction_shape(shape: Tuple[int, ...],
                         normalized_axis: int | Tuple[int, ...] | None,
                         keepdims: bool) -> Tuple[int, ...]:
    if normalized_axis is None:
        normalized_axis = tuple(range(len(shape)))
    if isinstance(normalized_axis, int):
        normalized_axis = (normalized_axis,)
    normalized_axis = set(normalized_axis)
    ret = []
    for i, size in enumerate(shape):
        if i in normalized_axis:
            if keepdims:
                ret.append(1)
        else:
            ret.append(size)
    return tuple(ret)


def reduce(fn: str, x: Var, axis: Optional[tuple[int, ...]], keepdims: bool,
           rounding_mode: Optional[RoundingMode] = None,
           flush_to_zero: bool = False) -> Var:
    x_type = require_tile_type(x)
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)
    x_shape = x_type.shape
    rank = len(x_shape)
    if axis is None:
        axis = tuple(range(rank))
    else:
        axis = tuple([normalize_axis(axis_value, rank) for axis_value in axis])

    x_dtype = datatype.default_int_type if datatype.is_boolean(x_type.dtype) else x_type.dtype
    x = _promote_and_broadcast_to(x, TileTy(x_dtype, x_shape))
    for i, axis_value in enumerate(axis):
        axis_value -= i
        x_shape = x_shape[:axis_value] + x_shape[axis_value + 1:]
        x = add_operation(
            TileReduce, TileTy(x_dtype, TupleTy(x_shape)),
            fn=fn, x=x, axis=axis_value,
            rounding_mode=rounding_mode, flush_to_zero=flush_to_zero
        )
    return reshape(x, _get_reduction_shape(x_type.shape_value, axis, keepdims))


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
def reduce_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, keepdims: Var, rounding_mode: Var,
                                flush_to_zero: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return reduce(fn, x, axis, keepdims, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


@impl(ct.max, fixed_args=["max"])
@impl(ct.min, fixed_args=["min"])
def reduce_impl_with_ftz(fn: str, x: Var, axis: Var, keepdims: Var, flush_to_zero: Var) -> Var:
    axis = _parse_reduce_axis(axis)
    keepdims = require_constant_bool(keepdims)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return reduce(fn, x, axis, keepdims, flush_to_zero=flush_to_zero)


class TileArgReduce(TypedOperation):
    def __init__(self, fn: str, x: Var, axis: Optional[int],
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_arg_reduce",
            operands={"x": x},
            attributes={"fn": fn, "axis": axis},
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_type = ctx.typeof(self.x)
        x_value = ctx.get_value(self.x)
        res_type = ctx.typeof(self.result_var)
        return lower_reduce_argmax_argmin(
            ctx, x_value, x_type, self.axis, res_type, self.fn
        )


def argreduce(fn: str, x: Var, axis: Optional[int], keepdims: bool) -> Var:
    x_type = require_tile_type(x)
    x_shape = x_type.shape
    if axis is not None:
        axis = normalize_axis(axis, len(x_shape))

    x_dtype = datatype.default_int_type if datatype.is_boolean(x_type.dtype) else x_type.dtype
    x = _promote_and_broadcast_to(x, TileTy(x_dtype, x_shape))
    output_dtype = datatype.int64
    output_shape = TupleTy([]) if axis is None else TupleTy(x_shape[:axis] + x_shape[axis + 1:])
    x = add_operation(
        TileArgReduce, TileTy(output_dtype, output_shape),
        fn=fn, x=x, axis=axis
    )
    return reshape(x, _get_reduction_shape(x_type.shape_value, axis, keepdims))


@impl(ct.argmax, fixed_args=["argmax"])
@impl(ct.argmin, fixed_args=["argmin"])
def argreduce_impl(fn: str, x: Var, axis: Var, keepdims: Var) -> Var:
    axis = require_optional_constant_int(axis)
    keepdims = require_constant_bool(keepdims)
    return argreduce(fn, x, axis, keepdims)


class TileScan(TypedOperation):
    def __init__(self, fn: str, x: Var, axis: int, reverse: bool,
                 rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_scan",
            operands={"x": x},
            attributes={
                "fn": fn, "axis": axis, "reverse": reverse,
                "rounding_mode": rounding_mode, "flush_to_zero": flush_to_zero,
            },
            result_vars=[result_var],
            loc=loc,
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        x_type = ctx.typeof(self.x)
        x_value = ctx.get_value(self.x)
        res_type = ctx.typeof(self.result_var)
        return lower_scan(ctx, x_value, x_type, self.axis, self.reverse,
                          res_type, self.fn, self.rounding_mode, self.flush_to_zero)


def scan(fn: str, x: Var, axis: int, reverse: bool, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> Var:
    x_type = require_tile_type(x)
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, x_type.dtype)
    x_shape = x_type.shape
    axis = normalize_axis(axis, len(x_shape))
    x_dtype = datatype.default_int_type if datatype.is_boolean(x_type.dtype) else x_type.dtype
    x = _promote_and_broadcast_to(x, TileTy(x_dtype, x_shape))
    return add_operation(
        TileScan, TileTy(x_dtype, x_shape),
        fn=fn, x=x, axis=axis, reverse=reverse,
        rounding_mode=rounding_mode, flush_to_zero=flush_to_zero
    )


@impl(ct.cumsum, fixed_args=["add"])
@impl(ct.cumprod, fixed_args=["mul"])
def scan_impl_with_rd_and_ftz(fn: str, x: Var, axis: Var, reverse: Var,
                              rounding_mode: Var, flush_to_zero: Var) -> Var:
    axis = require_constant_int(axis)
    reverse = require_constant_bool(reverse)
    rounding_mode = require_optional_constant_enum(rounding_mode, RoundingMode)
    flush_to_zero = require_constant_bool(flush_to_zero)
    return scan(fn, x, axis, reverse, rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def expand_dims(x: Var, axis: int) -> Var:
    x_ty = require_tile_type(x)
    axis = normalize_axis(axis, x_ty.ndim + 1)
    old_shape = x_ty.shape_value
    new_shape = (*old_shape[:axis], 1, *old_shape[axis:])
    res_type = make_tile_ty(x_ty.dtype, new_shape)
    return add_operation(TileReshape, res_type, x=x)


@impl(ct.expand_dims)
def expand_dims_impl(x: Var, axis: Var) -> Var:
    axis = require_constant_int(axis)
    return expand_dims(x, axis)


class TileCat(TypedOperation):
    def __init__(self, tiles: Var, axis: int, result_var: Var, loc: Loc):
        super().__init__(
            "tile_cat",
            operands={"tiles": tiles},
            attributes={"axis": axis},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return_type_id = ctx.typeid_of(self.result_var)
        x_value, y_value = ctx.get_value_tuple(self.tiles)
        return bc.encode_CatOp(ctx.builder, return_type_id, x_value, y_value, self.axis)


def cat(tiles: Var, axis: int) -> Var:
    tuple_ty = require_tuple_type(tiles)
    if len(tuple_ty) == 0:
        raise TileTypeError("cat() received an empty tuple")
    if not isinstance(first_tile := tuple_ty.value_types[0], TileTy):
        raise TileTypeError(f"Expected tuple of Tile, got a {first_tile}")
    dtype = first_tile.dtype
    rank = first_tile.ndim
    shape_value = list(first_tile.shape_value)
    axis = normalize_axis(axis, rank)
    for tile_ty in tuple_ty.value_types[1:]:
        if not isinstance(tile_ty, TileTy):
            raise TileTypeError(f"Expected tuple of Tile, got a {tile_ty}")
        if tile_ty.ndim != rank:
            raise TileTypeError(f"Expected tiles to have the same rank: {rank} != {tile_ty.ndim}")
        if tile_ty.dtype != dtype:
            raise TileTypeError(f"Expected tiles to have the same dtype: {dtype} != {tile_ty.dtype}")  # noqa: E501
        for i, (x, y) in enumerate(zip(shape_value, tile_ty.shape_value, strict=True)):
            if i != axis and x != y:
                raise TileTypeError("Expected tiles to have the same "
                                    "shape for non axis dimensions")
        shape_value[axis] += tile_ty.shape_value[axis]

    if not all(_is_power_of_2(x) for x in shape_value):
        raise TileTypeError(f"Result tile shape must be power of 2, got: {shape_value}")

    if len(tuple_ty) == 1:
        return tuple_item(tiles, 0)
    res_ty = make_tile_ty(dtype, shape_value)
    return add_operation(TileCat, res_ty, tiles=tiles, axis=axis)


def _is_power_of_2(x: int):
    if x <= 0:
        return False
    return x & (x - 1) == 0


@impl(ct.cat)
def cat_impl(tiles: Var, axis: Var) -> Var:
    const_axis = require_constant_int(axis)
    return cat(tiles, const_axis)


# Does not support broadcasting or type promotion
class RawWhereOperation(TypedOperation):
    def __init__(self, cond, x, y, result_var: Var, loc: Loc):
        super().__init__(
            "raw_where", operands={"cond": cond, "x": x, "y": y}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_typeid = ctx.typeid_of(self.result_var)
        cond = ctx.get_value(self.cond)
        x = ctx.get_value(self.x)
        y = ctx.get_value(self.y)
        return bc.encode_SelectOp(ctx.builder, res_typeid, cond, x, y)


def raw_where(cond: Var, x: Var, y: Var) -> Var:
    ty = x.get_type()
    assert ty == y.get_type()
    assert change_dtype(cond.get_type(), get_dtype(ty)) == ty
    return add_operation(RawWhereOperation, ty, cond=cond, x=x, y=y)


@impl(ct.where)
def where(cond, x, y) -> Var:
    cond_ty = require_tile_or_scalar_maybe_loose_type(cond)
    x_ty = require_tile_or_scalar_maybe_loose_type(x)
    y_ty = require_tile_or_scalar_maybe_loose_type(y)

    xy_ty = promote_types(x_ty, y_ty)
    dtype = get_dtype(xy_ty)
    cond_like_ty = change_dtype(cond_ty, dtype)
    res_ty = promote_types(cond_like_ty, xy_ty)

    cond = _promote_and_broadcast_to(cond, change_dtype(res_ty, datatype.bool_))
    x = _promote_and_broadcast_to(x, res_ty)
    y = _promote_and_broadcast_to(y, res_ty)
    return raw_where(cond, x, y)


@has_side_effects
class TilePrintf(TypedOperation):
    def __init__(self, format: str, args: Tuple[Var, ...], loc: Loc):
        super().__init__("tile_printf",
                         operands={"args": args},
                         attributes={"format": format},
                         result_vars=[],
                         loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        arg_vars = [ctx.get_value(arg) for arg in self.args]
        with tile_mutex("print_mutex", ctx):
            bc.encode_PrintOp(ctx.builder, arg_vars, self.format)
        return []


@impl(ct.printf)
def printf_impl(format: Var, args: Tuple[Var, ...]) -> None:
    format_str = require_constant_str(format)
    arg_types = tuple(require_tile_or_scalar_type(x) for x in args)
    parsed_format = PrintfValidator.parse_format(format_str, arg_types)
    add_operation(TilePrintf, (), format=parsed_format, args=args)


@has_side_effects
class TileAssert(TypedOperation):
    def __init__(self, cond: Var, message: str, loc: Loc):
        super().__init__("assert",
                         operands={"cond": cond},
                         attributes={"message": message},
                         result_vars=[], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        bc.encode_AssertOp(ctx.builder, ctx.get_value(self.cond), self.message)
        return []


@impl(ct.assert_)
def assert_impl(cond: Var, message: Var) -> None:
    ty = require_tile_or_scalar_type(cond)
    if get_dtype(ty) != datatype.bool_:
        raise TileTypeError(f"Type of condition must be bool, got {ty}")
    msg_str = require_optional_constant_str(message)
    msg_str = "" if msg_str is None else msg_str
    add_operation(TileAssert, (), cond=cond, message=msg_str)


# Covert a scalar to 0d tile
class ScalarToTile(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__("scalar_to_tile", operands={"x": x}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.get_value(self.x)


def scalar_to_tile(x: Var) -> Var:
    ty = require_scalar_or_0d_tile_type(x)
    if isinstance(ty, TileTy):
        return x
    else:
        res_ty = make_tile_ty(ty, ())
        return add_operation(ScalarToTile, res_ty, x=x)


class TileBroadcast(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__("tile_broadcast", operands={"x": x}, result_vars=[result_var], loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_BroadcastOp(ctx.builder, res_typeid, x_value)


def broadcast_to(x: Var, shape: Sequence[int], *, keep_scalar: bool = False):
    x_ty = require_tile_or_scalar_type(x)
    if isinstance(x_ty, TileTy):
        old_shape = x_ty.shape_value
    else:
        old_shape = ()

    if not is_shape_broadcastable_to(old_shape, shape):
        raise TileTypeError(f"Shape {old_shape} is not broadcastable to {tuple(shape)}")

    if len(shape) > len(old_shape):
        extra_ones = (1,) * (len(shape) - len(old_shape))
        old_shape = extra_ones + old_shape
        x = reshape(x, old_shape)

    if old_shape == shape:
        if len(shape) == 0 and not keep_scalar:
            return scalar_to_tile(x)
        else:
            return x
    else:
        result_ty = make_tile_ty(get_dtype(x_ty), shape)
        return add_operation(TileBroadcast, result_ty, x=x)


@impl(ct.broadcast_to)
def broadcast_to_impl(x: Var, shape: Var) -> Var:
    require_tile_type(x)
    shape = require_constant_shape(shape)
    return broadcast_to(x, shape)


class TileAsType(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_astype", operands={"x": x}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return convert_dtype(ctx, value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def astype(x: Var, dtype: DType) -> Var:
    x_ty = require_tile_or_scalar_type(x)
    from_dtype = get_dtype(x_ty)
    if from_dtype == dtype:
        return x

    if x.is_constant():
        val = dtype._py_type(x.get_constant())
        return strictly_typed_const(val, dtype)

    result_ty = dtype if isinstance(x_ty, DType) else make_tile_ty(dtype, x_ty.shape_value)
    return add_operation(TileAsType, result_ty, x=x)


@impl(ct.astype)
def astype_impl(x: Var, dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    return astype(x, dtype)


def tile_astype(x: Var, dtype: Var,
                block: Block, loc: Loc, res: Var) -> None:
    tile_astype_op = TileAsType(x, dtype, res, loc)
    block.append(tile_astype_op)


class TileBitCast(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_bitcast",
            operands={"x": x},
            attributes={},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return ctx.bitcast(value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def bitcast(x: Var, dtype: DType) -> Var:
    tile_ty = require_tile_type(x)
    x_dtype = tile_ty.dtype
    if x_dtype.bitwidth != dtype.bitwidth:
        raise TileTypeError(f"Cannot bitcast from {x_dtype} to {dtype}: "
                            f"bit width is different ({x_dtype.bitwidth} vs. {dtype.bitwidth}")
    res_ty = make_tile_ty(dtype, tile_ty.shape_value)
    return add_operation(TileBitCast, res_ty, x=x)


@impl(ct.bitcast)
def bitcast_impl(x: Var, dtype: Var) -> Var:
    dtype_val = require_dtype_spec(dtype)
    return bitcast(x, dtype_val)


class TileArange(TypedOperation):
    def __init__(self, result_var: Var, loc: Loc):
        super().__init__(
            "tile_arange",
            operands={},
            attributes={},
            result_vars=[result_var],
            loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_type = ctx.typeid_of(self.result_var)
        return bc.encode_IotaOp(ctx.builder, res_type)


def arange(size: int, dtype: DType) -> Var:
    res_ty = make_tile_ty(dtype, (size,))
    return add_operation(TileArange, res_ty)


@impl(ct.arange)
def arange_impl(size: Var, dtype: Var) -> Var:
    size_val = require_constant_int(size)
    dtype_val = require_dtype_spec(dtype)
    if not _is_power_of_2(size_val):
        raise TileTypeError(f"Result tile shape must be power of 2, got {size_val}")
    return arange(size_val, dtype_val)


class TileReshape(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_reshape", operands={"x": x}, result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ReshapeOp(ctx.builder, res_type_id, x_value)


def reshape(x: Var, new_shape: Tuple[int, ...]) -> Var:
    x_ty = require_tile_or_scalar_type(x)
    x_shape = x_ty.shape_value if isinstance(x_ty, TileTy) else ()
    numel = math.prod(x_shape)

    negative_one_index = None
    numel2 = 1
    for i, dim_value in enumerate(new_shape):
        if dim_value < 0:
            if dim_value < -1:
                raise TileTypeError(f"Dimension can only be -1 or non-negative, got {dim_value}")
            if negative_one_index is not None:
                raise TileTypeError(f"Only one dimension can be -1, got {new_shape}")
            negative_one_index = i
        else:
            numel2 *= dim_value

    if negative_one_index is not None:
        if numel2 == 0 or numel % numel2 != 0:
            raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")
        new_shape = list(new_shape)
        new_shape[negative_one_index] = numel // numel2
        new_shape = tuple(new_shape)
    elif numel != numel2:
        raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")

    if new_shape == x_shape:
        return x
    else:
        res_type = make_tile_ty(get_dtype(x_ty), new_shape)
        return add_operation(TileReshape, res_type, x=x)


@impl(ct.reshape)
def reshape_impl(x: Var, shape: Var) -> Var:
    require_tile_type(x)
    new_shape = require_constant_int_tuple(shape)
    return reshape(x, new_shape)


class TilePermute(TypedOperation):
    def __init__(self, x: Var, axes: Sequence[int], result_var: Var, loc: Loc):
        super().__init__(
            "tile_permute",
            operands={"x": x},
            attributes={"axes": axes},
            result_vars=[result_var],
            loc=loc)

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        ret_ty_id = ctx.typeid_of(self.result_var)
        x_value = ctx.get_value(self.x)
        return bc.encode_PermuteOp(ctx.builder, ret_ty_id, x_value, self.axes)


def permute(x: Var, axes: Sequence[int]) -> Var:
    ty = require_tile_type(x)
    axes = tuple(normalize_axis(ax, ty.ndim) for ax in axes)
    shape = tuple(ty.shape_value[i] for i in axes)
    result_ty = make_tile_ty(ty.dtype, shape)
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


class TileExtract(TypedOperation):
    def __init__(self, x: Var, index: Var, shape: Sequence[int],
                 result_var: Var, loc: Loc):
        super().__init__(
            "tile_extract",
            operands={"x": x, "index": index},
            attributes={"shape": tuple(shape)},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        index = ctx.get_value_tuple(self.index)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ExtractOp(ctx.builder, res_type_id, x_value, index)


def extract(x: Var, index: Var, shape: Sequence[int]) -> Var:
    dtype = get_dtype(x.get_type())
    res_ty = make_tile_ty(dtype, shape)
    return add_operation(TileExtract, res_ty, x=x, index=index, shape=shape)


@impl(ct.extract)
def extract_impl(x: Var, index: Var, shape: Var) -> Var:
    x_ty = require_tile_type(x)
    shape = require_constant_shape(shape, expected_rank=x_ty.ndim, allow_single_int=True,
                                   allow_0d_shape=True)
    orig_shape = shape
    if len(shape) == 0:
        shape = (1,) * x_ty.ndim

    index_ty = require_index_or_index_tuple_type(index)
    index_len = len(index_ty) if isinstance(index_ty, TupleTy) else 1
    if x_ty.ndim != index_len:
        raise TileTypeError(f"Index size {index_len} does not match the tile rank {x_ty.ndim}")

    for i, (s1, s2) in enumerate(zip(x_ty.shape_value, shape, strict=True)):
        if s2 == 0:
            raise TileTypeError(f"Zero shape at dimension #{i}: {shape}")
        if s1 % s2 != 0:
            raise TileTypeError(f"Input shape {x_ty.shape_value} is not divisible by"
                                f" result shape {shape} at dimension #{i}")
    result = extract(x, index, shape)
    return reshape(result, orig_shape)


class TileItem(TypedOperation):
    def __init__(self, x: Var, result_var: Var, loc: Loc):
        super().__init__(
            "tile_item", operands={"x": x},
            result_vars=[result_var], loc=loc
        )

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        x_value = ctx.get_value(self.x)
        x_type_id = ctx.typeid_of(self.x)
        res_type_id = ctx.typeid_of(self.result_var)
        if res_type_id == x_type_id:
            return x_value
        else:
            return bc.encode_ReshapeOp(ctx.builder, res_type_id, x_value)


@impl(ct._m_tile_item)
def tile_item(tile: Var) -> Var:
    x_ty = require_tile_type(tile)
    if x_ty.numel != 1:
        raise TileTypeError(f"Expected a tile of size 1 to get scalar item, "
                            f"got a tile of size {x_ty.numel}")
    return add_operation(TileItem, x_ty.dtype, x=tile)


LoadMemoryOperation = TileLoad | LoadPointer

StoreMemoryOperation = TileStore | StorePointer | TileAtomicCAS | TileAtomicRMW

MemoryOperation = LoadMemoryOperation | StoreMemoryOperation
