# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal, Sequence, get_type_hints

from cuda.lang._compiler_options import CompilerOptions
from cuda.lang._enums import VectorReduction
from cuda.lang._ir import ir, ops
from cuda.lang import _mlir as mlir
from cuda.tile._memory_model import MemoryOrder, MemoryScope
from cuda.lang._mlir._builtins import _Cursor
import cuda.lang._mlir.extras.types as T
import cuda.lang._ir.type as ir_type
from cuda.lang.compilation import KernelSignature
import cuda.lang._datatype as datatype
from cuda.tile._datatype import PointerInfo, is_integral
from cuda.lang._exception import InternalError, TypeCheckingError
from .type_conversion import (
    ir_type_to_mlir_type,
    mlir_constant_of_type,
    convert_dtype,
    dtype_to_mlir_type,
)


def _expect_arith_type(ty: ir_type.Type) -> ir_type.TensorLikeTy:
    assert isinstance(ty, ir_type.TensorLikeTy)
    assert datatype.is_arithmetic(ty.tensor_dtype())
    return ty


def _expect_pointer_type(ty: ir_type.Type) -> ir_type.PointerTy:
    assert isinstance(ty, ir_type.PointerTy)
    return ty


def _get_llvm_memory_ordering(mo: None | MemoryOrder):
    match mo:
        case None | MemoryOrder.WEAK:
            return mlir.llvm.AtomicOrdering.not_atomic
        case MemoryOrder.ACQUIRE:
            return mlir.llvm.AtomicOrdering.acquire
        case MemoryOrder.ACQ_REL:
            return mlir.llvm.AtomicOrdering.acq_rel
        case MemoryOrder.RELAXED:
            return mlir.llvm.AtomicOrdering.monotonic
        case MemoryOrder.RELEASE:
            return mlir.llvm.AtomicOrdering.release

    raise NotImplementedError(f"Unhandled {mo=}")


def _get_llvm_syncscope(memory_scope: MemoryScope) -> str | None:
    match memory_scope:
        case MemoryScope.BLOCK:
            return "block"
        case MemoryScope.CLUSTER:
            return "cluster"
        case MemoryScope.DEVICE:
            return "device"
        case MemoryScope.SYS:
            return None

    raise NotImplementedError(f"Unhandled {memory_scope=}")


def _get_llvm_cmpxchg_failure_ordering(memory_order: MemoryOrder):
    match memory_order:
        case MemoryOrder.ACQUIRE | MemoryOrder.ACQ_REL:
            return mlir.llvm.AtomicOrdering.acquire
        case MemoryOrder.RELAXED | MemoryOrder.RELEASE:
            return mlir.llvm.AtomicOrdering.monotonic
        case _:
            raise NotImplementedError(f"Unhandled {memory_order=}")


def _get_llvm_atomic_binop(
    kind: ops.AtomicRMWKind, dtype: datatype.DType
) -> mlir.llvm.AtomicBinOp:
    signed = datatype.is_signed(dtype)
    is_float = datatype.is_float(dtype)
    match kind:
        case ops.AtomicRMWKind.ADD:
            return mlir.llvm.AtomicBinOp.fadd if is_float else mlir.llvm.AtomicBinOp.add
        case ops.AtomicRMWKind.SUB:
            return mlir.llvm.AtomicBinOp.fsub if is_float else mlir.llvm.AtomicBinOp.sub
        case ops.AtomicRMWKind.AND:
            return mlir.llvm.AtomicBinOp._and
        case ops.AtomicRMWKind.OR:
            return mlir.llvm.AtomicBinOp._or
        case ops.AtomicRMWKind.XOR:
            return mlir.llvm.AtomicBinOp._xor
        case ops.AtomicRMWKind.MIN:
            if is_float:
                return mlir.llvm.AtomicBinOp.fminimum
            return mlir.llvm.AtomicBinOp.min if signed else mlir.llvm.AtomicBinOp.umin
        case ops.AtomicRMWKind.MAX:
            if is_float:
                return mlir.llvm.AtomicBinOp.fmaximum
            return mlir.llvm.AtomicBinOp.max if signed else mlir.llvm.AtomicBinOp.umax
        case ops.AtomicRMWKind.INC:
            return mlir.llvm.AtomicBinOp.uinc_wrap
        case ops.AtomicRMWKind.DEC:
            return mlir.llvm.AtomicBinOp.udec_wrap
        case _:
            raise NotImplementedError(f"Unsupported atomic {kind=} for {dtype=}")


@dataclass(frozen=True)
class _OpForDType:
    bool_op: mlir.Operation | None
    signed_integral_op: mlir.Operation | None
    unsigned_integral_op: mlir.Operation | None
    float_op: mlir.Operation | None


def _add_float_floordiv_op(lhs: mlir.Value, rhs: mlir.Value) -> mlir.Value:
    quotient = mlir.arith.add_DivFOp(lhs=lhs, rhs=rhs)
    return mlir.add_operation(
        name="math.floor",
        result_type=quotient.type,
        operands=(quotient,),
        properties=(),
    )


def _get_mlir_op_for_op_and_dtype(
    fn: str, dtype: datatype.DType
) -> Callable[[mlir.Value, mlir.Value], mlir.Value] | None:
    _OPERATIONS = {
        "floordiv": _OpForDType(
            None,
            mlir.arith.add_FloorDivSIOp,
            mlir.arith.add_DivUIOp,
            _add_float_floordiv_op,
        ),
        "cdiv": _OpForDType(
            None,
            mlir.arith.add_CeilDivSIOp,
            mlir.arith.add_CeilDivUIOp,
            None,
        ),
        "add": _OpForDType(
            mlir.arith.add_OrIOp,
            mlir.arith.add_AddIOp,
            mlir.arith.add_AddIOp,
            mlir.arith.add_AddFOp,
        ),
        "sub": _OpForDType(
            None,
            mlir.arith.add_SubIOp,
            mlir.arith.add_SubIOp,
            mlir.arith.add_SubFOp,
        ),
        "mul": _OpForDType(
            mlir.arith.add_AndIOp,
            mlir.arith.add_MulIOp,
            mlir.arith.add_MulIOp,
            mlir.arith.add_MulFOp,
        ),
        "truediv": _OpForDType(
            None,
            mlir.arith.add_DivSIOp,
            mlir.arith.add_DivUIOp,
            mlir.arith.add_DivFOp,
        ),
        "xor": _OpForDType(
            mlir.arith.add_XOrIOp,
            mlir.arith.add_XOrIOp,
            mlir.arith.add_XOrIOp,
            None,
        ),
        "or_": _OpForDType(
            mlir.arith.add_OrIOp,
            mlir.arith.add_OrIOp,
            mlir.arith.add_OrIOp,
            None,
        ),
        "and_": _OpForDType(
            mlir.arith.add_AndIOp,
            mlir.arith.add_AndIOp,
            mlir.arith.add_AndIOp,
            None,
        ),
        "c_mod": _OpForDType(
            None,
            mlir.arith.add_RemSIOp,
            mlir.arith.add_RemUIOp,
            None,
        ),
    }

    if fn not in _OPERATIONS:
        return None

    op_for_dtype = _OPERATIONS[fn]

    if datatype.is_boolean(dtype):
        return op_for_dtype.bool_op
    elif datatype.is_integral(dtype):
        if datatype.is_signed(dtype):
            return op_for_dtype.signed_integral_op
        else:
            return op_for_dtype.unsigned_integral_op
    elif datatype.is_float(dtype):
        return op_for_dtype.float_op
    else:
        return None


def _get_mlir_unary_op_for_op_and_type(
    fn: str,
    typ: ir_type.TensorLikeTy,
) -> Callable[[mlir.Value], mlir.Value] | None:
    dtype = typ.tensor_dtype()
    match fn:
        case "pos":
            return lambda operand: operand
        case "invert" if datatype.is_boolean(dtype):
            return _invert_boolean
        case "invert" if datatype.is_integral(dtype):
            return partial(_invert_int, dtype)
        case 'neg' if datatype.is_float(dtype):
            return mlir.arith.add_NegFOp
        case "neg" if datatype.is_integral(dtype):
            mlir_type = ir_type_to_mlir_type(typ)
            zero = mlir_constant_of_type(mlir_type, 0)
            return lambda operand: mlir.arith.add_SubIOp(lhs=zero, rhs=operand)


def _invert_boolean(operand):
    mlir_bool = dtype_to_mlir_type(datatype.bool_)
    false = mlir_constant_of_type(mlir_bool, 0)
    cmp = mlir.arith.add_CmpIOp(
        predicate=mlir.arith.CmpIPredicate.eq, lhs=operand, rhs=false
    )
    cmp = mlir.arith.add_ExtUIOp(out_type=mlir_bool, in_=cmp)
    return cmp


def _invert_int(dtype, operand):
    assert is_integral(dtype)
    mlir_dtype = dtype_to_mlir_type(dtype)
    all_ones = (1 << dtype.bitwidth) - 1
    all_ones_mlir = mlir_constant_of_type(mlir_dtype, all_ones)
    return mlir.arith.add_XOrIOp(lhs=operand, rhs=all_ones_mlir)


def _get_mlir_comparison_op(
    fn: str, dtype: datatype.DType
) -> Callable[[mlir.Value, mlir.Value], mlir.Value] | None:
    if datatype.is_float(dtype):
        match fn:
            case "lt":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OLT
                )
            case "le":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OLE
                )
            case "gt":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OGT
                )
            case "ge":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OGE
                )
            case "eq":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.OEQ
                )
            case "ne":
                return partial(
                    mlir.arith.add_CmpFOp, predicate=mlir.arith.CmpFPredicate.ONE
                )
            case _:
                return None
    else:
        signed = datatype.is_signed(dtype)
        match fn:
            case "lt":
                pred = (
                    mlir.arith.CmpIPredicate.slt
                    if signed
                    else mlir.arith.CmpIPredicate.ult
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "le":
                pred = (
                    mlir.arith.CmpIPredicate.sle
                    if signed
                    else mlir.arith.CmpIPredicate.ule
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "gt":
                pred = (
                    mlir.arith.CmpIPredicate.sgt
                    if signed
                    else mlir.arith.CmpIPredicate.ugt
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "ge":
                pred = (
                    mlir.arith.CmpIPredicate.sge
                    if signed
                    else mlir.arith.CmpIPredicate.uge
                )
                return partial(mlir.arith.add_CmpIOp, predicate=pred)
            case "eq":
                return partial(
                    mlir.arith.add_CmpIOp, predicate=mlir.arith.CmpIPredicate.eq
                )
            case "ne":
                return partial(
                    mlir.arith.add_CmpIOp, predicate=mlir.arith.CmpIPredicate.ne
                )
            case _:
                return None


# These operations have aggregate results. The RHS's elements are stored to
# the LHS's when lowering Assign operations and are no-ops at the MLIR level.
_NOOP_LOWERINGS = frozenset([
    ops.ReinterpretPointerAsArray,
])


@dataclass(kw_only=True)
class MLIRLoweringContext:
    """Mutable data shared by host and device MLIR operation lowering."""

    execution_space: ClassVar[Literal["host", "device"]]
    region: ir.Region
    ir_context: ir.IRContext
    var_map: dict[str, mlir.Value] = field(default_factory=dict)
    block_map: dict[ir.Block, mlir.Block] = field(default_factory=dict)
    module_op: mlir.Operation | None = None
    func_op: mlir.Operation | None = None
    current_op: ir.Operation | None = None

    def get_var(self, var: ir.Var) -> mlir.Value:
        try:
            return self.var_map[var.name]
        except KeyError:
            raise InternalError(f"Variable {var.name} not found") from None

    def is_defined(self, var: ir.Var) -> bool:
        return var.name in self.var_map

    def def_var(self, var: ir.Var, value: mlir.Value) -> None:
        if self.is_defined(var):
            raise InternalError(f"Variable {var.name} is already defined")
        self.var_map[var.name] = value

    def bind_aggregate(self, dst: ir.Var, src: ir.Var) -> None:
        assert dst.is_aggregate() == src.is_aggregate()
        assert dst.is_aggregate(), (
            f"Expected aggregate operands, got dst={dst.get_type()}, src={src.get_type()}"
        )

        dst_items = dst.get_aggregate().as_tuple()
        src_items = src.get_aggregate().as_tuple()
        assert len(dst_items) == len(src_items), (
            "Aggregate alias shape mismatch while lowering to MLIR: "
            f"{dst.name} has {len(dst_items)} items but {src.name} has {len(src_items)}"
        )

        for dst_item, src_item in zip(dst_items, src_items, strict=True):
            assert dst_item.is_aggregate() is src_item.is_aggregate()
            if dst_item.is_aggregate():
                self.bind_aggregate(dst_item, src_item)
            elif not self.is_defined(dst_item):
                self.def_var(dst_item, self.get_var(src_item))

        if self.is_defined(src) and not self.is_defined(dst):
            self.def_var(dst, self.get_var(src))

    @property
    def function_region(self) -> mlir.Region:
        assert self.func_op is not None, "MLIR function not initialized"
        return self.func_op.regions[0]


@dataclass(kw_only=True)
class DeviceLoweringContext(MLIRLoweringContext):
    """Data required while lowering a CUDA Lang device function."""

    execution_space = "device"
    signature: KernelSignature
    gpu_module_op: mlir.Operation | None = None
    seen_foreign_functions: dict[
        str, mlir.llvm.LLVMFunctionType
    ] = field(default_factory=dict)

    @property
    def gpu_module(self) -> mlir.Operation:
        assert self.gpu_module_op is not None, "MLIR GPU module not initialized"
        return self.gpu_module_op


class _MLIROperationLoweringRegistry:
    """Mapping from an IR operation and execution space to its MLIR lowering."""

    def __init__(self):
        self._handlers = {"host": {}, "device": {}}

    def register(self, *, host: bool = True, device: bool = True):
        if not host and not device:
            raise ValueError("an MLIR lowering must support host, device, or both")

        def decorator(handler):
            try:
                operation_type = get_type_hints(handler)["operation"]
            except KeyError:
                raise InternalError(
                    "an MLIR lowering must annotate its operation parameter"
                )
            for exec_space, active in (("host", host), ("device", device)):
                if not active:
                    continue
                registry = self._handlers[exec_space]
                if operation_type in registry:
                    raise InternalError(f"{operation_type} already "
                                        f"has lowering for {exec_space} context")
                registry[operation_type] = handler
            return handler

        return decorator

    def lower(
        self, context: MLIRLoweringContext, operation: ir.Operation
    ) -> Sequence[mlir.Value] | None:
        try:
            handler = self._handlers[context.execution_space][type(operation)]
        except KeyError:
            raise NotImplementedError(f"Unable to lower {operation=} to MLIR") from None
        return handler(context, operation)


_MLIR_OP_LOWERING = _MLIROperationLoweringRegistry()


def mlir_op_lowering(handler=None, *, host: bool = True, device: bool = True):
    decorator = _MLIR_OP_LOWERING.register(host=host, device=device)
    return decorator if handler is None else decorator(handler)


@mlir_op_lowering
def lower_comparison(
    context: MLIRLoweringContext, operation: ops.RawComparisonOperation
) -> Sequence[mlir.Value]:
    lhs_type = operation.lhs.get_type()
    rhs_type = operation.lhs.get_type()
    assert lhs_type == rhs_type

    dtype = _expect_arith_type(lhs_type).tensor_dtype()
    res_type = _expect_arith_type(operation.result_var.get_type())
    mlir_op = _get_mlir_comparison_op(operation.fn, dtype)
    if mlir_op is None:
        raise NotImplementedError(
            f"Comparison operation {operation.fn} not supported for {dtype}"
        )

    result = mlir_op(
        lhs=context.get_var(operation.lhs),
        rhs=context.get_var(operation.rhs),
    )
    result = mlir.arith.add_ExtUIOp(
        out_type=ir_type_to_mlir_type(res_type),
        in_=result,
    )
    return [result]


@mlir_op_lowering
def lower_raw_unary_arith(
    context: MLIRLoweringContext, operation: ops.Unary
) -> Sequence[mlir.Value]:
    res_type = _expect_arith_type(operation.result_var.get_type())
    mlir_op = _get_mlir_unary_op_for_op_and_type(operation.fn, res_type)
    if mlir_op is None:
        raise NotImplementedError(
            f"Arithmetic operation '{operation.fn}' not supported for {res_type=}"
        )
    return [mlir_op(operand=context.get_var(operation.operand))]


def _lower_raw_binary_arith(
    context: MLIRLoweringContext,
    operation: ops.RawBinaryArithmeticOperation | ops.RawBinaryBitwiseOperation,
) -> Sequence[mlir.Value]:
    lhs_type = _expect_arith_type(operation.lhs.get_type())
    rhs_type = _expect_arith_type(operation.rhs.get_type())
    res_type = _expect_arith_type(operation.result_var.get_type())
    assert lhs_type == rhs_type == res_type

    res_dtype = res_type.tensor_dtype()
    mlir_op = _get_mlir_op_for_op_and_dtype(operation.fn, res_dtype)
    if mlir_op is None:
        raise NotImplementedError(
            f"Arithmetic operation {operation.fn} not supported for {res_dtype=}"
        )
    return [
        mlir_op(
            lhs=context.get_var(operation.lhs),
            rhs=context.get_var(operation.rhs),
        )
    ]


@mlir_op_lowering
def lower_raw_binary_arith(
    context: MLIRLoweringContext, operation: ops.RawBinaryArithmeticOperation
) -> Sequence[mlir.Value]:
    return _lower_raw_binary_arith(context, operation)


@mlir_op_lowering
def lower_raw_binary_bitwise_arith(
    context: MLIRLoweringContext, operation: ops.RawBinaryBitwiseOperation
) -> Sequence[mlir.Value]:
    return _lower_raw_binary_arith(context, operation)


@mlir_op_lowering
def lower_typed_const(
    context: MLIRLoweringContext, operation: ops.TypedConst
) -> Sequence[mlir.Value]:
    del context
    target_type = operation.result_var.get_type()
    value = operation.value
    match target_type:
        case (
            ir_type.FunctionTy()
            | ir_type.ModuleTy()
            | ir_type.StringTy()
            | ir_type.DTypeConstructor()
            | ir_type.NoneType()
            | ir_type.TypeTy()
            | ir_type.EnumTy()
        ):
            return [value]
        case ir_type.ScalarTy() | ir_type.VectorTy():
            mlir_type = ir_type_to_mlir_type(target_type)
            return [mlir_constant_of_type(mlir_type, value)]
        case _:
            raise NotImplementedError(
                f"Unable to convert {value=} to MLIR constant of type {target_type=}"
            )


@mlir_op_lowering
def lower_branch(
    context: MLIRLoweringContext, operation: ops.Branch
) -> Sequence[mlir.Value]:
    mlir_block = context.block_map[operation.target]
    mlir.cf.add_BranchOp(
        dest=mlir_block.label,
        destOperands=tuple(context.get_var(arg) for arg in operation.args),
    )
    return []


@mlir_op_lowering
def lower_cond_branch(
    context: MLIRLoweringContext, operation: ops.CondBranch
) -> Sequence[mlir.Value]:
    cond = mlir.arith.add_TruncIOp(
        out_type=T.i1(),
        in_=context.get_var(operation.cond),
    )
    true_block = context.block_map[operation.true_target]
    false_block = context.block_map[operation.false_target]
    mlir.cf.add_CondBranchOp(
        condition=cond,
        trueDest=true_block.label,
        falseDest=false_block.label,
        trueDestOperands=tuple(context.get_var(arg) for arg in operation.true_args),
        falseDestOperands=tuple(context.get_var(arg) for arg in operation.false_args),
    )
    return []


@mlir_op_lowering
def lower_assign(context: MLIRLoweringContext, operation: ops.Assign) -> None:
    if operation.result_var.is_aggregate():
        context.bind_aggregate(operation.result_var, operation.value)
    else:
        context.def_var(operation.result_var, context.get_var(operation.value))


@mlir_op_lowering
def lower_raw_where(
    context: MLIRLoweringContext, operation: ops.RawWhereOperation
) -> Sequence[mlir.Value]:
    cond_type = _expect_arith_type(operation.cond.get_type())
    if cond_type.tensor_shape() == ():
        result_type = T.i1()
    else:
        result_type = mlir.VectorType(
            shape=cond_type.tensor_shape(),
            elementType=T.i1(),
            scalableDims=(False,) * len(cond_type.tensor_shape()),
        )
    cond = mlir.arith.add_TruncIOp(
        out_type=result_type,
        in_=context.get_var(operation.cond),
    )
    return [
        mlir.llvm.add_SelectOp(
            condition=cond,
            trueValue=context.get_var(operation.x),
            falseValue=context.get_var(operation.y),
        )
    ]


@mlir_op_lowering
def lower_dummy(
    context: MLIRLoweringContext, operation: ops.MakeDummy
) -> Sequence[mlir.Value]:
    del context
    result_type = ir_type_to_mlir_type(operation.result_var.get_type())
    if isinstance(result_type, mlir.llvm.LLVMPointerType):
        return [mlir.llvm.add_ZeroOp(res_type=result_type)]
    return [mlir_constant_of_type(result_type, 0)]


class DeviceIR2MLIR:
    def __init__(
        self,
        signature: KernelSignature,
        region: ir.Region,
        ctx: ir.IRContext,
        compiler_options: CompilerOptions,
    ):
        self.context = DeviceLoweringContext(
            signature=signature,
            region=region,
            ir_context=ctx,
        )
        self.compiler_options = compiler_options

    def __call__(self) -> mlir.Operation:
        context = self.context
        self.setup_func_op()
        self.setup_blocks()

        try:
            self.lower_region()
        except Exception:
            if context.ir_context.log_ir_on_error:
                highlight_loc = (
                    context.current_op.loc
                    if context.current_op is not None
                    else None
                )
                ir_str = context.region.to_string(highlight_loc=highlight_loc)
                print(
                    f"==== Encountered error converting IR to MLIR: ====\n\n{ir_str}\n\n",
                    file=sys.stderr,
                )
            raise

        assert context.module_op is not None, "MLIR module not initialized"
        return context.module_op

    def lower_region(self) -> None:
        context = self.context
        for ir_block in context.region.blocks:
            mlir_block = context.block_map[ir_block]
            assert len(mlir_block.args) == len(ir_block.params), (
                "MLIR block parameters do not match IR block parameters: "
                f"{ir_block._name}"
            )
            for mlir_arg, ir_arg in zip(mlir_block.args, ir_block.params):
                assert mlir_arg.type == ir_type_to_mlir_type(ir_arg.get_type()), (
                    "MLIR argument type does not match IR argument type: "
                    f"{ir_arg.name}"
                )
                context.def_var(ir_arg, mlir_arg)

            self.lower_block(ir_block)

    def lower_block(self, ir_block: ir.Block) -> None:
        context = self.context
        mlir_block = context.block_map[ir_block]
        with mlir_block.append_here():
            for operation in ir_block.operations:
                context.current_op = operation

                if isinstance(operation, tuple(_NOOP_LOWERINGS)):
                    continue

                results = _MLIR_OP_LOWERING.lower(self.context, operation)

                # Aggregate assignments do not materialize an MLIR SSA value.
                if isinstance(operation, ops.Assign):
                    continue

                assert len(operation.result_vars) == len(results)
                for lhs, rhs in zip(operation.result_vars, results):
                    context.def_var(lhs, rhs)

    def setup_blocks(self):
        context = self.context
        assert context.func_op is not None, "MLIR GPU function not initialized"
        func_region = context.func_op.regions[0]
        for ir_block in context.region.blocks:
            mlir_block_param_types = tuple(
                ir_type_to_mlir_type(param.get_type()) for param in ir_block.params
            )
            block_param_names = tuple(param.name for param in ir_block.params)
            args = tuple(
                mlir.Value(param_type, param_name)
                for param_type, param_name in zip(
                    mlir_block_param_types, block_param_names
                )
            )
            mlir_block = func_region.new_block(args=args, block_id=ir_block._name)
            context.block_map[ir_block] = mlir_block

    def setup_func_op(self):
        context = self.context
        with mlir.Block().append_here() as top_block:
            module_region = mlir.Region()
            mlir.add_ModuleOp(
                bodyRegion=module_region,
                extra_attributes=[("gpu.container_module", mlir.UnitAttr())],
            )
        context.module_op = top_block[0]

        with module_region.new_block().append_here() as module_block:
            gpu_module_region = mlir.Region()
            mlir.gpu.add_GPUModuleOp(
                sym_name="kernels",
                targets=mlir.ArrayAttr(value=[mlir.nvvm.NVVMTargetAttr()]),
                bodyRegion=gpu_module_region,
            )
        context.gpu_module_op = module_block[0]

        with gpu_module_region.new_block().append_here():
            body_region = mlir.Region()
            input_types = tuple(
                ir_type_to_mlir_type(param.get_type())
                for param in context.region.blocks[0].params
            )
            function_type = mlir.llvm.LLVMFunctionType(
                returnType=mlir.llvm.LLVMVoidType(), params=input_types, varArg=False
            )
            arg_attrs = [
                self._get_arg_attributes(param.get_type())
                for param in context.region.blocks[0].params
            ]

            mlir.llvm.add_LLVMFuncOp(
                sym_name=context.signature.symbol,
                function_type=function_type,
                arg_attrs=mlir.ArrayAttr(value=arg_attrs),
                body=body_region,
                extra_attributes=self._get_nvvm_function_attributes(),
            )
            context.func_op = gpu_module_region.blocks[0].operations[-1]

    def _get_nvvm_function_attributes(self):
        attributes = [
            ("nvvm.kernel", mlir.UnitAttr()),
        ]
        if self.compiler_options.max_threads_per_block is not None:
            attributes.append(
                (
                    "nvvm.maxntid",
                    mlir.DenseI32ArrayAttr(self.compiler_options.max_threads_per_block),
                )
            )

        if self.compiler_options.max_blocks_per_cluster is not None:
            attributes.append(
                (
                    "nvvm.cluster_max_blocks",
                    mlir.IntegerAttr(
                        type=mlir.IntegerType.signless(32),
                        value=self.compiler_options.max_blocks_per_cluster,
                    ),
                )
            )

        if self.compiler_options.min_blocks_per_sm is not None:
            attributes.append(
                (
                    "nvvm.minctasm",
                    mlir.IntegerAttr(
                        type=mlir.IntegerType.signless(32),
                        value=self.compiler_options.min_blocks_per_sm,
                    ),
                )
            )

        if self.compiler_options.max_registers_per_thread is not None:
            attributes.append(
                (
                    "nvvm.maxnreg",
                    mlir.IntegerAttr(
                        type=mlir.IntegerType.signless(32),
                        value=self.compiler_options.max_registers_per_thread,
                    ),
                )
            )
        return attributes

    def _get_arg_attributes(self, ty: ir_type.Type) -> mlir.DictionaryAttr:
        named_attrs = []
        if isinstance(ty, ir_type.TensorMapTy):
            i64_ty = mlir.IntegerType.signless(64)
            i64x16_arr_ty = mlir.llvm.LLVMArrayType(elementType=i64_ty, numElements=16)
            tensormap_struct_ty = mlir.llvm.LLVMStructType(types=(i64x16_arr_ty,))
            named_attrs.append(mlir.NamedAttribute.make(
                "llvm.align", mlir.IntegerAttr.make(i64_ty, 64)))
            named_attrs.append(mlir.NamedAttribute.make(
                "llvm.byval", mlir.TypeAttr(value=tensormap_struct_ty)))
            named_attrs.append(mlir.NamedAttribute.make(
                "nvvm.grid_constant", mlir.UnitAttr()))
        return mlir.DictionaryAttr(value=named_attrs)


@mlir_op_lowering
def lower_assume_bounded(
    context: MLIRLoweringContext, operation: ops.AssumeBounded
) -> Sequence[mlir.Value]:
    return [context.get_var(operation.x)]


@mlir_op_lowering
def lower_assume_div_by(
    context: MLIRLoweringContext, operation: ops.AssumeDivBy
) -> Sequence[mlir.Value]:
    return [context.get_var(operation.x)]


@mlir_op_lowering
def lower_astype(
    context: MLIRLoweringContext, operation: ops.TileAsType
) -> Sequence[mlir.Value]:
    src = context.get_var(operation.x)
    src_type = _expect_arith_type(operation.x.get_type())
    dst_type = _expect_arith_type(operation.result_var.get_type())
    assert src_type.tensor_shape() == dst_type.tensor_shape()
    assert datatype.is_arithmetic(src_type.tensor_dtype())
    assert datatype.is_arithmetic(dst_type.tensor_dtype())
    result = convert_dtype(src_type, dst_type, src)
    return [result]


@mlir_op_lowering(host=False)
def lower_atomic_rmw(
    context: DeviceLoweringContext, operation: ops.AtomicRMW
) -> Sequence[mlir.Value]:
    pointer = context.get_var(operation.pointer)
    value = context.get_var(operation.value)
    value_dtype = operation.value.get_type().dtype
    bin_op = _get_llvm_atomic_binop(operation.kind, value_dtype)

    result = mlir.llvm.add_AtomicRMWOp(
        bin_op=bin_op,
        ptr=pointer,
        val=value,
        ordering=_get_llvm_memory_ordering(operation.memory_order),
        syncscope=_get_llvm_syncscope(operation.memory_scope),
    )
    return [result]


@mlir_op_lowering(host=False)
def lower_atomic_exchange(
    context: DeviceLoweringContext, operation: ops.AtomicExchange
) -> Sequence[mlir.Value]:
    pointer = context.get_var(operation.pointer)
    value = context.get_var(operation.value)
    result = mlir.llvm.add_AtomicRMWOp(
        bin_op=mlir.llvm.AtomicBinOp.xchg,
        ptr=pointer,
        val=value,
        ordering=_get_llvm_memory_ordering(operation.memory_order),
        syncscope=_get_llvm_syncscope(operation.memory_scope),
    )
    return [result]


@mlir_op_lowering(host=False)
def lower_atomic_cas(
    context: DeviceLoweringContext, operation: ops.AtomicCAS
) -> Sequence[mlir.Value]:
    pointer = context.get_var(operation.pointer)
    compare = context.get_var(operation.compare)
    value = context.get_var(operation.value)
    pair = mlir.llvm.add_AtomicCmpXchgOp(
        ptr=pointer,
        cmp=compare,
        val=value,
        success_ordering=_get_llvm_memory_ordering(operation.memory_order),
        failure_ordering=_get_llvm_cmpxchg_failure_ordering(
            operation.memory_order
        ),
        syncscope=_get_llvm_syncscope(operation.memory_scope),
    )

    # llvm.cmpxchg returns {old_value, success_flag}, we want the old value.
    old_value = mlir.llvm.add_ExtractValueOp(
        res_type=value.type,
        container=pair,
        position=(0,),
    )
    return [old_value]


@mlir_op_lowering(host=False)
def lower_return(context: DeviceLoweringContext, operation: ops.Return) -> Sequence[mlir.Value]:
    assert operation.result_vars == (), "Kernels may not return values"
    mlir.llvm.add_ReturnOp()
    return []


@mlir_op_lowering(host=False)
def lower_printf(
    context: DeviceLoweringContext, operation: ops.TilePrintf
) -> Sequence[mlir.Value]:

    def extend_lt32b_integral_args(arg):
        "Sub-32b integral arguments need to be extended for the %d format"

        ty = arg.get_type()
        value = context.get_var(arg)

        if not isinstance(ty, ir_type.ScalarTy):
            return value
        dtype = ty.dtype
        if datatype.is_boolean(dtype):
            return mlir.arith.add_ExtUIOp(
                out_type=T.i32(),
                in_=value,
            )
        if datatype.is_integral(dtype) and dtype.bitwidth < 32:
            converter = (
                mlir.arith.add_ExtSIOp
                if datatype.is_signed(dtype)
                else mlir.arith.add_ExtUIOp
            )
            return converter(out_type=T.i32(), in_=value)
        return value

    args = tuple(map(extend_lt32b_integral_args, operation.args))
    mlir.gpu.add_PrintfOp(format=operation.format, args=args)
    # printOp returns a token, which is only used in cutile token order pass,
    # to match the number of return vars, we return a dummy value None
    # that will not get used.
    return [None]


@mlir_op_lowering
def lower_pointer_offset(
    context: MLIRLoweringContext, operation: ops.PointerOffset
) -> Sequence[mlir.Value]:
    ptr_ty = _expect_pointer_type(operation.pointer.get_type())
    element_type = dtype_to_mlir_type(ptr_ty.pointee_dtype)

    pointer = context.get_var(operation.pointer)
    offset = context.get_var(operation.offset)
    dynamic_32b_sentinel = -1 << 31
    offset_pointer = mlir.llvm.add_GEPOp(
        res_type=pointer.type,
        base=pointer,
        dynamicIndices=[offset],
        rawConstantIndices=[dynamic_32b_sentinel],
        elem_type=element_type,
    )
    return [offset_pointer]


@mlir_op_lowering
def lower_vector_getitem(
    context: MLIRLoweringContext, operation: ops.VectorGetItem
) -> Sequence[mlir.Value]:
    vector = context.get_var(operation.x)
    position = context.get_var(operation.index)
    element = mlir.llvm.add_ExtractElementOp(vector=vector, position=position)
    return [element]


@mlir_op_lowering
def lower_vector_reduce(
    context: MLIRLoweringContext, operation: ops.VectorReduce
) -> Sequence[mlir.Value]:
    vector = context.get_var(operation.x)
    result_type = ir_type_to_mlir_type(operation.result_var.get_type())
    vector_type = operation.x.get_type()
    assert isinstance(vector_type, ir_type.VectorTy)
    dtype = vector_type.element_dtype

    match operation.kind:
        case VectorReduction.add if datatype.is_integral(dtype):
            result = mlir.llvm.add_vector_reduce_add(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.add:
            start = mlir_constant_of_type(result_type, 0.0)
            result = mlir.llvm.add_vector_reduce_fadd(
                res_type=result_type,
                start_value=start,
                input=vector,
                fastmathFlags=(
                    mlir.llvm.FastmathFlags.reassoc
                    if operation.reassociate
                    else mlir.llvm.FastmathFlags(0)
                ),
            )
        case VectorReduction.mul if datatype.is_integral(dtype):
            result = mlir.llvm.add_vector_reduce_mul(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.mul:
            start = mlir_constant_of_type(result_type, 1.0)
            result = mlir.llvm.add_vector_reduce_fmul(
                res_type=result_type,
                start_value=start,
                input=vector,
                fastmathFlags=(
                    mlir.llvm.FastmathFlags.reassoc
                    if operation.reassociate
                    else mlir.llvm.FastmathFlags(0)
                ),
            )
        case VectorReduction.bitwise_and:
            result = mlir.llvm.add_vector_reduce_and(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.bitwise_or:
            result = mlir.llvm.add_vector_reduce_or(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.bitwise_xor:
            result = mlir.llvm.add_vector_reduce_xor(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.max if datatype.is_float(dtype):
            builder = (
                mlir.llvm.add_vector_reduce_fmaximum
                if operation.propagate_nan
                else mlir.llvm.add_vector_reduce_fmax
            )
            result = builder(res_type=result_type, in_=vector)
        case VectorReduction.max if datatype.is_signed(dtype):
            result = mlir.llvm.add_vector_reduce_smax(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.max:
            result = mlir.llvm.add_vector_reduce_umax(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.min if datatype.is_float(dtype):
            builder = (
                mlir.llvm.add_vector_reduce_fminimum
                if operation.propagate_nan
                else mlir.llvm.add_vector_reduce_fmin
            )
            result = builder(res_type=result_type, in_=vector)
        case VectorReduction.min if datatype.is_signed(dtype):
            result = mlir.llvm.add_vector_reduce_smin(
                res_type=result_type,
                in_=vector,
            )
        case VectorReduction.min:
            result = mlir.llvm.add_vector_reduce_umin(
                res_type=result_type,
                in_=vector,
            )
        case _:
            raise InternalError(
                f"Unable to lower vector reduction {operation.kind.value} "
                f"for {dtype}"
            )

    return [result]


@mlir_op_lowering(host=False)
def lower_load_pointer(
    context: DeviceLoweringContext, operation: ops.LoadPointer
) -> Sequence[mlir.Value]:
    ptr_dtype = operation.pointer.get_type().pointer_dtype
    ordering = _get_llvm_memory_ordering(operation.memory_order)
    info = PointerInfo(ptr_dtype)
    assert not info.opaque, f"Expected a typed pointer, got {ptr_dtype}"
    result_type = ir_type_to_mlir_type(operation.result_var.get_type())
    pointer = context.get_var(operation.pointer)
    result = mlir.llvm.add_LoadOp(
        res_type=result_type,
        addr=pointer,
        alignment=operation.alignment,
        volatile_=operation.volatile,
        ordering=ordering,
    )
    return [result]


@mlir_op_lowering(host=False)
def lower_store_pointer(
    context: DeviceLoweringContext, operation: ops.StorePointer
) -> Sequence[mlir.Value]:
    pointer = context.get_var(operation.pointer)
    ordering = _get_llvm_memory_ordering(operation.memory_order)
    value = context.get_var(operation.value)
    mlir.llvm.add_StoreOp(
        value=value,
        addr=pointer,
        alignment=operation.alignment,
        volatile_=operation.volatile,
        ordering=ordering,
    )
    return []


@mlir_op_lowering(host=False)
def lower_alloc_local_memory(
    context: DeviceLoweringContext, operation: ops.AllocLocalMemory
) -> Sequence[mlir.Value]:
    result_ty = operation.result_var.get_type()
    assert isinstance(result_ty, ir_type.PointerTy)
    result_ty_mlir = ir_type_to_mlir_type(result_ty)
    elem_type_mlir = dtype_to_mlir_type(result_ty.pointee_dtype)

    with context.function_region.blocks[0].prepend_here():
        array_size = mlir_constant_of_type(T.i64(), operation.count)
        ptr = mlir.llvm.add_AllocaOp(
            res_type=result_ty_mlir,
            arraySize=array_size,
            elem_type=elem_type_mlir,
            alignment=operation.alignment,
        )
    # It would be nice to emit llvm.intr.lifetime_start() here but MLIR's region-simplify
    # pass doesn't respect it and produces invalid IR as a result.
    # mlir.llvm.add_LifetimeStartOp(ptr=ptr)
    return [ptr]


@mlir_op_lowering(host=False)
def lower_dealloc_local_memory(context: DeviceLoweringContext, operation: ops.DeallocLocalMemory):
    # It would be nice to emit llvm.intr.lifetime_end() here but MLIR's region-simplify
    # pass doesn't respect it and produces invalid IR as a result.
    # mlir.llvm.add_LifetimeEndOp(ptr=context.get_var(operation.ptr))
    return ()


@mlir_op_lowering(host=False)
def lower_alloc_static_shared_memory(
    context: DeviceLoweringContext, operation: ops.AllocStaticSharedMemory
) -> Sequence[mlir.Value]:
    result_ty = operation.result_var.get_type()
    assert isinstance(result_ty, ir_type.PointerTy)
    elem_type = dtype_to_mlir_type(result_ty.pointee_dtype)
    global_type = mlir.llvm.LLVMArrayType(
        elementType=elem_type,
        numElements=operation.count,
    )
    sym = f"static_shared_memory_{operation.result_var.name}"
    with context.gpu_module.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_GlobalOp(
            global_type=global_type,
            sym_name=sym,
            linkage=mlir.llvm.Linkage.Internal,
            addr_space=ir_type.MemorySpace.SHARED._value_,
            visibility_=mlir.llvm.Visibility.Default,
            initializer=mlir.Region(),
            alignment=operation.alignment,
        )
    base = mlir.llvm.add_AddressOfOp(
        res_type=ir_type_to_mlir_type(result_ty),
        global_name=sym,
    )
    return [base]


@mlir_op_lowering(host=False)
def lower_get_dyn_shared_memory_base_ptr(
    context: DeviceLoweringContext,
    operation: ops.GetDynSharedMemoryBasePtr,
) -> Sequence[mlir.Value]:
    global_type = mlir.llvm.LLVMArrayType(elementType=T.i8(), numElements=0)
    sym = context.signature.symbol + "$dynamic_shared_memory"
    addr_space = ir_type.MemorySpace.SHARED._value_
    with context.gpu_module.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_GlobalOp(
            global_type=global_type,
            sym_name=sym,
            linkage=mlir.llvm.Linkage.External,
            addr_space=addr_space,
            visibility_=mlir.llvm.Visibility.Default,
            initializer=mlir.Region(),
            alignment=ops.GetDynSharedMemoryBasePtr.initial_alignment,
        )
    res_type = ir_type_to_mlir_type(operation.result_var.get_type())
    assert isinstance(res_type, mlir.llvm.LLVMPointerType)
    ptr = mlir.llvm.add_AddressOfOp(res_type=res_type, global_name=sym)
    return [ptr]


@mlir_op_lowering
def lower_tensor_map_as_opaque_ptr(
    context: MLIRLoweringContext, operation: ops.TensorMapAsOpaquePtr
):
    tm = context.get_var(operation.tensor_map)
    return [tm]


@mlir_op_lowering(host=False)
def lower_inline_ptx(
    context: DeviceLoweringContext, operation: ops.InlinePTX
) -> Sequence[mlir.Value]:
    ptx_code = operation.ptx_code
    ro_args = tuple(context.get_var(arg) for arg in operation.read_only_operands)
    rw_args = tuple(context.get_var(arg) for arg in operation.read_write_operands)
    wo_args = tuple(dtype_to_mlir_type(arg) for arg in operation.write_only_operands)
    results = mlir.nvvm.add_InlinePtxOp(
        ptxCode=ptx_code,
        readOnlyArgs=ro_args,
        readWriteArgs=rw_args,
        writeOnlyArgs_types=wo_args,
    )
    return tuple(results)


def _lower_intrinsic_operand(context: MLIRLoweringContext, operand: ir.Var) -> mlir.Value:
    """
    Bool operands need to be truncated from their cutile storage type
    to their MLIR storage type
    """
    operand_value = context.get_var(operand)
    operand_type = operand.get_type()
    match operand_type:
        case ir_type.VectorTy() as vt if vt.element_dtype == datatype.bool_:
            res_ty = mlir.VectorType(
                shape=[vt.length], elementType=T.i1(), scalableDims=[False]
            )
            return mlir.arith.add_TruncIOp(out_type=res_ty, in_=operand_value)
        case ir_type.ScalarTy() as st if st.dtype == datatype.bool_:
            return mlir.arith.add_TruncIOp(out_type=T.i1(), in_=operand_value)
        case _:
            return operand_value


def _lower_intrinsic_result(
    context: MLIRLoweringContext, results: Sequence[mlir.Value]
) -> Sequence[mlir.Value]:
    for result in results:
        if result.type == T.i1():
            yield mlir.arith.add_ExtUIOp(out_type=T.i8(), in_=result)
        elif (
            isinstance(result.type, mlir.VectorType)
            and result.type.elementType == T.i1()
        ):
            out_type = mlir.VectorType(
                shape=result.type.shape, elementType=T.i8(), scalableDims=[False]
            )
            yield mlir.arith.add_ExtUIOp(out_type=out_type, in_=result)
        else:
            yield result


def _lower_intrinsic_result_type(
    context: MLIRLoweringContext, result_types: Sequence[ir_type.Type]
) -> Sequence[mlir.Type]:
    for result_type in result_types:
        if result_type == ir_type.ScalarTy(datatype.bool_):
            yield T.i1()
        elif (
            isinstance(result_type, ir_type.VectorTy)
            and result_type.element_dtype == datatype.bool_
        ):
            yield mlir.VectorType(
                shape=[result_type.length], elementType=T.i1(), scalableDims=[False]
            )
        else:
            yield ir_type_to_mlir_type(result_type)


def _extract_aggregate_elements(
    context: DeviceLoweringContext, value: mlir.Value
) -> Sequence[mlir.Value]:
    del context
    assert isinstance(value.type, mlir.llvm.LLVMStructType)
    element_types = value.type.types
    for i, element_type in enumerate(element_types):
        yield mlir.llvm.add_ExtractValueOp(
            res_type=element_type,
            container=value,
            position=[i]
        )


@mlir_op_lowering(host=False)
def lower_raw_nvvm_intrinsic(
    context: DeviceLoweringContext, operation: ops.RawNVVMIntrinsic
) -> Sequence[mlir.Value]:
    operands = tuple(
        _lower_intrinsic_operand(context, operand)
        for operand in operation.operands_
    )
    result_types = tuple(
        _lower_intrinsic_result_type(
            context,
            (result.get_type() for result in operation.result_vars),
        )
    )

    match len(result_types):
        case 0:
            mlir_result_type = None
        case 1:
            mlir_result_type = result_types[0]
        case _:
            mlir_result_type = mlir.llvm.LLVMStructType(types=result_types)

    intrinsic_call_result = mlir.llvm.add_CallIntrinsicOp(
        results_type=mlir_result_type,
        intrin=operation.intrinsic,
        args=operands,
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )

    match len(result_types):
        case 0:
            mlir_values = []
        case 1:
            mlir_values = [intrinsic_call_result]
        case _:
            mlir_values = list(_extract_aggregate_elements(context, intrinsic_call_result))

    return tuple(_lower_intrinsic_result(context, mlir_values))


@mlir_op_lowering
def lower_raw_mlir_operation(
    context: MLIRLoweringContext, operation: ops.RawMLIROperation
) -> Sequence[mlir.Value]:
    operands = tuple(
        _lower_intrinsic_operand(context, operand) for operand in operation.operands_
    )
    result_types = tuple(
        _lower_intrinsic_result_type(
            context,
            (result_var.get_type() for result_var in operation.result_vars),
        )
    )
    results = mlir.add_operation(
        name=operation.op_name,
        result_type=result_types,
        operands=operands,
        properties=(),
        attributes=operation.mlir_attributes,
    )
    return tuple(_lower_intrinsic_result(context, results))


@mlir_op_lowering
def lower_reinterpret_pointer(
    context: MLIRLoweringContext, operation: ops.ReinterpretPointer
) -> Sequence[mlir.Value]:
    # This operation only exists to reinterpret the type of a pointer
    # at the IR level, but the MLIR representation will be the same either way.
    return [context.get_var(operation.pointer)]


@mlir_op_lowering
def lower_addrspace_cast(
    context: MLIRLoweringContext, operation: ops.AddrSpaceCast
) -> Sequence[mlir.Value]:
    value = context.get_var(operation.pointer)
    ir_result_type: ir_type.PointerTy = operation.result_var.get_type()
    if ir_result_type.memory_space.value == value.type.addressSpace:
        return [value]
    mlir_result_type = ir_type_to_mlir_type(ir_result_type)
    result = mlir.llvm.add_AddrSpaceCastOp(
        res_type=mlir_result_type,
        arg=value,
    )
    return [result]


@mlir_op_lowering
def lower_bitshift(
    context: MLIRLoweringContext, operation: ops.RawBitwiseShiftOperation
) -> Sequence[mlir.Value]:
    lhs_type = _expect_arith_type(operation.lhs.get_type())
    rhs_type = _expect_arith_type(operation.rhs.get_type())
    res_type = _expect_arith_type(operation.result_var.get_type())

    assert lhs_type == rhs_type == res_type
    assert datatype.is_integral(res_type.tensor_dtype())

    lhs = context.get_var(operation.lhs)
    rhs = context.get_var(operation.rhs)

    match operation.fn:
        case "lshift":
            result = mlir.arith.add_ShLIOp(lhs=lhs, rhs=rhs)
        case "rshift":
            shift_op = (
                mlir.arith.add_ShRSIOp
                if datatype.is_signed(res_type.tensor_dtype())
                else mlir.arith.add_ShRUIOp
            )
            result = shift_op(lhs=lhs, rhs=rhs)
        case _:
            raise NotImplementedError(
                f"Bitwise shift operation {operation.fn} not supported"
            )

    return [result]


@mlir_op_lowering
def lower_tile_broadcast(
    context: MLIRLoweringContext, operation: ops.TileBroadcast
) -> Sequence[mlir.Value]:
    res_ty = operation.result_var.get_type()
    x = operation.x
    x_ty = x.get_type()
    if (
        not isinstance(x_ty, ir_type.VectorTy)
        or not isinstance(res_ty, ir_type.VectorTy)
        or x_ty.length != 1
    ):
        raise InternalError(
            "Expected length-1 vector but got result "
            f"type {res_ty} and operand type {x_ty}"
        )
    if x_ty.element_dtype != res_ty.element_dtype:
        raise InternalError(
            "Expected broadcast operand and result type to have same "
            f"dtype but got result type {res_ty} and operand type {x_ty}"
        )

    mask = [0 for _ in range(res_ty.length)]
    res_ty = ir_type_to_mlir_type(res_ty)
    x = context.get_var(x)
    res = mlir.llvm.add_ShuffleVectorOp(res_type=res_ty, v1=x, v2=x, mask=mask)
    return [res]


@mlir_op_lowering
def lower_tile_reshape(
    context: MLIRLoweringContext, operation: ops.TileReshape
) -> Sequence[mlir.Value]:
    res_ty = operation.result_var.get_type()
    x = operation.x
    x_ty = x.get_type()
    match x_ty, res_ty:
        case ir_type.ScalarTy(), ir_type.VectorTy():
            res_ty = ir_type_to_mlir_type(res_ty)
            res = mlir.llvm.add_PoisonOp(res_type=res_ty)
            if len(res_ty.shape) != 1:
                raise InternalError(
                    f"Expected vector to have 1d shape but got {res_ty}"
                )
            for i in range(res_ty.shape[0]):
                position = mlir_constant_of_type(T.i32(), i)
                res = mlir.llvm.add_InsertElementOp(
                    vector=res,
                    value=context.get_var(x),
                    position=position,
                )
            return [res]
    raise NotImplementedError(
        f"Could not broadcast value of type {x_ty} to type {res_ty}"
    )


@mlir_op_lowering(host=False)
def lower_foreign_function(
    context: DeviceLoweringContext, operation: ops.ForeignFunction
) -> Sequence[mlir.Value]:
    function_name = operation.function_name
    operands = tuple(context.get_var(operand) for operand in operation.operands_)
    operand_types = tuple(
        ir_type_to_mlir_type(operand.get_type()) for operand in operation.operands_
    )

    has_result = len(operation.result_vars) > 0
    result_type = (
        ir_type_to_mlir_type(operation.result_var.get_type())
        if has_result
        else None
    )
    function_type = mlir.llvm.LLVMFunctionType(
        returnType=result_type if result_type else mlir.llvm.LLVMVoidType(),
        params=operand_types,
        varArg=False,
    )
    if prev_type := context.seen_foreign_functions.get(function_name):
        if prev_type != function_type:
            raise TypeCheckingError(
                f"Tried calling foreign function {function_name} with type {function_type} "
                f"but it has already been declared with type {prev_type!r}"
            )
    else:
        mlir.llvm.add_LLVMFuncOp(
            sym_name=function_name,
            linkage=mlir.llvm.Linkage.External,
            body=mlir.Region(),
            function_type=function_type,
        )
        func_op = _Cursor.current()._block.operations.pop()
        context.gpu_module.regions[0].blocks[0].operations.insert(0, func_op)
        context.seen_foreign_functions[function_name] = function_type

    call = mlir.llvm.add_CallOp(
        result_type=result_type,
        callee=function_name,
        callee_operands=operands,
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )
    return [call] if has_result else []


@mlir_op_lowering
def lower_bitcast(
    context: MLIRLoweringContext, operation: ops.BitCast
) -> Sequence[mlir.Value]:
    x = context.get_var(operation.x)
    src_ty, dst_ty = operation.x.get_type(), operation.result_var.get_type()
    src_mlir_ty, dst_mlir_ty = (
        ir_type_to_mlir_type(src_ty),
        ir_type_to_mlir_type(dst_ty),
    )
    if src_mlir_ty == dst_mlir_ty:
        return [x]
    match src_ty, dst_ty:
        case ir_type.PointerTy(), ir_type.PointerTy():
            res = mlir.llvm.add_AddrSpaceCastOp(res_type=dst_mlir_ty, arg=x)
            return [res]
        case ir_type.ScalarTy() as st, ir_type.PointerTy():
            if not datatype.is_integral(st.dtype):
                raise InternalError(
                    "bitcast to or from pointer must go through integer"
                )
            res = mlir.llvm.add_IntToPtrOp(res_type=dst_mlir_ty, arg=x)
            return [res]
        case ir_type.PointerTy(), ir_type.ScalarTy() as st:
            if not datatype.is_integral(st.dtype):
                raise InternalError(
                    "bitcast to or from pointer must go through integer"
                )
            res = mlir.llvm.add_PtrToIntOp(res_type=dst_mlir_ty, arg=x)
            return [res]
        case _:
            res = mlir.llvm.add_BitcastOp(res_type=dst_mlir_ty, arg=x)
            return [res]


def ir2mlir(
    signature: KernelSignature,
    body: ir.Region,
    ctx: ir.IRContext,
    compiler_options: CompilerOptions,
) -> mlir.Operation:
    lower = DeviceIR2MLIR(signature, body, ctx, compiler_options)
    op = lower()
    return op


__all__ = ("ir2mlir",)
