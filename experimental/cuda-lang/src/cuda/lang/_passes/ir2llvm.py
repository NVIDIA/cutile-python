# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Mapping

from cuda.lang._ir import ir
from cuda.lang._ir import type as ir_type
from cuda.lang._target import TargetInfo

from .. import _llvm_bitcode as llvm
from .._llvm_bitcode import BitcodeBuilder, DATALAYOUT_PTX
from cuda.lang import _datatype as datatype
from cuda.tile._bytecode import float_to_bits
from cuda.tile._datatype import dtype_simple_bytecode_type, is_integral, is_boolean
from cuda.tile._exception import InternalError, Loc
from cuda.tile._ir.control_flow_ops import Return

DIRECTLY_SUPPORTED_FLOATS = {
    datatype.float16: llvm.FloatKind.f16,
    datatype.bfloat16: llvm.FloatKind.bf16,
    datatype.float32: llvm.FloatKind.f32,
    datatype.float64: llvm.FloatKind.f64,
}


def dtype_to_llvm(dtype: datatype.DType, tt: llvm.TypeTable,
                  storage: bool) -> llvm.Type:
    if datatype.is_pointer_dtype(dtype):
        info = datatype.PointerInfo(dtype)
        return tt.pointer(info.memory_space._value_)
    elif dtype == datatype.bool_:
        return tt.integer(8 if storage else 1)
    elif is_integral(dtype):
        return tt.integer(dtype.bitwidth)
    elif (kind := DIRECTLY_SUPPORTED_FLOATS.get(dtype)) is not None:
        return tt.float(kind)
    elif datatype.is_float(dtype):
        return tt.integer((dtype.bitwidth + 7) // 8 if storage else dtype.bitwidth)
    elif dtype == datatype.tensor_map_descriptor:
        return tt.array(tt.integer(64), dtype.bitwidth // 64)
    elif dtype == datatype.mbarrier:
        return tt.integer(64)
    elif dtype == datatype.cluster_launch_control_token:
        return tt.integer(128)
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")


def type_to_llvm(ty: ir_type.Type, tt: llvm.TypeTable, storage: bool) -> llvm.Type:
    if isinstance(ty, ir_type.ScalarTy):
        return dtype_to_llvm(ty.dtype, tt, storage)
    elif isinstance(ty, ir_type.PointerTy):
        return dtype_to_llvm(ty.pointer_dtype, tt, storage)
    elif isinstance(ty, ir_type.VectorTy):
        el_ty = dtype_to_llvm(ty.element_dtype, tt, storage)
        return tt.vector(el_ty, ty.length)
    else:
        raise NotImplementedError(f"Unsupported type {ty}")


class LLVMLoweringContext:
    def __init__(self,
                 builder: BitcodeBuilder,
                 ir_ctx: ir.IRContext,
                 target_info: TargetInfo | None,  # None for host
                 all_blocks: Sequence[ir.Block],
                 ):
        self.ir_ctx = ir_ctx
        self.builder = builder
        self.target_info = target_info
        self._value_map: dict[str, llvm.Value] = dict()
        self._block_map: dict[ir.Block, int] = {block: i for i, block in enumerate(all_blocks)}
        self._function_declarations: dict[str, llvm.Value] = dict()
        self._loc = Loc.unknown()

    @contextmanager
    def change_loc(self, loc: Loc):
        old_loc = self._loc
        self._loc = loc
        try:
            yield
        finally:
            self._loc = old_loc

    def block(self, b: ir.Block) -> int:
        return self._block_map[b]

    def value(self, var: ir.Var) -> llvm.Value:
        val = self._value_map.get(var.name)
        if val is None:
            self._value_map[var.name] = val = self.builder.forward_reference(self.typeof(var))
        return val

    def set_value(self, var: ir.Var, value: llvm.Value):
        name = var.name
        fwd_ref = self._value_map.get(name)
        if fwd_ref is not None:
            self.builder.resolve_forward_reference(fwd_ref, value)
        self._value_map[name] = value
        assert self.typeof(var).type_id == value.type.type_id, \
            f"{name}: {self.typeof(var)} != {value.type}"

    def typeof(self, var: ir.Var) -> llvm.Type:
        return type_to_llvm(var.get_type(), self.builder.type_table, storage=False)

    def dtype(self, dtype: datatype.DType, storage: bool) -> llvm.Type:
        return dtype_to_llvm(dtype, self.builder.type_table, storage=storage)

    def type(self, type: ir_type.Type, storage: bool) -> llvm.Type:
        return type_to_llvm(type, self.builder.type_table, storage=storage)

    def constant(self, value: bool | int | float, ty: ir_type.TensorLikeTy) -> llvm.Value:
        dtype = ty.tensor_dtype()
        llvm_ty = type_to_llvm(ty, self.builder.type_table, storage=False)
        if is_boolean(dtype) or is_integral(dtype):
            value = int(value)
            if value < 0:
                value += 1 << dtype.bitwidth
            return self.builder.constants.integer_constant(value, llvm_ty)
        elif datatype.is_float(dtype):
            value = float(value)
            # TODO: make float_to_bits() etc. independent of the tile bytecode enums
            bits = float_to_bits(value, dtype_simple_bytecode_type(dtype))
            if dtype in DIRECTLY_SUPPORTED_FLOATS:
                return self.builder.constants.float_constant(bits, llvm_ty)
            else:
                return self.builder.constants.integer_constant(bits, llvm_ty)
        else:
            raise NotImplementedError(f"Cannot create a constant of type {ty}")

    def declare_function(self, name: str, func_ty: llvm.FunctionType) -> llvm.Value:
        if name in self._function_declarations:
            fptr = self._function_declarations[name]
            assert fptr.type.type_id == func_ty.type_id
        else:
            with self.builder.global_scope(), self.builder.function(name, func_ty) as f:
                fptr = f.value
            self._function_declarations[name] = fptr
        return fptr

    def unpack_struct(self, struct_value: llvm.Value) -> tuple[llvm.Value, ...]:
        assert isinstance(struct_value.type, llvm.AnonStructType)
        return tuple(self.builder.extract_value(struct_value, i)
                     for i in range(len(struct_value.type.fields)))


def generate_nvvm_bitcode_for_kernel(body: ir.Region,
                                     symbol: str,
                                     target_info: TargetInfo) -> bytes:
    assert isinstance(target_info, TargetInfo)
    builder = BitcodeBuilder(target_triple="nvptx64-nvidia-cuda", data_layout=DATALAYOUT_PTX)
    ctx = LLVMLoweringContext(builder=builder,
                              ir_ctx=body.ctx,
                              target_info=target_info,
                              all_blocks=body.blocks)
    builder.append_nvvm_version_metadata(2, 0)
    _lower_function(body=body,
                    name=symbol,
                    calling_convention=llvm.CallingConvention.PTX_Kernel,
                    ctx=ctx)
    return builder.build()


def _lower_function(body: ir.Region,
                    name: str,
                    calling_convention: llvm.CallingConvention,
                    ctx: LLVMLoweringContext,
                    ) -> llvm.Function:
    tt = ctx.builder.type_table
    func_ty = tt.function(tt.VOID, [ctx.typeof(p) for p in body.blocks[0].params])
    predecessors_by_block_id = _get_predecessors(body.blocks, ctx._block_map)
    with ctx.builder.function(name, func_ty, calling_convention=calling_convention) as func:
        for param, llvm_value in zip(body.blocks[0].params, func.parameters, strict=True):
            ctx.set_value(param, llvm_value)

        for block_id, (block, predecessors) in enumerate(zip(body.blocks, predecessors_by_block_id,
                                                             strict=True)):
            # Convert block parameters to PHI nodes
            if block_id > 0:
                for pred in predecessors:
                    assert len(pred.incoming_values) == len(block.params)
                for param_idx, param in enumerate(block.params):
                    incoming = [
                        (ctx.value(pred.incoming_values[param_idx]), pred.incoming_block_id)
                        for pred in predecessors
                    ]
                    phi = ctx.builder.phi(incoming)
                    ctx.set_value(param, phi)

            _lower_ops(ctx, block)
    return func


def _lower_ops(ctx: LLVMLoweringContext, ops: Sequence[ir.Operation]):
    for op in ops:
        with op.loc, ctx.change_loc(op.loc):
            try:
                if _try_rewrite_before_llvm_gen(ctx, op):
                    continue

                result_values = op.generate_llvm(ctx)

                if isinstance(result_values, llvm.Value):
                    result_values = (result_values,)
                elif result_values is None:
                    result_values = ()

                for result_var, val in zip(op.result_vars, result_values, strict=True):
                    assert isinstance(val, llvm.Value)
                    ctx.set_value(result_var, val)
            except Exception as e:
                raise InternalError(f"Internal error: {e}") from e


def _try_rewrite_before_llvm_gen(ctx: LLVMLoweringContext, op: ir.Operation) -> bool:
    # Default implementation always returns NotImplemented -- skip it
    if op.rewrite_before_llvm_gen.__func__ is op.rewrite_before_llvm_gen:
        return False

    with ir.TileBuilder(ctx.ir_ctx, ctx._loc) as ir_builder:
        result = op.rewrite_before_llvm_gen()

    if result is NotImplemented:
        return False

    if not isinstance(result, tuple):
        result = (result,)

    assert len(result) == len(op.result_vars)

    _lower_ops(ctx, ir_builder.ops)

    for old_var, new_var in zip(op.result_vars, result):
        if old_var.name != new_var.name:
            ctx.set_value(old_var, ctx.value(new_var))
    return True


@dataclass(frozen=True)
class _Predecessor:
    incoming_block_id: int
    incoming_values: tuple[ir.Var, ...]


def _get_predecessors(blocks: Sequence[ir.Block],
                      block_map: Mapping[ir.Block, int]) -> list[list[_Predecessor]]:
    from .._ir.ops import CondBranch, Branch
    ret = [[] for _ in blocks]
    for pred_id, predecessor in enumerate(blocks):
        terminator = predecessor[-1]
        if isinstance(terminator, CondBranch):
            successors = ((terminator.true_target, terminator.true_args),
                          (terminator.false_target, terminator.false_args))
        elif isinstance(terminator, Branch):
            successors = ((terminator.target, terminator.args),)
        else:
            assert isinstance(terminator, Return)
            successors = ()

        for succ, args in successors:
            succ_id = block_map[succ]
            ret[succ_id].append(_Predecessor(pred_id, args))
    # The entry block can't be jumped back to
    assert len(ret[0]) == 0
    return ret
