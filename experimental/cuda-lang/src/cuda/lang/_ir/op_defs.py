# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Any, TYPE_CHECKING


from cuda.lang._enums import (
    CTAGroup,
    MemoryOrder,
    RoundingMode,
    SaturationMode,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
    TMALoadMode,
    TMAStoreMode,
)
from typing_extensions import override
from cuda.tile._memory_model import MemoryScope
from cuda.tile._ir.ir import MemoryEffect, add_operation_variadic
from cuda.tile._ir.type import TensorLikeTy
from cuda.tile._ir.core_ops import loosely_typed_const
from cuda.tile._ir.arithmetic_ops import astype
from cuda.lang import _datatype as datatype
from cuda.lang._enums import VectorReduction
from .ir import Operation, Var, attribute, operand
from .type import Type, VectorTy, ScalarTy, PointerTy
from .. import _llvm_bitcode as llvm
from .._passes.ir2llvm import LLVMLoweringContext, type_to_llvm
from ...tile._datatype import is_integral, is_float

if TYPE_CHECKING:
    from cuda.lang._execution import kernel


@dataclass(eq=False)
class RawLLVMIntrinsic(
    Operation, opcode="llvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var | None, ...] = operand()
    metadata_args: tuple[Any, ...] = attribute(default=())

    def generate_llvm(self, ctx):
        tt = ctx.builder.type_table
        all_operands: list[llvm.Value | llvm.Metadata] = []
        operand_types_llvm = []

        meta_iter = iter(self.metadata_args)
        for x in self.operands_:
            if x is None:
                meta = next(meta_iter)
                all_operands.append(_metadata_to_llvm(meta, ctx.builder.metadata))
                operand_types_llvm.append(tt.METADATA)
            else:
                all_operands.append(ctx.value(x))
                operand_types_llvm.append(ctx.typeof(x))

        assert tuple(meta_iter) == ()

        # Create an llvm.FunctionType for the signature
        if len(self.result_vars) == 0:
            ret_ty_llvm = tt.VOID
        elif len(self.result_vars) == 1:
            ret_ty_llvm = ctx.typeof(self.result_var)
        else:
            ret_ty_llvm = tt.struct_anonymous([ctx.typeof(r) for r in self.result_vars])
        func_ty = tt.function(ret_ty_llvm, operand_types_llvm)

        # Generate a declaration if necessary
        from .._stub._nvvm_support import mangle_intrinsic_name
        mangled_name = mangle_intrinsic_name(
                self.intrinsic,
                [None if x is None else x.get_type() for x in self.operands_])
        callee = ctx.declare_function(mangled_name, func_ty)

        # Call the intrinsic
        result = ctx.builder.call(func_ty, callee, all_operands)

        # Unpack the result
        if len(self.result_vars) == 0:
            return ()
        elif len(self.result_vars) == 1:
            return (result,)
        else:
            return ctx.unpack_struct(result)


def call_intrinsic(stub, *args: Var):
    from cuda.lang._stub._nvvm_support import match_intrinsic_signature

    name = stub._llvm_intrinsic_name
    if name is None:
        name = stub.__name__.replace("_", ".")

    prepared_operands, result_types, make_retval, metadata_args = match_intrinsic_signature(
        stub, args)
    return make_retval(add_operation_variadic(
        RawLLVMIntrinsic,
        tuple(result_types),
        intrinsic=stub._cutile_custom_implementation_handler.prefix + name,
        operands_=prepared_operands,
        metadata_args=metadata_args
    ))


def _metadata_to_llvm(meta: Any, metadata_table: llvm.MetadataTable) -> llvm.Metadata:
    if isinstance(meta, str):
        return metadata_table.string(meta)
    else:
        raise TypeError(f"Unexpected LLVM metadata type '{type(meta)}'")


@dataclass(eq=False)
class MathUnaryOperation(Operation, opcode="math_unary"):
    fn: str = attribute()
    x: Var = operand()
    approx: bool = attribute(default=False)
    flush_to_zero: bool = attribute(default=False)

    @override
    def rewrite_before_llvm_gen(self):
        from cuda.lang._passes.ir2llvm import DIRECTLY_SUPPORTED_FLOATS
        from cuda.lang._stub import llvm
        from cuda.lang._llvm_bitcode import FPClass

        if self.flush_to_zero:
            return NotImplemented

        x = self.x
        input_dtype = x.get_type().tensor_dtype()
        match self.fn:
            case "floor" if input_dtype in DIRECTLY_SUPPORTED_FLOATS:
                return call_intrinsic(llvm.floor, x)
            case "ceil" if input_dtype in DIRECTLY_SUPPORTED_FLOATS:
                return call_intrinsic(llvm.ceil, x)
            case "isnan":
                return call_intrinsic(llvm.is_fpclass, x, loosely_typed_const(FPClass.Nan))
            case "isinf":
                return call_intrinsic(llvm.is_fpclass, x, loosely_typed_const(FPClass.Inf))
            case "isfinite":
                return call_intrinsic(llvm.is_fpclass, x, loosely_typed_const(FPClass.Finite))
            case "abs" if is_integral(input_dtype):
                return call_intrinsic(llvm.abs, x, loosely_typed_const(False))
            case "abs" if is_float(input_dtype):
                return _apply_libdevice_func(x, "fabs", fast=False)
            case "sqrt" | "rsqrt" | "exp2" | "tanh" | "cosh" | "sinh":
                return _apply_libdevice_func(x, self.fn, fast=False)
            case "exp" | "log" | "log2" | "sin" | "cos" | "tan":
                return _apply_libdevice_func(x, self.fn, fast=self.approx)
            case "sincos":
                return _apply_libdevice_func(x, self.fn, fast=self.approx, return_via_args=2)

        return NotImplemented


def _apply_libdevice_func(x, base_name: str, fast: bool, *, return_via_args: int = 0):
    from cuda.lang._stub import _libdevice
    from cuda.lang._ir.op_impl.vector_impl import vector_elementwise_apply
    from cuda.lang._ir.op_impl.pointer_impl import load_pointer
    from cuda.lang._ir.ops import alloc_local_memory

    dtype = x.get_type().tensor_dtype()
    f32_fallback = False
    if dtype == datatype.float64:
        full_name = base_name
    else:
        full_name = f"fast_{base_name}f" if fast else f"{base_name}f"
        if dtype != datatype.float32:
            x = astype(x, datatype.float32)
        f32_fallback = True

    func = getattr(_libdevice, "__nv_" + full_name)

    def call(t):
        bufs = [alloc_local_memory(dtype, 1) for _ in range(return_via_args)]
        libdevice_ret = call_libdevice_function(func, t, *bufs)
        ret = []
        if isinstance(libdevice_ret, Var):
            ret.append(libdevice_ret)
        elif isinstance(libdevice_ret, tuple):
            ret.extend(libdevice_ret)
        else:
            assert libdevice_ret is None
        ret.extend([load_pointer(t) for t in bufs])
        return ret[0] if len(ret) == 1 else tuple(ret)

    x = vector_elementwise_apply(call, x)
    if f32_fallback:
        x = tuple_itemwise_apply(lambda t: astype(t, dtype), x)
    return x


def tuple_itemwise_apply(fn, x):
    if isinstance(x, tuple):
        return tuple(tuple_itemwise_apply(fn, item) for item in x)
    else:
        return fn(x)


@dataclass(eq=False)
class MathBinaryOperation(Operation, opcode="math_binary"):
    fn: str = attribute()
    lhs: Var = operand()
    rhs: Var = operand()
    approx: bool = attribute(default=False)
    propagate_nan: bool = attribute(default=False)


@dataclass(eq=False)
class VectorConstruct(Operation, opcode="vector_construct"):
    elements: tuple[Var[ScalarTy | PointerTy], ...] = operand()

    @override
    def generate_llvm(self, ctx):
        vec = ctx.builder.constants.poison(ctx.typeof(self.result_var))
        for i, x in enumerate(self.elements):
            i_val = ctx.builder.constants.integer_constant(i, ctx.builder.type_table.I32)
            vec = ctx.builder.insert_element(vec, ctx.value(x), i_val)
        return vec


@dataclass(eq=False)
class VectorInsert(Operation, opcode="vector_insert"):
    vector: Var[VectorTy] = operand()
    value: Var[ScalarTy | PointerTy] = operand()
    index: Var[ScalarTy] = operand()


@dataclass(eq=False)
class CopyAsyncBulkTensorGlobalToShared(
    Operation, opcode="copy_async_bulk_tensor_g2s", memory_effect=MemoryEffect.STORE
):
    dst_memory: Var = operand()
    tensor_map: Var = operand()
    coordinates: tuple[Var, ...] = operand()
    mbarrier: Var = operand()
    im2col_offsets: tuple[Var, ...] = operand()
    multicast_mask: Var | None = operand(default=None)
    l2_cache_hint: Var | None = operand(default=None)
    predicate: Var | None = operand(default=None)
    mode: TMALoadMode = attribute()
    is_cta_only: bool = attribute()
    cta_group: CTAGroup | None = attribute(default=None)


@dataclass(eq=False)
class CopyAsyncBulkTensorSharedToGlobal(
    Operation, opcode="copy_async_bulk_tensor_s2g", memory_effect=MemoryEffect.STORE
):
    tensor_map: Var = operand()
    src_memory: Var = operand()
    coordinates: tuple[Var, ...] = operand()
    l2_cache_hint: Var | None = operand(default=None)
    predicate: Var | None = operand(default=None)
    mode: TMAStoreMode = attribute()


@dataclass(eq=False)
class Tcgen05Copy(
    Operation, opcode="tcgen05_copy", memory_effect=MemoryEffect.STORE
):
    address: Var = operand()
    shared_memory_descriptor: Var = operand()
    shape: Tcgen05CopyShape = attribute()
    cta_group: CTAGroup = attribute()
    multicast: Tcgen05CopyMulticast | None = attribute(default=None)
    source_format: Tcgen05CopySourceFormat | None = attribute(default=None)


@dataclass(frozen=True)
class InlineAsmInput:
    index: int


@dataclass(frozen=True)
class InlineAsmOutput:
    index: int


InlineAsmPiece = str | InlineAsmInput | InlineAsmOutput


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    text: tuple[InlineAsmPiece, ...] = attribute()
    inputs: tuple[Var, ...] = operand()

    @override
    def generate_llvm(self, ctx):
        num_outputs = len(self.result_vars)
        llvm_types = []
        constraints = []
        for i, var in enumerate(itertools.chain(self.result_vars, self.inputs)):
            ty = var.get_type()
            assert ty.tensor_shape() == ()
            dtype = ty.tensor_dtype()
            code = _dtype_to_inline_ptx_constraint(dtype)
            prefix = "=" if i < num_outputs else ""
            constraints.append(prefix + code)
            llvm_types.append(ctx.typeof(var))

        pieces = []
        for p in self.text:
            if isinstance(p, str):
                pieces.append(p)
            else:
                if isinstance(p, InlineAsmInput):
                    linear_index = num_outputs + p.index
                else:
                    assert isinstance(p, InlineAsmOutput)
                    linear_index = p.index
                pieces.append(f"${linear_index}")

        tt = ctx.builder.type_table
        if num_outputs == 0:
            ret_ty = tt.VOID
        elif num_outputs == 1:
            ret_ty = llvm_types[0]
        else:
            ret_ty = tt.struct_anonymous(llvm_types[:num_outputs])

        func_ty = tt.function(ret_ty, llvm_types[num_outputs:])

        asm = ctx.builder.constants.inline_asm(func_ty, "".join(pieces), ",".join(constraints),
                                               side_effects=True)
        r = ctx.builder.call(func_ty, asm, [ctx.value(x) for x in self.inputs])

        if num_outputs == 0:
            return ()
        elif num_outputs == 1:
            return (r,)
        else:
            return ctx.unpack_struct(r)


def _dtype_to_inline_ptx_constraint(dtype: datatype.DType) -> str:
    if dtype == datatype.float32:
        return "f"
    elif dtype == datatype.float64:
        return "d"
    elif dtype == datatype.bool_:
        return "b"
    elif dtype.bitwidth == 16:
        return "h"
    elif dtype.bitwidth == 32:
        return "r"
    elif dtype.bitwidth == 64:
        return "l"
    elif dtype.bitwidth == 128:
        return "q"
    raise NotImplementedError(f"Can't map dtype {dtype} to InlineAsm constraint")


@dataclass(eq=False)
class Fence(Operation, opcode="fence", memory_effect=MemoryEffect.STORE):
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class ForeignFunction(
    Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE
):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()

    @override
    def generate_llvm(self, ctx):
        tt = ctx.builder.type_table
        if len(self.result_vars) == 0:
            ret_ty_llvm = tt.VOID
        else:
            assert len(self.result_vars) == 1
            ret_ty_llvm = ctx.typeof(self.result_var)
        func_ty = tt.function(ret_ty_llvm, [ctx.typeof(x) for x in self.operands_])
        callee = ctx.declare_function(self.function_name, func_ty)
        llvm_res = ctx.builder.call(func_ty, callee, [ctx.value(x) for x in self.operands_])
        return () if len(self.result_vars) == 0 else llvm_res


def call_libdevice_function(stub, *args: Var):
    from cuda.lang._stub._nvvm_support import match_intrinsic_signature

    name = stub.__name__
    if not name.startswith("__nv_"):
        name = "__nv_" + name

    prepared_operands, result_types, make_retval, metadata_args = match_intrinsic_signature(
        stub, args)
    assert len(metadata_args) == 0
    return make_retval(add_operation_variadic(
        ForeignFunction,
        tuple(result_types),
        function_name=name,
        operands_=tuple(prepared_operands),
    ))


@dataclass(eq=False)
class VectorGetItem(
    Operation, opcode="vector_getitem", memory_effect=MemoryEffect.LOAD
):
    x: Var[VectorTy] = operand()
    index: Var[ScalarTy] = operand()

    @override
    def generate_llvm(self, ctx):
        return ctx.builder.extract_element(ctx.value(self.x), ctx.value(self.index))


@dataclass(eq=False)
class VectorReduce(Operation, opcode="vector_reduce"):
    x: Var[VectorTy] = operand()
    kind: VectorReduction = attribute()
    propagate_nan: bool = attribute(default=False)
    reassociate: bool = attribute(default=False)


@dataclass(eq=False)
class BitCast(Operation, opcode="bitcast"):
    x: Var = operand()


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    pointer: Var[PointerTy] = operand()
    value: Var[TensorLikeTy] = operand()
    alignment: Optional[int] = attribute()

    @override
    def generate_llvm(self, ctx: LLVMLoweringContext):
        value = ctx.value(self.value)
        storage_type = type_to_llvm(self.value.get_type(), ctx.builder.type_table, storage=True)
        register_type = ctx.typeof(self.value)
        if storage_type != register_type:
            # Extend i1 -> i8 for booleans etc.
            value = ctx.builder.cast(storage_type, llvm.Cast.ZEXT, value)
        pointer = ctx.value(self.pointer)
        ctx.builder.store(pointer, value, self.alignment)


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var[PointerTy] = operand()
    alignment: Optional[int] = attribute()

    @override
    def generate_llvm(self, ctx: LLVMLoweringContext):
        pointer = ctx.value(self.pointer)
        storage_type = type_to_llvm(self.result_var.get_type(), ctx.builder.type_table,
                                    storage=True)
        register_type = ctx.typeof(self.result_var)
        value = ctx.builder.load(storage_type, pointer, self.alignment)
        if storage_type != register_type:
            # Truncate i8 -> i1 for booleans etc.
            value = ctx.builder.cast(register_type, llvm.Cast.TRUNC, value)
        return value


@dataclass(eq=False)
class AtomicStore(Operation, opcode="atomic_store", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )


@dataclass(eq=False)
class AtomicLoad(Operation, opcode="atomic_load", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )

    @property
    def has_observable_effect(self) -> bool:
        return self.mmio or self.memory_order is MemoryOrder.ACQUIRE


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()


@dataclass(eq=False)
class FmaOperation(Operation, opcode="fma"):
    x: Var = operand()
    y: Var = operand()
    z: Var = operand()
    rounding_mode: RoundingMode = attribute()
    saturation_mode: SaturationMode = attribute()
    flush_to_zero: bool = attribute()
    relu: bool = attribute()
    oob: bool = attribute()


@dataclass(eq=False)
class KernelLaunch(
    Operation, opcode="kernel_launch", memory_effect=MemoryEffect.STORE
):
    stream: Var = operand()
    block_count: tuple[Var, ...] = operand()
    thread_count: tuple[Var, ...] = operand()
    kernel_argument_leaves: tuple[Var, ...] = operand()
    kernel_argument_types: tuple[Type, ...] = attribute()
    launched_kernel: kernel = attribute()
    cooperative: bool = attribute()
    block_in_cluster_count: tuple[Var, ...] | None = operand()
    preferred_block_in_cluster_count: tuple[Var, ...] | None = operand()
    programmatic_dependent_launch: bool = attribute()
