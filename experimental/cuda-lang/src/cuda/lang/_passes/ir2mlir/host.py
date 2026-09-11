# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lower native CUDA Lang host functions to MLIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cuda.tile import _cext
from cuda.lang import _datatype as datatype
from cuda.tile._ir.ir import Var

from cuda.lang import _mlir as mlir
from cuda.lang._exception import InternalError, TypeCheckingError
from cuda.lang._ir import ir, ops
from cuda.lang._ir.op_defs import KernelLaunch
from cuda.lang._ir.type import PointerTy, ScalarTy, StreamTy
import cuda.lang._mlir.extras.types as T
from cuda.lang._passes.ir2mlir.pass_definition import (
    MLIRLoweringContext,
    create_mlir_blocks,
    lower_mlir_region,
    mlir_op_lowering,
)
from cuda.lang._passes.ir2mlir.type_conversion import (
    dtype_to_mlir_type,
    ir_type_to_mlir_type,
    mlir_constant_of_type,
    mlir_integer_cast,
)


@dataclass(frozen=True)
class _KernelLaunchBinding:
    launch_site_index: int
    argument_sources: tuple[Var, ...]


_HOST_ENTRY_SYMBOL = "cuda_lang_host_entry"
_LAUNCH_KERNEL_BUILTIN = "cuda_lang_runtime_launch_kernel"
_ENCODE_TENSOR_MAP_TILED_BUILTIN = "cuda_lang_runtime_encode_tensor_map_tiled"
_HOST_ENTRY_PARAMETER_NAMES = ("abi_arguments", "runtime", "global_lock")


@dataclass(frozen=True)
class _HostEntryABI:
    """Ordered native ABI for the physical compiled-host entry block."""

    function_type: mlir.llvm.LLVMFunctionType
    parameter_names: tuple[str, ...]

    @property
    def parameter_types(self) -> tuple[mlir.Type, ...]:
        return tuple(self.function_type.params)

    def __post_init__(self) -> None:
        if len(self.parameter_names) != len(self.parameter_types):
            raise ValueError("host entry ABI parameter names and types must match")


@dataclass(frozen=True)
class _HostRuntimeArguments:
    """Physical arguments supplied by the compiled-host runtime."""

    abi_arguments: mlir.Value
    runtime: mlir.Value
    global_lock: mlir.Value


@dataclass(kw_only=True)
class HostLoweringContext(MLIRLoweringContext):
    """Data required while lowering a native CUDA Lang host function."""

    kernel_launch_bindings: dict[KernelLaunch, _KernelLaunchBinding] = field(
        default_factory=dict
    )
    continuation_index: int = 0
    physical_entry_block: mlir.Block | None = None
    runtime_arguments: _HostRuntimeArguments | None = None
    runtime_error_block: mlir.Block | None = None
    host_entry_function_type: mlir.llvm.LLVMFunctionType | None = None
    pointer_type: mlir.llvm.LLVMPointerType = field(
        default_factory=mlir.llvm.LLVMPointerType
    )
    print_format_index: int = 0
    printf_declared: bool = False
    launch_kernel_function_type: mlir.llvm.LLVMFunctionType | None = None
    encode_tensor_map_tiled_function_type: mlir.llvm.LLVMFunctionType | None = None


class HostIR2MLIR:
    """Build a native host function using shared operation lowering."""

    def __init__(
        self,
        region: ir.Region,
        ctx: ir.IRContext,
        kernel_launch_bindings: Sequence[
            tuple[KernelLaunch, _KernelLaunchBinding]
        ] = (),
    ):
        self.context = HostLoweringContext(
            region=region,
            ir_context=ctx,
            kernel_launch_bindings=dict(kernel_launch_bindings),
        )
        entry_type = mlir.llvm.LLVMFunctionType(
            returnType=T.i32(),
            params=(
                self.context.pointer_type,
                self.context.pointer_type,
                self.context.pointer_type,
            ),
            varArg=False,
        )
        self.context.host_entry_function_type = entry_type
        self.host_entry_abi = _HostEntryABI(
            function_type=entry_type,
            parameter_names=_HOST_ENTRY_PARAMETER_NAMES,
        )

    def __call__(self) -> mlir.Operation:
        self.setup_func_op()
        self.setup_blocks()
        self.lower_physical_entry()
        lower_mlir_region(self.context)
        assert self.context.module_op is not None
        return self.context.module_op

    def setup_func_op(self) -> None:
        context = self.context
        with mlir.Block().append_here() as top_block:
            module_region = mlir.Region()
            mlir.add_ModuleOp(bodyRegion=module_region)
        context.module_op = top_block[0]

        with module_region.new_block().append_here() as module_block:
            body_region = mlir.Region()
            mlir.llvm.add_LLVMFuncOp(
                sym_name=_HOST_ENTRY_SYMBOL,
                function_type=self.host_entry_abi.function_type,
                body=body_region,
            )
        context.func_op = module_block.operations[-1]

    def setup_blocks(self) -> None:
        context = self.context
        physical_args = tuple(
            mlir.Value(parameter_type, parameter_name)
            for parameter_type, parameter_name in zip(
                self.host_entry_abi.parameter_types,
                self.host_entry_abi.parameter_names,
                strict=True,
            )
        )
        context.physical_entry_block = context.function_region.new_block(
            args=physical_args,
            block_id="physical_entry",
        )
        context.runtime_arguments = _HostRuntimeArguments(
            **dict(zip(self.host_entry_abi.parameter_names, physical_args, strict=True))
        )
        create_mlir_blocks(context)

        entry_type = context.host_entry_function_type
        assert entry_type is not None
        error_status = mlir.Value(entry_type.returnType, "error_status")
        context.runtime_error_block = context.function_region.new_block(
            args=(error_status,), block_id="runtime_error"
        )
        with context.runtime_error_block.append_here():
            mlir.llvm.add_ReturnOp(arg=error_status)

    def lower_physical_entry(self) -> None:
        context = self.context
        physical_entry = context.physical_entry_block
        assert physical_entry is not None
        logical_entry = context.region.blocks[0]
        with physical_entry.append_here():
            arguments = tuple(
                _load_abi_argument(context, parameter, slot)
                for slot, parameter in enumerate(logical_entry.params)
            )
            mlir.cf.add_BranchOp(
                dest=context.block_map[logical_entry].label,
                destOperands=arguments,
            )


@mlir_op_lowering(device=False)
def lower_return(
    context: HostLoweringContext, operation: ops.Return
) -> Sequence[mlir.Value]:
    assert operation.result_vars == (), "compiled host returns must not produce values"
    entry_type = context.host_entry_function_type
    assert entry_type is not None
    success = mlir_constant_of_type(entry_type.returnType, 0)
    mlir.llvm.add_ReturnOp(arg=success)
    return ()


def _abi_argument_address(
    context: HostLoweringContext, slot: int
) -> mlir.Value:
    runtime_arguments = context.runtime_arguments
    assert runtime_arguments is not None
    address_slot = mlir.llvm.add_GEPOp(
        res_type=context.pointer_type,
        base=runtime_arguments.abi_arguments,
        dynamicIndices=(),
        rawConstantIndices=(slot,),
        elem_type=context.pointer_type,
    )
    return mlir.llvm.add_LoadOp(
        res_type=context.pointer_type, addr=address_slot
    )


def _load_abi_argument(
    context: HostLoweringContext, value: Var, slot: int
) -> mlir.Value:
    address = _abi_argument_address(context, slot)
    ty = value.get_type()
    if isinstance(ty, PointerTy | StreamTy):
        return mlir.llvm.add_LoadOp(
            res_type=ir_type_to_mlir_type(ty), addr=address
        )
    if not isinstance(ty, ScalarTy):
        raise TypeCheckingError(
            f"host ABI value of type {ty} is not supported", loc=value.loc
        )
    return mlir.llvm.add_LoadOp(
        res_type=ir_type_to_mlir_type(ty), addr=address
    )


def _promote_printf_argument(
    context: HostLoweringContext, argument: Var
) -> mlir.Value:
    """Default argument promotion for C variadic function"""
    ty = argument.get_type()
    value = context.get_var(argument)
    if isinstance(ty, PointerTy):
        return value
    if not isinstance(ty, ScalarTy):
        raise TypeCheckingError(
            f"host printf does not support argument type {ty}",
            loc=argument.loc,
        )
    dtype = ty.dtype
    if datatype.is_boolean(dtype):
        return mlir.arith.add_ExtUIOp(out_type=T.i32(), in_=value)
    if datatype.is_integral(dtype) and dtype.bitwidth < 32:
        converter = (
            mlir.arith.add_ExtSIOp
            if datatype.is_signed(dtype)
            else mlir.arith.add_ExtUIOp
        )
        return converter(out_type=T.i32(), in_=value)
    if datatype.is_float(dtype) and dtype.bitwidth < 64:
        return mlir.arith.add_ExtFOp(out_type=T.f64(), in_=value)
    return value


@mlir_op_lowering(device=False)
def lower_printf(
    context: HostLoweringContext, operation: ops.TilePrintf
) -> Sequence[mlir.Value]:
    assert context.module_op is not None

    # Store format string as a module global constant
    encoded_format = operation.format.encode() + b"\0"
    symbol = f"cuda_lang_host_format_{context.print_format_index}"
    context.print_format_index += 1
    with context.module_op.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_GlobalOp(
            global_type=mlir.llvm.LLVMArrayType(
                elementType=T.i8(),
                numElements=len(encoded_format),
            ),
            constant=True,
            sym_name=symbol,
            linkage=mlir.llvm.Linkage.Internal,
            value=mlir.StringAttr(value=encoded_format.decode()),
            visibility_=mlir.llvm.Visibility.Default,
            initializer=mlir.Region(),
        )
    # Take address of the format string
    format_address = mlir.llvm.add_AddressOfOp(
        res_type=context.pointer_type,
        global_name=symbol,
    )
    # Get the printf function address
    function_type = mlir.llvm.LLVMFunctionType(
        returnType=T.i32(),
        params=(context.pointer_type,),
        varArg=True,
    )
    if not context.printf_declared:
        with context.module_op.regions[0].blocks[0].prepend_here():
            mlir.llvm.add_LLVMFuncOp(
                sym_name="printf",
                linkage=mlir.llvm.Linkage.External,
                body=mlir.Region(),
                function_type=function_type,
            )
        context.printf_declared = True
    mlir.llvm.add_CallOp(
        result_type=T.i32(),
        var_callee_type=function_type,
        callee="printf",
        callee_operands=(
            format_address,
            *(_promote_printf_argument(context, arg) for arg in operation.args),
        ),
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )
    return [None]


def _allocate_entry_storage(
    context: HostLoweringContext,
    count: int,
    element_type: mlir.Type,
    *,
    alignment: int | None = None,
) -> mlir.Value:
    # The alloca executes once at function entry even when the corresponding
    # launch occurs in a loop.
    assert context.physical_entry_block is not None
    with context.physical_entry_block.prepend_here():
        array_size = mlir_constant_of_type(T.i64(), count)
        return mlir.llvm.add_AllocaOp(
            res_type=context.pointer_type,
            arraySize=array_size,
            elem_type=element_type,
            alignment=alignment,
        )


def _store_array_item(
    context: HostLoweringContext,
    base: mlir.Value,
    index: int,
    value: mlir.Value,
    element_type: mlir.Type,
) -> None:
    address = mlir.llvm.add_GEPOp(
        res_type=context.pointer_type,
        base=base,
        dynamicIndices=(),
        rawConstantIndices=(index,),
        elem_type=element_type,
    )
    mlir.llvm.add_StoreOp(value=value, addr=address)


def _materialize_array(
    context: HostLoweringContext,
    values: Sequence[mlir.Value],
    element_type: mlir.Type,
) -> mlir.Value:
    storage = _allocate_entry_storage(
        context,
        len(values),
        element_type,
    )
    for index, value in enumerate(values):
        if value.type != element_type:
            raise InternalError(
                f"array element has type {value.type}, expected {element_type}"
            )
        _store_array_item(
            context,
            storage,
            index,
            value,
            element_type,
        )
    return storage


def _materialize_scalar_value(
    context: HostLoweringContext,
    source: Var,
    source_type: ScalarTy,
) -> mlir.Value:
    if source.is_constant():
        return mlir_constant_of_type(
            ir_type_to_mlir_type(source_type),
            source.get_constant(),
        )
    return context.get_var(source)


def _cast_integer(
    context: HostLoweringContext,
    value: Var,
    target_type: mlir.IntegerType,
    *,
    error_message: str,
) -> mlir.Value:
    ty = value.get_type()
    if not isinstance(ty, ScalarTy) or not datatype.is_integral(ty.dtype):
        raise TypeCheckingError(error_message, loc=value.loc)
    return mlir_integer_cast(
        _materialize_scalar_value(context, value, ty),
        target_type,
        signed=datatype.is_signed(ty.dtype),
    )


def _materialize_integer_array(
    context: HostLoweringContext,
    values: Sequence[Var],
    element_type: mlir.IntegerType,
    *,
    error_message: str,
) -> mlir.Value:
    return _materialize_array(
        context,
        tuple(
            _cast_integer(
                context,
                value,
                element_type,
                error_message=error_message,
            )
            for value in values
        ),
        element_type,
    )


def _materialize_launch_argument_array(
    context: HostLoweringContext,
    argument_addresses: Sequence[mlir.Value],
) -> mlir.Value:
    if not argument_addresses:
        return mlir.llvm.add_ZeroOp(res_type=context.pointer_type)
    return _materialize_array(
        context,
        argument_addresses,
        context.pointer_type,
    )


def _get_address_of(
    context: HostLoweringContext,
    value: mlir.Value,
) -> mlir.Value:
    return _materialize_array(
        context,
        (value,),
        value.type,
    )


def _pad_dim3(
    context: HostLoweringContext,
    dimensions: Sequence[Var],
) -> tuple[mlir.Value, mlir.Value, mlir.Value]:
    values = [
        _cast_integer(
            context, value, T.i64(),
            error_message="kernel launch dimensions must be integral scalars",
        )
        for value in dimensions
    ]
    values.extend(
        mlir_constant_of_type(T.i64(), 1) for _ in range(3 - len(values))
    )
    return values[0], values[1], values[2]


def _materialize_kernel_argument_pointer(
    context: HostLoweringContext,
    source: Var,
) -> mlir.Value:
    source_type = source.get_type()
    if isinstance(source_type, ScalarTy):
        value = _materialize_scalar_value(context, source, source_type)
        value = _normalize_scalar_launch_argument(context, value, source_type, loc=source.loc)
        return _get_address_of(context, value)

    if source.is_constant():
        raise InternalError(
            "non-scalar constant should not be materialized as a native launch "
            "argument"
        )
    return _get_address_of(context, context.get_var(source))


def _normalize_scalar_launch_argument(
    context: HostLoweringContext,
    value: mlir.Value,
    source_type: ScalarTy,
    *,
    loc,
) -> mlir.Value:
    del context
    dtype = source_type.dtype
    if dtype is datatype.tensor_map_descriptor:
        return value
    if datatype.is_boolean(dtype):
        return mlir.arith.add_ExtUIOp(out_type=T.i64(), in_=value)
    if datatype.is_integral(dtype):
        if dtype.bitwidth == 64:
            return value
        assert dtype.bitwidth == 32
        converter = (
            mlir.arith.add_ExtSIOp
            if datatype.is_signed(dtype)
            else mlir.arith.add_ExtUIOp
        )
        return converter(out_type=T.i64(), in_=value)
    if datatype.is_float(dtype):
        if dtype.bitwidth == 64:
            return value
        assert dtype.bitwidth == 32
        return mlir.arith.add_ExtFOp(out_type=T.f64(), in_=value)
    raise TypeCheckingError(
        f"kernel launch scalar dtype {dtype} is not supported",
        loc=loc,
    )


def _optional_dim3(
    context: HostLoweringContext,
    dimensions: Sequence[Var] | None,
) -> tuple[mlir.Value, mlir.Value, mlir.Value]:
    if dimensions is None:
        zero = mlir_constant_of_type(T.i64(), 0)
        return zero, zero, zero

    return _pad_dim3(context, dimensions)


def _declare_launch_kernel_builtin(
    context: HostLoweringContext,
) -> mlir.llvm.LLVMFunctionType:
    function_type = context.launch_kernel_function_type
    if function_type is not None:
        return function_type
    pointer_type = context.pointer_type
    i32 = T.i32()
    i64 = T.i64()
    function_type = mlir.llvm.LLVMFunctionType(
        returnType=i32,
        params=(
            pointer_type,  # compiled-host launch runtime
            i32,  # launch-site index
            pointer_type,  # stream
            i64, i64, i64,  # grid
            i64, i64, i64,  # block
            i64, i64, i64,  # cluster
            i64, i64, i64,  # preferred cluster
            i32,  # cluster present
            i32,  # preferred cluster present
            pointer_type,  # kernel arguments
            i32,  # cooperative
            i32,  # programmatic dependent launch
        ),
        varArg=False,
    )
    assert context.module_op is not None
    with context.module_op.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_LLVMFuncOp(
            sym_name=_LAUNCH_KERNEL_BUILTIN,
            linkage=mlir.llvm.Linkage.External,
            body=mlir.Region(),
            function_type=function_type,
        )
    context.launch_kernel_function_type = function_type
    return function_type


def _declare_encode_tensor_map_tiled_builtin(
    context: HostLoweringContext,
) -> mlir.llvm.LLVMFunctionType:
    function_type = context.encode_tensor_map_tiled_function_type
    if function_type is not None:
        return function_type
    pointer_type = context.pointer_type
    i32 = T.i32()
    function_type = mlir.llvm.LLVMFunctionType(
        returnType=i32,
        params=(
            pointer_type,  # GlobalLock
            pointer_type,  # destination CUtensorMap
            i32,  # CUtensorMapDataType
            i32,  # rank
            pointer_type,  # global base address
            pointer_type,  # global dimensions
            pointer_type,  # global element strides
            pointer_type,  # tile dimensions
            i32,  # CUtensorMapInterleave
            i32,  # CUtensorMapSwizzle
            i32,  # CUtensorMapL2promotion
            i32,  # CUtensorMapFloatOOBfill
        ),
        varArg=False,
    )
    assert context.module_op is not None
    with context.module_op.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_LLVMFuncOp(
            sym_name=_ENCODE_TENSOR_MAP_TILED_BUILTIN,
            linkage=mlir.llvm.Linkage.External,
            body=mlir.Region(),
            function_type=function_type,
        )
    context.encode_tensor_map_tiled_function_type = function_type
    return function_type


def _success_continuation(
    context: HostLoweringContext,
    status: mlir.Value,
) -> mlir.Block:
    assert context.runtime_error_block is not None
    zero = mlir_constant_of_type(status.type, 0)
    succeeded = mlir.arith.add_CmpIOp(
        predicate=mlir.arith.CmpIPredicate.eq,
        lhs=status,
        rhs=zero,
    )
    continuation = context.function_region.new_block(
        block_id=f"after_launch_{context.continuation_index}"
    )
    context.continuation_index += 1
    mlir.cf.add_CondBranchOp(
        condition=succeeded,
        trueDest=continuation.label,
        falseDest=context.runtime_error_block.label,
        trueDestOperands=(),
        falseDestOperands=(status,),
    )
    return continuation


@mlir_op_lowering(device=False)
def lower_create_tensor_map(
    context: HostLoweringContext,
    operation: ops.CreateTensorMap,
) -> Sequence[mlir.Value]:
    rank = len(operation.tile_shape)

    descriptor_type = dtype_to_mlir_type(datatype.tensor_map_descriptor)
    descriptor = _allocate_entry_storage(
        context,
        1,
        descriptor_type,
        alignment=_cext._TENSOR_MAP_DESCRIPTOR_ALIGNMENT,
    )
    tensor_map_extent_error = (
        "tensor-map dimensions and strides must be integral scalars"
    )
    global_dimensions = _materialize_integer_array(
        context,
        operation.array_shape,
        T.i64(),
        error_message=tensor_map_extent_error,
    )
    global_element_strides = _materialize_integer_array(
        context,
        operation.array_strides,
        T.i64(),
        error_message=tensor_map_extent_error,
    )
    tile_dimensions = _materialize_integer_array(
        context,
        operation.tile_shape,
        T.i64(),
        error_message="tensor-map tile dimensions must be integral scalars",
    )
    function_type = _declare_encode_tensor_map_tiled_builtin(context)
    runtime_arguments = context.runtime_arguments
    assert runtime_arguments is not None
    callee_operands = (
        runtime_arguments.global_lock,
        descriptor,
        mlir_constant_of_type(
            T.i32(),
            operation.data_type.value,
        ),
        mlir_constant_of_type(T.i32(), rank),
        context.get_var(operation.base_ptr),
        global_dimensions,
        global_element_strides,
        tile_dimensions,
        mlir_constant_of_type(T.i32(), operation.interleave.value),
        mlir_constant_of_type(T.i32(), operation.swizzle.value),
        mlir_constant_of_type(T.i32(), operation.l2_promotion.value),
        mlir_constant_of_type(T.i32(), operation.oob_fill.value),
    )
    if tuple(value.type for value in callee_operands) != tuple(
        function_type.params
    ):
        raise InternalError(
            f"lowered operands do not match the ABI of "
            f"{_ENCODE_TENSOR_MAP_TILED_BUILTIN!r}"
        )
    status = mlir.llvm.add_CallOp(
        result_type=function_type.returnType,
        callee=_ENCODE_TENSOR_MAP_TILED_BUILTIN,
        callee_operands=callee_operands,
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )
    assert status is not None
    context.insertion_block = _success_continuation(context, status)
    with context.insertion_block.append_here():
        return (mlir.llvm.add_LoadOp(
            res_type=descriptor_type, addr=descriptor,
            alignment=_cext._TENSOR_MAP_DESCRIPTOR_ALIGNMENT,
        ),)


@mlir_op_lowering(device=False)
def lower_kernel_launch(
    context: HostLoweringContext,
    operation: KernelLaunch,
) -> Sequence[mlir.Value]:
    try:
        binding = context.kernel_launch_bindings[operation]
    except KeyError:
        raise TypeCheckingError(
            "missing native launch binding",
            loc=operation.loc,
        ) from None
    runtime_arguments = context.runtime_arguments
    assert runtime_arguments is not None

    argument_array = _materialize_launch_argument_array(
        context,
        tuple(
            _materialize_kernel_argument_pointer(context, source)
            for source in binding.argument_sources
        ),
    )
    stream_type = operation.stream.get_type()
    if isinstance(stream_type, StreamTy):
        stream = context.get_var(operation.stream)
    else:
        stream = mlir.llvm.add_IntToPtrOp(
            res_type=context.pointer_type,
            arg=context.get_var(operation.stream),
        )
    grid = _pad_dim3(context, operation.block_count)
    block = _pad_dim3(context, operation.thread_count)
    cluster = _optional_dim3(context, operation.block_in_cluster_count)
    preferred_cluster = _optional_dim3(
        context,
        operation.preferred_block_in_cluster_count,
    )
    function_type = _declare_launch_kernel_builtin(context)
    callee_operands = (
        runtime_arguments.runtime,
        mlir_constant_of_type(T.i32(), binding.launch_site_index),
        stream,
        *grid,
        *block,
        *cluster,
        *preferred_cluster,
        mlir_constant_of_type(
            T.i32(),
            int(operation.block_in_cluster_count is not None),
        ),
        mlir_constant_of_type(
            T.i32(),
            int(operation.preferred_block_in_cluster_count is not None),
        ),
        argument_array,
        mlir_constant_of_type(T.i32(), int(operation.cooperative)),
        mlir_constant_of_type(
            T.i32(),
            int(operation.programmatic_dependent_launch),
        ),
    )
    if tuple(value.type for value in callee_operands) != tuple(
        function_type.params
    ):
        raise InternalError(
            f"lowered operands do not match the ABI of "
            f"{_LAUNCH_KERNEL_BUILTIN!r}"
        )
    status = mlir.llvm.add_CallOp(
        result_type=function_type.returnType,
        callee=_LAUNCH_KERNEL_BUILTIN,
        callee_operands=callee_operands,
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )
    assert status is not None
    context.insertion_block = _success_continuation(context, status)
    return ()
