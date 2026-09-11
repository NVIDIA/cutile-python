# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compile a CUDA Lang host function."""

from __future__ import annotations

from dataclasses import dataclass, fields
import os
import subprocess
import tempfile
from types import FunctionType
from typing import Iterable, Iterator

from cuda.tile import _cext
from cuda.lang import _datatype as datatype
from cuda.tile._annotated_function import get_annotated_function
from cuda.tile._compile import _create_kernel_parameters
from cuda.tile._ir.ir import Var
from cuda.tile._ir.type import ArrayTy, DataclassTy, ListTy, TupleTy, Type
from cuda.tile._ir.typing_support import create_dataclass_instance
from cuda.tile._passes.ast2hir import HirMode
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
from cuda.tile._passes.hir2ir import hir2ir

from cuda.lang._compile import get_compiler_binary_path
from cuda.lang._exception import CompilerExecutionError, TypeCheckingError
from cuda.lang._ir import ir
from cuda.lang._ir.op_defs import KernelLaunch
from cuda.lang._ir.ops import cuda_lang_impl_registry
from cuda.lang._ir.type import PointerTy, ScalarTy
from cuda.lang._passes.ast2hir import get_function_hir
from cuda.lang._passes.flatten_cfg import flatten_cfg
from cuda.lang._passes.ir2mlir.host import HostIR2MLIR, _KernelLaunchBinding
from cuda.lang.compilation import KernelSignature


@dataclass(frozen=True)
class HostCompilation:
    """Compiler output retained for one typed host function."""

    signature: KernelSignature
    host_ir: ir.Block | None
    host_mlir: str | None


@dataclass(frozen=True)
class _BoundKernelLaunch:
    launch_site_index: int
    argument_sources: tuple[Var, ...]
    kernel: object
    arguments: tuple[object, ...]
    host_constant_args: tuple[bool, ...]


def _all_ops(block: ir.Block) -> Iterable[ir.Operation]:
    for operation in block:
        yield operation
        for nested in operation.nested_blocks:
            yield from _all_ops(nested)


def _collect_kernel_launches(host_body: ir.Block) -> list[KernelLaunch]:
    return [
        operation
        for operation in _all_ops(host_body)
        if isinstance(operation, KernelLaunch)
    ]


class _FakeArray:
    def __init__(self, dtype: datatype.DType, ndim: int):
        self._cuda_lang_fake_array_dtype = dtype
        self._cuda_lang_fake_array_ndim = ndim


def _fake_array(dtype: datatype.DType, ndim: int):
    return _FakeArray(dtype, ndim)


class _FakePointer:
    def __init__(self, dtype: datatype.DType):
        if not datatype.is_pointer_dtype(dtype):
            raise TypeError(f"fake pointer dtype must be a pointer dtype, got {dtype}")
        info = datatype.PointerInfo(dtype)
        if info.opaque:
            raise TypeCheckingError(
                "opaque pointers are not supported by native launch sites"
            )
        if info.memory_space != datatype.MemorySpace.GENERIC:
            raise TypeCheckingError(
                "native launch sites only support generic pointers"
            )
        self._cuda_lang_fake_pointer_dtype = info.pointee_dtype


def _fake_pointer(dtype: datatype.DType):
    return _FakePointer(dtype)


class _FakeInt(int):
    pass


class _FakeFloat(float):
    pass


def _fake_scalar(dtype: datatype.DType):
    if dtype is datatype.tensor_map_descriptor:
        from cuda.lang._stub.tensor_map import TensorMap
        return TensorMap._from_bytes(bytes(_cext._TENSOR_MAP_DESCRIPTOR_BYTES))
    if datatype.is_boolean(dtype):
        return False
    if datatype.is_integral(dtype):
        value = _FakeInt(0)
        setattr(value, "_native_source_dtype", dtype)
        return value
    if datatype.is_float(dtype):
        value = _FakeFloat(0.0)
        setattr(value, "_native_source_dtype", dtype)
        return value
    raise TypeCheckingError(f"dtype {dtype} is not supported by native launch sites")


def _make_fake_launch_argument(
    argument_type: Type,
    argument_leaves: Iterator[Var],
    sources: list[Var],
    host_constant_args: list[bool],
) -> object:
    if isinstance(argument_type, TupleTy):
        item_types = tuple(argument_type)
        return tuple(
            _make_fake_launch_argument(
                item_type,
                argument_leaves,
                sources,
                host_constant_args,
            )
            for item_type in item_types
        )

    if _cext.cconv_v3_enabled() and isinstance(argument_type, DataclassTy):
        cls = argument_type.cls
        item_names = tuple(f.name for f in fields(cls))
        item_types = argument_type.field_types
        fields_values = [
            _make_fake_launch_argument(item_type, argument_leaves, sources, host_constant_args)
            for item_name, item_type in zip(item_names, item_types, strict=True)
        ]
        return create_dataclass_instance(cls, fields_values)

    if isinstance(argument_type, ListTy):
        raise TypeCheckingError(
            "kernel launch with list argument is not supported in compiled host code"
        )

    if isinstance(argument_type, ArrayTy):
        sources.extend(
            next(argument_leaves) for _ in argument_type.flatten_aggregate()
        )
        host_constant_args.append(False)
        return _fake_array(argument_type.dtype, argument_type.ndim,)

    argument = next(argument_leaves)
    host_constant_args.append(argument.is_constant())

    if isinstance(argument_type, PointerTy):
        sources.append(argument)
        return _fake_pointer(argument_type.pointer_dtype)

    if isinstance(argument_type, ScalarTy):
        sources.append(argument)
        return _fake_scalar(argument_type.dtype)

    if argument.is_constant():
        # Identity constants are used during launch-site creation time.
        return argument.get_constant()

    raise TypeCheckingError(
        f"kernel argument type {argument_type} is not supported by native launch sites",
        loc=argument.loc,
    )


def _bind_kernel_launch(
    kernel_launch: KernelLaunch,
    launch_site_index: int,
) -> _BoundKernelLaunch:
    target = kernel_launch.launched_kernel
    argument_types = kernel_launch.kernel_argument_types

    argument_leaves = iter(kernel_launch.kernel_argument_leaves)
    sources: list[Var] = []
    arguments: list[object] = []
    host_constant_args: list[bool] = []
    for argument_type in argument_types:
        arg = _make_fake_launch_argument(
            argument_type,
            argument_leaves,
            sources,
            host_constant_args,
        )
        arguments.append(arg)
    assert next(argument_leaves, None) is None
    arguments_tuple = tuple(arguments)
    return _BoundKernelLaunch(
        launch_site_index=launch_site_index,
        argument_sources=tuple(sources),
        kernel=target,
        arguments=arguments_tuple,
        host_constant_args=tuple(host_constant_args),
    )


def _compile_native_host(mlir_text: str):
    from cuda.lang import _host_jit

    # mlir2cubin owns MLIR-to-native code generation.
    # The host_jit extension only links and loads the resulting object.
    executable = get_compiler_binary_path()
    with tempfile.TemporaryDirectory(prefix="cuda-lang-host-") as directory:
        object_path = os.path.join(directory, "host.o")
        argv = [
            executable,
            "-",
            "-o",
            object_path,
            "--gpu-name=unused",
            "--arch=unused",
            "--emit-host-object",
        ]
        try:
            subprocess.run(
                argv, input=mlir_text.encode(), capture_output=True, check=True
            )
        except subprocess.CalledProcessError as error:
            raise CompilerExecutionError(
                return_code=error.returncode,
                stderr=error.stderr.decode(),
                compiler_flags=argv,
                compiler_version=None,
            ) from None
        with open(object_path, "rb") as object_file:
            runtime_symbols = _cext._get_compiled_host_runtime_symbols()
            return _host_jit.load_object(object_file.read(), runtime_symbols)


def _compile(
    function: FunctionType,
    signature: KernelSignature,
    *,
    keep_ir: bool = False,
    keep_mlir: bool = False,
) -> _cext._CompiledHostProgram:
    """Compile one explicitly typed host function."""

    constraints = signature.parameters
    annotated = get_annotated_function(function)
    if len(annotated.pysig.parameters) != len(constraints):
        raise TypeError("host signature must contain every host function argument")

    host_hir = get_function_hir(function, mode=HirMode.ENTRY_POINT)
    parameter_names = tuple(host_hir.signature.parameters)
    ctx = ir.IRContext(execution_space="host")
    with (
        ir.TileBuilder(ctx, host_hir.body.loc) as builder,
        cuda_lang_impl_registry.as_current(),
    ):
        parameters = _create_kernel_parameters(
            constraints,
            annotated.parameter_annotations,
            parameter_names,
            host_hir.param_locs,
            ctx,
        )
        hir2ir(host_hir, parameters.aggregate_vars, ctx)
    host_body = ctx.make_block("host_entry", host_hir.body.loc)
    host_body.params = sum(
        (leaves for leaves, _ in parameters.nonconstant_flat_vars), ()
    )
    host_body.extend(builder.ops)

    eliminate_assign_ops(host_body)
    dead_code_elimination_pass(host_body)
    kernel_launches = _collect_kernel_launches(host_body)
    bound_launches = tuple(
        _bind_kernel_launch(kernel_launch, launch_site_index)
        for launch_site_index, kernel_launch in enumerate(kernel_launches)
    )
    launch_bindings = tuple(
        _KernelLaunchBinding(
            launch_site_index=bound_launch.launch_site_index,
            argument_sources=bound_launch.argument_sources,
        )
        for bound_launch in bound_launches
    )
    launch_site_descriptions = tuple(
        (
            bound_launch.kernel,
            bound_launch.arguments,
            bound_launch.host_constant_args,
        )
        for bound_launch in bound_launches
    )

    host_cfg = flatten_cfg(host_body, ctx)
    host_module = HostIR2MLIR(
        host_cfg,
        ctx,
        tuple(zip(kernel_launches, launch_bindings, strict=True)),
    )()
    host_mlir = str(host_module)
    loaded_host_code = _compile_native_host(host_mlir)
    compilation = HostCompilation(
        signature=signature,
        host_ir=host_body if keep_ir else None,
        host_mlir=host_mlir if keep_mlir else None,
    )
    return _cext._CompiledHostProgram(
        launch_site_descriptions,
        loaded_host_code,
        compilation,
    )


__all__ = ("HostCompilation",)
