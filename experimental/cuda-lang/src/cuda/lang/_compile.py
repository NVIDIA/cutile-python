# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from functools import total_ordering
import sys
import tempfile
from typing import Sequence
from dataclasses import dataclass
import os.path
from types import FunctionType
import subprocess

from cuda.tile._passes.hir2ir import hir2ir
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
from cuda.tile._compile import _create_kernel_parameters, get_sm_arch
from cuda.tile._annotated_function import (
    AnnotatedFunction,
    LeafAnnotationNode,
    ParameterAnnotationNode,
    get_annotated_function,
)
from cuda.tile._cext import get_compute_capability as _get_compute_capability
from ._compiler_options import CompilerOptions
from cuda.lang._logging import get_log_flags
from cuda.lang._ir import hir as hir_ir
from cuda.lang._ir import ir
from cuda.lang._passes.ast2hir import get_function_hir
from cuda.lang._passes.ir2mlir import ir2mlir
from cuda.lang._passes.flatten_cfg import flatten_cfg
from cuda.lang._passes.simt_semantics import simt_semantic_analysis
from cuda.lang._passes.handle_dyn_shared_mem import handle_dynamic_shared_memory
from cuda.lang._passes.hoist_tensor_map import hoist_tensor_maps, HoistedTensorMap
from cuda.lang.compilation import (
    KernelSignature,
    ParameterConstraint,
    ScalarConstraint,
    ArrayConstraint,
    ListConstraint,
    ConstantConstraint,
)
from cuda.lang._exception import CompilerExecutionError
from ._execution import kernel
from cuda.lang._ir.ops import cuda_lang_impl_registry
from ._ir._host_program import HostProgram, get_host_programs_by_var
import contextlib


@dataclass(frozen=True)
class MLIR2CubinResult:
    cubin: bytes
    stderr: bytes
    ptx: str | None


def mlir2cubin(
    mlir_text: str,
    gpu_name: str,
    arch: str,
    emit_ptx: bool = False,
) -> MLIR2CubinResult:
    executable = get_compiler_binary_path()
    argv = [executable, "-", "-o", "-", f"--gpu-name={gpu_name}", f"--arch={arch}"]
    custom_flags = os.environ.get("CUDA_LANG_MLIR2CUBIN_FLAGS", None)

    if custom_flags is not None:
        argv.extend(custom_flags.split())

    with contextlib.ExitStack() as ec:
        ptx_file, ptx_src = None, None

        if emit_ptx:
            ptx_file = ec.enter_context(tempfile.NamedTemporaryFile(mode="w+t"))
            argv.extend(["--dump-ptx=" + ptx_file.name])

        try:
            completed = subprocess.run(
                argv, input=mlir_text.encode(), capture_output=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise CompilerExecutionError(
                return_code=e.returncode,
                stderr=e.stderr.decode(),
                compiler_flags=argv,
                compiler_version=None,
            )

        if emit_ptx:
            assert ptx_file is not None
            ptx_file.seek(0)
            ptx_src = ptx_file.read()

    return MLIR2CubinResult(completed.stdout, completed.stderr, ptx_src)


def get_compiler_binary_path() -> str:
    binary_name = "mlir2cubin"
    if os.name == "nt":
        binary_name += ".exe"
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin", binary_name)


@total_ordering
@dataclass(frozen=True)
class ComputeCapability:
    major: int
    minor: int

    def __lt__(self: "ComputeCapability", other: "ComputeCapability | tuple[int, int]"):
        match other:
            case tuple():
                assert len(other) == 2
                return (self.major, self.minor) < other
            case ComputeCapability():
                return (self.major, self.minor) < (other.major, other.minor)

    def __iter__(self):
        yield self.major
        yield self.minor

    @property
    def arch(self):
        return f"compute_{self.major}{self.minor}"

    @property
    def gpu_name(self):
        return f"sm_{self.major}{self.minor}"


def get_compute_capability() -> ComputeCapability:
    return ComputeCapability(*_get_compute_capability())


@dataclass
class CompilationResult:
    kernel_signatures: Sequence[KernelSignature]
    dyn_smem_size_program: HostProgram | None
    hoisted_tensor_maps: list[HoistedTensorMap]

    hir: hir_ir.Function | None = None
    final_ir: ir.Region | None = None
    mlir: str | None = None
    stderr: bytes | None = None
    ptx: str | None = None
    cubin: bytes | None = None


def get_function_ir(
    function: hir_ir.Function,
    signature: KernelSignature,
    ctx: ir.IRContext,
    parameter_annotations: Sequence[ParameterAnnotationNode] | None = None,
) -> ir.Block:
    if parameter_annotations is None:
        parameter_annotations = [LeafAnnotationNode(constant=False)] * len(
            signature.parameters
        )
    parameter_names = function.signature.parameters.keys()
    with (
        ir.TileBuilder(ctx, function.body.loc) as builder,
        cuda_lang_impl_registry.as_current(),
    ):
        params = _create_kernel_parameters(
            signature.parameters,
            parameter_annotations,
            parameter_names,
            function.param_locs,
            ctx,
        )
        hir2ir(function, params.aggregate_vars, ctx)
    func_body = ctx.make_block("entry", function.body.loc)
    func_body.params = sum((vars for vars, _ in params.nonconstant_flat_vars), ())
    func_body.extend(builder.ops)
    return func_body


def _transform_ir(
    func_ir: ir.Block, ctx: ir.IRContext
) -> tuple[HostProgram | None, list[HoistedTensorMap]]:
    simt_semantic_analysis(func_ir, ctx)

    host_program_by_var = get_host_programs_by_var(func_ir)
    dyn_smem_size_program = handle_dynamic_shared_memory(func_ir, host_program_by_var)
    hoisted_tensor_maps = hoist_tensor_maps(func_ir, host_program_by_var)

    eliminate_assign_ops(func_ir)
    dead_code_elimination_pass(func_ir)

    return dyn_smem_size_program, hoisted_tensor_maps


def compile_simt(
    function: AnnotatedFunction | FunctionType,
    signatures: Sequence[KernelSignature],
    gpu_name: str | None = None,
    arch: str | None = None,
    compiler_options: CompilerOptions = CompilerOptions(),
    ctx: ir.IRContext | None = None,
    keep_hir: bool = False,
    keep_final_ir: bool = False,
    keep_mlir: bool = False,
    keep_ptx: bool = False,
    log_hir: bool = False,
    log_ir: bool = False,
    log_mlir: bool = False,
    log_ptx: bool = False,
) -> CompilationResult:
    match function:
        case FunctionType():
            function = get_annotated_function(function)
        case kernel():
            function = get_annotated_function(function._pyfunc)

    log_flags = deepcopy(get_log_flags())
    log_flags.log_hir |= log_hir
    log_flags.log_ir |= log_ir
    log_flags.log_mlir |= log_mlir
    log_flags.log_ptx |= log_ptx

    def _dump(phase: str, contents: object) -> None:
        logging_template = (
            "=" * 20 + " cuda.lang {header} dump: " + "=" * 20 + "\n" + "{body}" + "\n"
        )
        print(
            logging_template.format(header=str(phase), body=str(contents)),
            file=sys.stderr,
        )

    func_hir = get_function_hir(function.pyfunc, entry_point=True)
    if log_flags.log_hir:
        _dump("HIR", func_hir.body)

    [signature] = signatures
    if signature.symbol is None:
        signature = signature.with_mangled_symbol(function.pyfunc.__name__)

    ctx = ctx or ir.IRContext(log_ir_on_error=log_flags.log_hir or log_flags.log_ir)

    func_ir = get_function_ir(func_hir, signature, ctx, function.parameter_annotations)

    if log_flags.log_ir:
        _dump("IR (pre-transforms)", func_ir)

    dyn_smem_size_program, hoisted_tensor_maps = _transform_ir(func_ir, ctx)

    if log_flags.log_ir:
        _dump("IR (post-transforms)", func_ir)

    flattened_ir = flatten_cfg(func_ir, ctx)

    if log_flags.log_flattened_ir:
        _dump("Flattened IR", flattened_ir)

    mlir_module = ir2mlir(signature, flattened_ir, ctx, compiler_options)
    mlir_text = str(mlir_module)

    if log_flags.log_mlir:
        _dump("MLIR", mlir_text)

    if gpu_name is None or arch is None:
        cc = get_compute_capability()
        suffix = "a" if cc >= (9, 0) else ""
        gpu_name = gpu_name or cc.gpu_name + suffix
        arch = arch or cc.arch + suffix

    need_ptx = log_flags.log_ptx or keep_ptx
    compiled = mlir2cubin(
        mlir_text, gpu_name=gpu_name, arch=arch, emit_ptx=need_ptx
    )

    if need_ptx:
        assert compiled.ptx is not None

    if log_flags.log_ptx:
        _dump("PTX", compiled.ptx)

    return CompilationResult(
        kernel_signatures=[signature],
        dyn_smem_size_program=dyn_smem_size_program,
        hoisted_tensor_maps=hoisted_tensor_maps,
        hir=func_hir if keep_hir else None,
        final_ir=flattened_ir if keep_final_ir else None,
        mlir=mlir_text if keep_mlir else None,
        ptx=compiled.ptx if keep_ptx else None,
        stderr=compiled.stderr,
        cubin=compiled.cubin,
    )


__all__ = (
    "mlir2cubin",
    "get_compiler_binary_path",
    "compile_simt",
    "get_sm_arch",
    "get_function_hir",
    "KernelSignature",
    "ParameterConstraint",
    "ScalarConstraint",
    "ArrayConstraint",
    "ListConstraint",
    "ConstantConstraint",
    "CompilationResult",
)
