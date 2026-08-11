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

from cuda.tile._passes.ast2hir import HirMode
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
from cuda.lang._target import TargetInfo
from ._execution import kernel
from cuda.lang._ir.ops import cuda_lang_impl_registry
from ._ir._host_program import HostProgram, get_host_programs_by_var
from ._timing import CompilationTimer, CompilationTimings
import contextlib


@dataclass(frozen=True)
class MLIR2CubinResult:
    cubin: bytes
    stderr: bytes
    ptx: str | None
    nvvm: str | None
    timings_ns: dict[str, int] | None


def _read_mlir2cubin_timings(filename: str) -> dict[str, int]:
    try:
        with open(filename, encoding="utf-8") as timing_file:
            phases_ns = {}
            for line in timing_file:
                name, separator, elapsed_ns_text = line.rstrip("\n").partition("=")
                if not separator or not name or name in phases_ns:
                    raise ValueError
                elapsed_ns = int(elapsed_ns_text)
                if elapsed_ns < 0:
                    raise ValueError
                phases_ns[name] = elapsed_ns
    except (OSError, ValueError) as error:
        raise RuntimeError("Failed to read mlir2cubin timing output") from error

    phases_ns.pop("total", None)
    if not phases_ns:
        raise RuntimeError("mlir2cubin timing output is empty")
    return phases_ns


def mlir2cubin(
    mlir_text: str,
    gpu_name: str,
    arch: str,
    *,
    emit_ptx: bool = False,
    emit_nvvm: bool = False,
    opt_level: int | None = None,
    generate_line_info: bool = False,
    ptx_compiler_options: Sequence[str] = (),
    emit_timings: bool = False,
) -> MLIR2CubinResult:
    executable = get_compiler_binary_path()
    argv = [executable, "-", "-o", "-", f"--gpu-name={gpu_name}", f"--arch={arch}"]
    custom_flags = os.environ.get("CUDA_LANG_MLIR2CUBIN_FLAGS", None)

    argv.extend(
        f"--ptx-compiler-option={option}" for option in ptx_compiler_options
    )
    if opt_level is not None:
        argv.append(f"--opt={opt_level}")
    if generate_line_info:
        argv.append("--generate-device-line-info")

    if custom_flags is not None:
        argv.extend(custom_flags.split())

    with contextlib.ExitStack() as ec:
        nvvm_file, nvvm_src = None, None
        ptx_file, ptx_src = None, None
        timing_filename, timings_ns = None, None

        if emit_timings:
            timing_dir = ec.enter_context(tempfile.TemporaryDirectory())
            timing_filename = os.path.join(timing_dir, "timings.txt")
            argv.append("--timing-output=" + timing_filename)

        if emit_nvvm:
            nvvm_file = ec.enter_context(tempfile.NamedTemporaryFile(mode="w+t"))
            argv.extend(["--dump-nvvm=" + nvvm_file.name])

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

        if emit_nvvm:
            assert nvvm_file is not None
            nvvm_file.seek(0)
            nvvm_src = nvvm_file.read()

        if emit_ptx:
            assert ptx_file is not None
            ptx_file.seek(0)
            ptx_src = ptx_file.read()

        if timing_filename is not None:
            timings_ns = _read_mlir2cubin_timings(timing_filename)

    return MLIR2CubinResult(
        completed.stdout,
        completed.stderr,
        ptx_src,
        nvvm_src,
        timings_ns,
    )


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
    timings: CompilationTimings | None = None

    stderr: bytes | None = None
    hir: hir_ir.Function | None = None
    final_ir: ir.Region | None = None
    mlir: str | None = None
    nvvm: str | None = None
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
    func_ir: ir.Block,
    ctx: ir.IRContext,
    timer: CompilationTimer | None = None,
) -> tuple[HostProgram | None, list[HoistedTensorMap]]:
    timer = timer or CompilationTimer()

    with timer.phase("ir.simt_semantic_analysis"):
        simt_semantic_analysis(func_ir, ctx)

    with timer.phase("ir.host_program_analysis"):
        host_program_by_var = get_host_programs_by_var(func_ir)
    with timer.phase("ir.dynamic_shared_memory"):
        dyn_smem_size_program = handle_dynamic_shared_memory(
            func_ir, host_program_by_var
        )
    with timer.phase("ir.tensor_map_hoisting"):
        hoisted_tensor_maps = hoist_tensor_maps(func_ir, host_program_by_var)

    with timer.phase("ir.eliminate_assign_ops"):
        eliminate_assign_ops(func_ir)
    with timer.phase("ir.dead_code_elimination"):
        dead_code_elimination_pass(func_ir)

    return dyn_smem_size_program, hoisted_tensor_maps


def compile_simt(
    function: AnnotatedFunction | FunctionType,
    signatures: Sequence[KernelSignature],
    gpu_name: str | None = None,
    arch: str | None = None,
    compute_capability: ComputeCapability | None = None,
    compiler_options: CompilerOptions = CompilerOptions(),
    ctx: ir.IRContext | None = None,
    keep_hir: bool = False,
    keep_final_ir: bool = False,
    keep_mlir: bool = False,
    keep_nvvm: bool = False,
    keep_ptx: bool = False,
    keep_timings: bool = False,
    log_hir: bool = False,
    log_ir: bool = False,
    log_mlir: bool = False,
    log_nvvm: bool = False,
    log_ptx: bool = False,
    log_timings: bool = False,
) -> CompilationResult:
    log_flags = deepcopy(get_log_flags())
    log_flags.log_hir |= log_hir
    log_flags.log_ir |= log_ir
    log_flags.log_mlir |= log_mlir
    log_flags.log_nvvm |= log_nvvm
    log_flags.log_ptx |= log_ptx
    log_flags.log_timings |= log_timings
    need_timings = keep_timings or log_flags.log_timings
    timer = CompilationTimer(enabled=need_timings)

    match function:
        case FunctionType():
            function = get_annotated_function(function)
        case kernel():
            function = get_annotated_function(function._pyfunc)

    def _dump(phase: str, contents: object) -> None:
        logging_template = (
            "=" * 20 + " cuda.lang {header} dump: " + "=" * 20 + "\n" + "{body}" + "\n"
        )
        print(
            logging_template.format(header=str(phase), body=str(contents)),
            file=sys.stderr,
        )

    with timer.phase("ir.ast2hir"):
        func_hir = get_function_hir(function.pyfunc, mode=HirMode.ENTRY_POINT)

    if log_flags.log_hir:
        _dump("HIR", func_hir.body)

    [signature] = signatures
    if signature.symbol is None:
        signature = signature.with_mangled_symbol(function.pyfunc.__name__)

    ctx = ctx or ir.IRContext(log_ir_on_error=log_flags.log_hir or log_flags.log_ir)

    with timer.phase("ir.hir2ir"):
        func_ir = get_function_ir(
            func_hir, signature, ctx, function.parameter_annotations
        )

    if log_flags.log_ir:
        _dump("IR (pre-transforms)", func_ir)

    dyn_smem_size_program, hoisted_tensor_maps = _transform_ir(func_ir, ctx, timer)

    if log_flags.log_ir:
        _dump("IR (post-transforms)", func_ir)

    with timer.phase("ir.flatten_cfg"):
        flattened_ir = flatten_cfg(func_ir, ctx)

    if log_flags.log_flattened_ir:
        _dump("Flattened IR", flattened_ir)

    with timer.phase("ir.target_resolution"):
        if gpu_name is None or arch is None:
            cc = compute_capability or get_compute_capability()
            suffix = "a" if cc >= (9, 0) else ""
            gpu_name = gpu_name or cc.gpu_name + suffix
            arch = arch or cc.arch + suffix

        target_info = TargetInfo.from_arch(arch)

    with timer.phase("ir2mlir"):
        mlir_module = ir2mlir(
            signature,
            flattened_ir,
            ctx,
            compiler_options,
            target_info,
        )
    with timer.phase("mlir_serialization"):
        mlir_text = str(mlir_module)

    if log_flags.log_mlir:
        _dump("MLIR", mlir_text)

    need_nvvm = log_flags.log_nvvm or keep_nvvm
    need_ptx = log_flags.log_ptx or keep_ptx
    ptx_compiler_options = compiler_options._ptx_compiler_options
    with timer.phase("mlir2cubin"):
        compiled = mlir2cubin(
            mlir_text,
            gpu_name=gpu_name,
            arch=arch,
            emit_nvvm=need_nvvm,
            emit_ptx=need_ptx,
            opt_level=compiler_options.opt_level,
            generate_line_info=compiler_options.debug_info == "line",
            ptx_compiler_options=ptx_compiler_options,
            emit_timings=need_timings,
        )
    if compiled.timings_ns is not None:
        timer.add_phases("mlir2cubin", compiled.timings_ns)

    if compiled.stderr and ptx_compiler_options:
        _dump("PTX compiler", compiled.stderr.decode())

    if need_nvvm:
        assert compiled.nvvm is not None

    if need_ptx:
        assert compiled.ptx is not None

    if log_flags.log_nvvm:
        _dump("NVVM", compiled.nvvm)

    if log_flags.log_ptx:
        _dump("PTX", compiled.ptx)

    timings = timer.finish() if need_timings else None
    if log_flags.log_timings:
        assert timings is not None
        _dump("Timings", timings.format_summary())

    return CompilationResult(
        kernel_signatures=[signature],
        dyn_smem_size_program=dyn_smem_size_program,
        hoisted_tensor_maps=hoisted_tensor_maps,
        timings=timings if keep_timings else None,
        hir=func_hir if keep_hir else None,
        final_ir=flattened_ir if keep_final_ir else None,
        mlir=mlir_text if keep_mlir else None,
        nvvm=compiled.nvvm if keep_nvvm else None,
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
