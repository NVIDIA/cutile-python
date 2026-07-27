# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import importlib.metadata
import math
import re
import warnings
import contextlib
from dataclasses import dataclass
import datetime
import functools
from functools import cache
import logging
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import traceback
from types import FunctionType
from typing import Optional, Sequence
import zipfile

from cuda.tile import ArrayAnnotation
from cuda.tile._annotated_function import (
    AnnotatedFunction, HeterogeneousTupleNode, HomogeneousTupleNode, LeafAnnotationNode,
    ParameterAnnotationNode, get_annotated_function)
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._cext import get_compute_capability, TileContext, default_tile_context
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._datatype import DType
from cuda.tile._exception import (
    TileCompilerError,
    TileCompilerExecutionError,
    TileCompilerTimeoutError, FunctionDesc, Loc
)
from cuda.tile._ir import ir, hir
from cuda.tile._ir.core_ops import build_dataclass_instance
from cuda.tile._ir.ir import TypingHooks
from cuda.tile._ir.aggregate_support import flatten_block_parameters
from cuda.tile._ir.ops import loosely_typed_const, tile_impl_registry, build_tuple
from cuda.tile._ir.type import TileTy, ArrayTy, ListTy
from cuda.tile._ir.typing_support import get_dataclass_info
from cuda.tile._passes.ast2hir import get_function_hir, HirMode
from cuda.tile._passes.for_loop_break import lower_for_with_break
from cuda.tile._passes.code_motion import hoist_loop_invariants
from cuda.tile._passes.unhoist_partition_views import unhoist_partition_views
from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
from cuda.tile._passes.hir2ir import hir2ir
from cuda.tile._passes.loop_split import split_loops
from cuda.tile._passes.rewrite_patterns import rewrite_patterns
from cuda.tile._cext import dev_features_enabled
from cuda.tile._debug import (
    CUDA_TILE_TESTING_DISABLE_DIV,
    CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER,
    CUDA_TILE_DUMP_BYTECODE,
    CUDA_TILE_DUMP_TILEIR,
    EXPERIMENTAL_CUDA_TILE_DEBUG_BUILD,
)

from cuda.tile._passes.dataflow_analysis import DataflowResult, dataflow_analysis
from cuda.tile._passes.check_dtype_support import check_dtype_support
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.materialize_constants import materialize_constants_pass
from cuda.tile._passes.propagate_divby import add_divby_pass
from cuda.tile._passes.token_order import token_order_pass
from cutile_cache._cache import MetadataV1, cache_key, cache_lookup, cache_store, evict_lru
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
from cuda.tile._version import __version__ as cutile_version
import cuda.tile._bytecode as bc
from cuda.tile.compilation._signature import KernelSignature, ParameterConstraint, \
    ScalarConstraint, ArrayConstraint, ListConstraint, TupleConstraint, ConstantConstraint, \
    DataclassConstraint

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    kernel_signatures: Sequence[KernelSignature]
    cubin: bytes | None = None
    bytecode: bytearray | None = None
    final_ir: Sequence[ir.Block] | None = None


# Create a global lock
_compiler_lock = threading.RLock()


def global_compiler_lock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _compiler_lock:
            return func(*args, **kwargs)
    return wrapper


def _transform_ir(func_body: ir.Block,
                  bytecode_version: bc.BytecodeVersion,
                  param_constraints: Sequence[tuple[tuple[ir.Var, ...], ParameterConstraint]]
                  ) -> DataflowResult:
    eliminate_assign_ops(func_body)
    lower_for_with_break(func_body)
    dead_code_elimination_pass(func_body)
    dataflow_result = dataflow_analysis(func_body, param_constraints)

    materialize_constants_pass(func_body, dataflow_result)

    if not CUDA_TILE_TESTING_DISABLE_DIV:
        add_divby_pass(func_body, dataflow_result)

    if not CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER:
        token_order_pass(func_body, dataflow_result)

    rewrite_patterns(func_body)

    # Loop invariant code motion needs to run after the token order pass.
    # Otherwise, it may incorrectly hoist load operations out of the loop.
    hoist_loop_invariants(func_body)

    # For version < V_13_3, MakePartitionView must be emitted inline before its consumer.
    # Code motion may hoist it to an outer block; copy it back where needed.
    if bytecode_version < BytecodeVersion.V_13_3:
        unhoist_partition_views(func_body)

    split_loops(func_body)
    dead_code_elimination_pass(func_body)

    return dataflow_result


@dataclass
class _KernelParameters:
    aggregate_vars: Sequence[ir.Var]
    nonconstant_flat_vars: Sequence[tuple[tuple[ir.Var, ...], ParameterConstraint]]


def _create_kernel_parameters(parameter_constraints: Sequence[ParameterConstraint],
                              parameter_annotations: Sequence[ParameterAnnotationNode],
                              parameter_names: Sequence[str],
                              parameter_locations: Sequence[Loc],
                              ir_ctx: ir.IRContext) -> _KernelParameters:
    nonconstant_flat_vars = []
    parameter_vars = tuple(ir_ctx.make_var(name, loc)
                           for name, loc in zip(parameter_names, parameter_locations, strict=True))

    for pos, (constraint, annotation, name, var) in enumerate(
            zip(parameter_constraints, parameter_annotations,
                parameter_names, parameter_vars, strict=True)):
        path = ParameterPath(name, ())
        _create_parameter(constraint, annotation, path, var, nonconstant_flat_vars)
    return _KernelParameters(parameter_vars, nonconstant_flat_vars)


@dataclass
class ParameterPath:
    name: str
    tuple_indices_or_field_names: tuple[int | str, ...]

    def with_item(self, index_or_field_name: int | str) -> "ParameterPath":
        return ParameterPath(self.name, self.tuple_indices_or_field_names + (index_or_field_name,))


def _create_parameter(
        constraint: ParameterConstraint,
        annotation: ParameterAnnotationNode,
        path: ParameterPath,
        var: ir.Var,
        nonconstant_flat_vars: list[tuple[tuple[ir.Var, ...], ParameterConstraint]]
):
    if isinstance(annotation, LeafAnnotationNode):
        annotation.validate()

        if annotation.array is not None and not isinstance(constraint, ArrayConstraint):
            raise _make_constraint_error("ArrayAnnotation/IndexedWithInt64 can only be applied"
                                         " to an array or a list-of-array parameter.", path)

        if annotation.list is not None and not isinstance(constraint, ListConstraint):
            raise _make_constraint_error("ArrayAnnotation/IndexedWithInt64 can only be applied"
                                         " to an array or a list-of-array parameter.", path)

        if annotation.scalar is not None and not isinstance(constraint, ScalarConstraint):
            raise _make_constraint_error(
                "ScalarAnnotation/ScalarInt64 can only be applied"
                " to a non-constant scalar parameter.", path)

    if isinstance(constraint, TupleConstraint):
        if isinstance(annotation, LeafAnnotationNode):
            item_nodes = [annotation] * len(constraint.items)
        elif isinstance(annotation, HomogeneousTupleNode):
            item_nodes = [annotation.each] * len(constraint.items)
        elif isinstance(annotation, HeterogeneousTupleNode):
            if len(annotation.items) != len(constraint.items):
                raise _make_constraint_error(
                        f"Received a tuple of length {len(constraint.items)}"
                        f" but the annotation implies length {len(annotation.items)}.",
                        path)
            item_nodes = annotation.items
        else:
            assert False

        item_vars = []
        for i, (item, node) in enumerate(zip(constraint.items, item_nodes, strict=True)):
            item_var = var.ctx.make_var(var.name + f"_{i}", var.loc)
            _create_parameter(item, node, path.with_item(i), item_var, nonconstant_flat_vars)
            item_vars.append(item_var)
        build_tuple(item_vars, result_var=var)
        return

    if not isinstance(annotation, LeafAnnotationNode):
        raise _make_constraint_error("Non-tuple parameter is annotated as a tuple.", path)

    if isinstance(constraint, DataclassConstraint):
        info = get_dataclass_info(constraint.cls)
        field_vars = []
        for i, (field_constraint, field_name) in enumerate(zip(
                constraint.fields, info.field_names, strict=True)):
            field_var = var.ctx.make_var(var.name + f"_{i}", var.loc)
            _create_parameter(field_constraint, annotation, path.with_item(field_name), field_var,
                              nonconstant_flat_vars)
            field_vars.append(field_var)
        build_dataclass_instance(field_vars, info, result_var=var)
        return

    if annotation.constant and not isinstance(constraint, ConstantConstraint):
        raise _make_constraint_error("Expected a scalar/tuple/dataclass constant,"
                                     " as implied by the Constant annotation.", path)

    if isinstance(constraint, ConstantConstraint):
        if not annotation.constant:
            raise _make_constraint_error("ConstantConstraint is only valid for parameters"
                                         " annotated as Constant.", path)

        loosely_typed_const(constraint.value, result_var=var)
        return

    if isinstance(constraint, ScalarConstraint):
        if annotation.scalar is not None and annotation.scalar.dtype != constraint.dtype:
            raise _make_constraint_error(f"ScalarConstraint.dtype {constraint.dtype} does not match"
                                         f" the annotated dtype {annotation.scalar.dtype}.", path)
        ty = var.ctx.typing_hooks.get_tensor_like_type(constraint.dtype, ())
    elif isinstance(constraint, ArrayConstraint):
        ty = _get_array_ty(constraint, annotation.array, path, var.ctx.typing_hooks)
    elif isinstance(constraint, ListConstraint):
        assert isinstance(constraint.element, ArrayConstraint)
        array_ann = None if annotation.list is None else annotation.list.element
        array_ty = _get_array_ty(constraint.element, array_ann, path, var.ctx.typing_hooks)
        ty = ListTy(array_ty)
    else:
        raise _make_constraint_error(f"Unsupported constraint type"
                                     f" '{type(constraint).__name__}'.", path)

    var.set_type(ty)
    [flat_vars] = flatten_block_parameters([var])
    nonconstant_flat_vars.append((flat_vars, constraint))


def _make_constraint_error(message: str, path: ParameterPath):
    what = f"kernel parameter '{path.name}'"
    for tuple_index_or_field_name in path.tuple_indices_or_field_names:
        if isinstance(tuple_index_or_field_name, int):
            what = f"item #{tuple_index_or_field_name} of {what}"
        else:
            assert isinstance(tuple_index_or_field_name, str)
            what = f"field '{tuple_index_or_field_name}' of {what}"
    return TypeError(f"Invalid {what}: {message}")


def _resolve_static_axes(axes: tuple[int, ...],
                         ndim: int,
                         field_name: str,
                         path: ParameterPath) -> list[bool]:
    mask: list[bool] = [False] * ndim
    for axis in axes:
        if not -ndim <= axis < ndim:
            raise _make_constraint_error(f"Axis {axis} found in `{field_name}`"
                                         f" is out of range for an array of rank {ndim}.", path)
        normalized = axis + ndim if axis < 0 else axis
        if mask[normalized]:
            raise _make_constraint_error(
                    f"Axis {axis} appears more than once in `{field_name}`.", path)
        mask[normalized] = True
    return mask


def _select_annotated(mask: list[bool],
                      constants: tuple[int | None, ...],
                      field_name: str,
                      what: str,
                      path: ParameterPath) -> tuple[int | None, ...]:
    """Keep the constant value on annotated axes (validating it exists), None elsewhere."""
    result: list[int | None] = []
    for axis, (annotated, value) in enumerate(zip(mask, constants, strict=True)):
        if annotated and value is None:
            raise _make_constraint_error(
                f"Axis {axis} is annotated in `{field_name}`, but no "
                f"constant {what} value is available there.", path)
        result.append(value if annotated else None)
    return tuple(result)


def _get_array_ty(param: ArrayConstraint,
                  array_ann: ArrayAnnotation | None,
                  path: ParameterPath,
                  typing_hooks: TypingHooks):
    if array_ann is None:
        array_ann = ArrayAnnotation()

    for static_stride, bound in zip(param.stride_constant, param.stride_lower_bound_incl,
                                    strict=True):
        if static_stride is not None:
            continue
        if bound is None or bound < 0:
            raise _make_constraint_error("Negative strides are currently not supported:"
                                         " please specify stride_lower_bound_incl=0", path)

    static_shape_mask = _resolve_static_axes(
            array_ann.static_shape_dims, param.ndim, "static_shape_dims", path)
    array_ty_shape = _select_annotated(static_shape_mask, param.shape_constant,
                                       "static_shape_dims", "shape", path)

    static_stride_mask = _resolve_static_axes(
            array_ann.static_stride_dims, param.ndim, "static_stride_dims", path)
    array_ty_strides = _select_annotated(static_stride_mask, param.stride_constant,
                                         "static_stride_dims", "stride", path)

    return ArrayTy(param.dtype,
                   shape=array_ty_shape,
                   strides=array_ty_strides,
                   index_dtype=param.index_dtype,
                   typing_hooks=typing_hooks)


def _log_mlir(bytecode_buf):
    try:
        from cuda.tile_internal import _internal_cext
    except ImportError:
        print("Can't print MLIR because the internal extension is missing. "
              "This is currently not a public feature", file=sys.stderr)
        return

    try:
        text = _internal_cext.bytecode_to_mlir_text(bytecode_buf)
    except Exception:
        print("Failed to print MLIR", file=sys.stderr)
        traceback.print_exc()
        return

    print(f"Lowering\n==== TILEIR MLIR module ====\n\n{text}", file=sys.stderr)


def _compiler_crash_dump(final_ir: Sequence[ir.Block],
                         func_name: str,
                         anonymized_bytecode: bytearray,
                         error_msg,
                         compiler_flags,
                         compiler_version):
    debug_info = (
        f"error:\n{error_msg}\n\n"
        f"compiler flags:\n{compiler_flags}\n\n"
        f"compiler version:\n{compiler_version or 'Unkown'}\n\n"
        f"cutile version:\n{cutile_version}\n"
    )

    artifacts = {
        f"{func_name}.bytecode": bytes(anonymized_bytecode),
        "debug_info.txt": debug_info,
    }

    for i, block in enumerate(final_ir):
        artifacts[f"{func_name}.{i}.cutileir"] = f"{block.to_string(include_loc=False)}\n"

    timestamp = datetime.datetime.now().timestamp()
    zip_filename = os.path.abspath(f"crash_dump_{func_name}_{timestamp}.zip")
    print(f"Dumping crash artifacts to {zip_filename}\n", file=sys.stderr)

    with zipfile.ZipFile(zip_filename, "w") as z:
        for filename, content in artifacts.items():
            z.writestr(filename, content)


@contextlib.contextmanager
def unique_path_from_func_desc(base_dir: str, desc: FunctionDesc, suffix: str, mode: str = "wb"):
    prefix = []
    if desc.name is not None:
        prefix.append(desc.name)
    else:
        prefix.append("lambda")
    prefix.append(Path(desc.filename).stem)
    prefix.append(f"ln{desc.line}")
    prefix = ".".join(prefix) + "."
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=base_dir,
                                     delete=False, mode=mode) as f:
        yield f


class _TileTypingHooks(TypingHooks):
    def get_tensor_like_type(self, dtype: DType, shape: Sequence[int]) -> TileTy:
        return TileTy(dtype, shape)


class _IrKeeper:
    def __init__(self,
                 ann_func: AnnotatedFunction,
                 func_hir: hir.Function,
                 signatures: Sequence[KernelSignature],
                 bytecode_version: bc.BytecodeVersion,
                 sm_arch: str | None,
                 log_cutile_ir: bool,
                 keep_all: bool):
        self.ann_func = ann_func
        self._func_hir = func_hir
        self.signatures = signatures
        self.bytecode_version = bytecode_version
        self.sm_arch = sm_arch
        self._log_cutile_ir = log_cutile_ir
        self.final_ir: list[ir.Block | None] | None = [None] * len(signatures) if keep_all else None
        self._dataflow_results: list[DataflowResult | None] = [None] * len(signatures)

    @property
    def num_signatures(self):
        return len(self.signatures)

    def get_final_ir(self, signature_index: int) -> ir.Block:
        if self.final_ir is None or self.final_ir[signature_index] is None:
            sig = self.signatures[signature_index]
            param_names = tuple(self.ann_func.pysig.parameters.keys())
            ir_ctx = ir.IRContext(log_ir_on_error=self._log_cutile_ir,
                                  tileiras_version=self.bytecode_version,
                                  typing_hooks=_TileTypingHooks())
            with ir.Builder(ir_ctx, self._func_hir.body.loc) as ir_builder:
                with tile_impl_registry.as_current():
                    params = _create_kernel_parameters(sig.parameters,
                                                       self.ann_func.parameter_annotations,
                                                       param_names,
                                                       self._func_hir.param_locs,
                                                       ir_ctx)
                    hir2ir(self._func_hir, params.aggregate_vars, ir_ctx)

            func_body = ir.Block(ir_ctx, self._func_hir.body.loc)
            func_body.params = sum((vars for vars, _ in params.nonconstant_flat_vars), ())
            func_body.extend(ir_builder.ops)

            dataflow_result = _transform_ir(
                func_body, self.bytecode_version, params.nonconstant_flat_vars
            )
            self._dataflow_results[signature_index] = dataflow_result

            if self._log_cutile_ir:
                code = (f"==== CuTile IR for {self._func_hir.desc.name}==== \n\n"
                        f"{func_body.to_string(include_loc=False)}\n\n")
                print(f'\n{code}', file=sys.stderr)
            check_dtype_support(func_body, self.sm_arch, self.bytecode_version)
            if self.final_ir is not None:
                self.final_ir[signature_index] = func_body
            return func_body
        else:
            return self.final_ir[signature_index]

    def get_dataflow_result(self, signature_index: int) -> DataflowResult:
        dataflow_result = self._dataflow_results[signature_index]
        assert dataflow_result is not None
        return dataflow_result


def _get_bytecode(ir_keeper: _IrKeeper,
                  compiler_options: CompilerOptions,
                  anonymize_debug_info: bool) -> bytearray:
    bytecode_buf = bytearray()

    with bc.write_bytecode(num_functions=ir_keeper.num_signatures,
                           buf=bytecode_buf, version=ir_keeper.bytecode_version) as writer:
        for i in range(ir_keeper.num_signatures):
            func_body = ir_keeper.get_final_ir(i)
            dataflow_result = ir_keeper.get_dataflow_result(i)
            symbol = ir_keeper.signatures[i].symbol
            generate_bytecode_for_kernel(
                func_body, dataflow_result, symbol, compiler_options, ir_keeper.sm_arch, writer,
                anonymize_debug_attr=anonymize_debug_info
            )
    return bytecode_buf


def parse_bytecode_version(version_str: str) -> bc.BytecodeVersion:
    for v in _all_bytecode_versions(dev_features_enabled()):
        if v.as_string() == version_str:
            return v
    supported_versions_str = ", ".join(v.as_string() for v in _SUPPORTED_VERSIONS)
    raise ValueError(f"Unsupported bytecode version '{version_str}'."
                     f" Supported versions are: {supported_versions_str}")


@global_compiler_lock
def compile_tile(ann_func: AnnotatedFunction | FunctionType,
                 signatures: Sequence[KernelSignature],
                 sm_arch: str | None = None,
                 compiler_options: CompilerOptions = CompilerOptions(),
                 context: TileContext = default_tile_context,
                 bytecode_version: bc.BytecodeVersion | None = None,
                 return_final_ir: bool = False,
                 return_bytecode: bool = False,
                 return_cubin: bool = True) -> CompilationResult:
    if isinstance(ann_func, FunctionType):
        ann_func = get_annotated_function(ann_func)
    elif not isinstance(ann_func, AnnotatedFunction):
        raise TypeError(f"Expected a Python function or an AnnotatedFunction"
                        f" for `ann_func`, got {type(ann_func)}")

    signatures = list(signatures)
    for i in range(len(signatures)):
        if signatures[i].symbol is None:
            signatures[i] = signatures[i].with_mangled_symbol(ann_func.pyfunc.__name__)

    if sm_arch is None and return_cubin:
        # Fall back to the current device's arch iff exporting a cubin.
        sm_arch = get_sm_arch()

    if bytecode_version is None:
        bytecode_version = _get_max_supported_bytecode_version(context.config.temp_dir,
                                                               allow_dev=dev_features_enabled())

    func_hir = get_function_hir(ann_func.pyfunc, mode=HirMode.ENTRY_POINT)
    func_desc = func_hir.desc
    ir_keeper = _IrKeeper(ann_func=ann_func,
                          func_hir=func_hir,
                          signatures=signatures,
                          bytecode_version=bytecode_version,
                          sm_arch=sm_arch,
                          log_cutile_ir=context.config.log_cutile_ir,
                          keep_all=context.config.enable_crash_dump or return_final_ir)

    need_bytecode = return_bytecode or return_cubin
    if not need_bytecode:
        for i in range(ir_keeper.num_signatures):
            ir_keeper.get_final_ir(i)
        return CompilationResult(signatures, final_ir=ir_keeper.final_ir)

    bytecode_buf = _get_bytecode(ir_keeper, compiler_options, anonymize_debug_info=False)

    if context.config.log_tileir:
        _log_mlir(bytecode_buf)

    if CUDA_TILE_DUMP_BYTECODE is not None:
        if not os.path.isdir(CUDA_TILE_DUMP_BYTECODE):
            os.makedirs(CUDA_TILE_DUMP_BYTECODE)
        with unique_path_from_func_desc(CUDA_TILE_DUMP_BYTECODE,
                                        func_desc, '.tileirbc') as f:
            print(f"Dumping TILEIR bytecode to file: {f.name}", file=sys.stderr)
            f.write(bytecode_buf)

    # Write MLIR module to file
    if CUDA_TILE_DUMP_TILEIR is not None:
        try:
            from cuda.tile_internal._internal_cext import bytecode_to_mlir_text
            mlir_text = bytecode_to_mlir_text(bytecode_buf)
            if not os.path.isdir(CUDA_TILE_DUMP_TILEIR):
                os.makedirs(CUDA_TILE_DUMP_TILEIR)
            with unique_path_from_func_desc(CUDA_TILE_DUMP_TILEIR,
                                            func_desc, '.tileir', mode="w") as f:
                print(f"Dumping TILEIR MLIR module to file: {f.name}", file=sys.stderr)
                f.write(mlir_text)
        except ImportError:
            print("Can't print MLIR because the internal extension is missing. "
                  "This is currently not a public feature.", file=sys.stderr)

    ret = CompilationResult(signatures,
                            bytecode=bytecode_buf if return_bytecode else None,
                            final_ir=ir_keeper.final_ir)
    if not return_cubin:
        return ret

    # Check disk cache before invoking tileiras
    cache_dir = context.config.cache_dir
    compiler_ver = _get_compiler_version_string()
    key = None
    if cache_dir is None:
        logger.debug("disk cache disabled: context.config.cache_dir is not set")
    elif compiler_ver is None:
        logger.warning("disk cache disabled: compiler version is unknown")
    else:
        effective_opt, device_debug = _tileiras_effective_opt_and_device_debug(
            compiler_options, sm_arch)
        key = cache_key(
            compiler_ver, sm_arch, effective_opt, bytecode_buf, device_debug
        )
        cubin = cache_lookup(cache_dir, key)
        if cubin is not None:
            ret.cubin = cubin
            return ret

    # Compile MLIR module and generate cubin
    with tempfile.NamedTemporaryFile(suffix='.bytecode', prefix=func_desc.name,
                                     dir=context.config.temp_dir, delete=False) as f:
        f.write(bytecode_buf)
        f.flush()

        capture_remarks = (cache_dir is not None
                           and key is not None
                           and _tileiras_supports_remarks(context.config.temp_dir))
        remarks_file = Path(f.name).with_suffix(".remarks.yaml") if capture_remarks else None
        compilation_start = time.perf_counter()
        try:
            cubin_file = compile_cubin(f.name, compiler_options, sm_arch,
                                       timeout_sec=context.config.compiler_timeout_sec,
                                       remarks_output_file=remarks_file)
        except TileCompilerError as e:
            if context.config.enable_crash_dump:
                anonymized_bytecode = _get_bytecode(ir_keeper, compiler_options,
                                                    anonymize_debug_info=True)

                _compiler_crash_dump(ir_keeper.final_ir, func_desc.name,
                                     anonymized_bytecode, e.message,
                                     e.compiler_flags, e.compiler_version)

            raise e
        compilation_time = time.perf_counter() - compilation_start
        ret.cubin = Path(cubin_file).read_bytes()

    if cache_dir is not None and key is not None:
        remarks = ""
        if capture_remarks:
            try:
                remarks = remarks_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.debug("failed to read compilation remarks from %s",
                             remarks_file, exc_info=True)
        metadata = MetadataV1(
            kernel_names=[signature.symbol or "" for signature in signatures],
            compiler_version=compiler_ver.strip() if compiler_ver else None,
            compilation_timestamp=time.time(),
            compilation_time_seconds=compilation_time,
            remarks=remarks,
        )
        cache_store(cache_dir, key, ret.cubin, metadata.to_dict())
        evict_lru(cache_dir, context.config.cache_size_limit)

    return ret


def _tileiras_effective_opt_and_device_debug(
    compiler_options: CompilerOptions, sm_arch: str
) -> tuple[int, bool]:
    """
    When EXPERIMENTAL_CUDA_TILE_DEBUG_BUILD is set, tileiras must use -O0 and
    --device-debug; the disk cache key must match (see cache_key).
    """
    if EXPERIMENTAL_CUDA_TILE_DEBUG_BUILD:
        return 0, True
    return compiler_options.opt_level_for_target(sm_arch), False


def is_windows() -> bool:
    return sys.platform == "win32"


def _get_cuda_home() -> Optional[str]:
    if is_windows():
        if (ret := os.environ.get("CUDA_PATH")):
            return ret
    return os.environ.get("CUDA_HOME")


@dataclass
class _CompilerBinary:
    path: str
    bin_path: str
    ld_path: str
    pass_cuda_home_var: bool

    def run(self,
            args: list[str],
            flags: list[str],
            timeout_sec: float | None = None):
        command = [self.path, *args]

        logger.debug(f"Invoke tile compiler: {' '.join(command + flags)}\n"
                     f"LD_LIBRARY_PATH:{self.ld_path}\n"
                     f"PATH:{self.bin_path}")
        try:
            env = os.environ.copy()
            env['LD_LIBRARY_PATH'] = self.ld_path
            env['PATH'] = self.bin_path
            if not self.pass_cuda_home_var:
                for key in {"CUDA_HOME", "CUDA_PATH"}:
                    env.pop(key, None)
            subprocess.run(command + flags, env=env, check=True, capture_output=True,
                           timeout=timeout_sec)
        except subprocess.CalledProcessError as e:
            raise TileCompilerExecutionError(e.returncode, e.stderr.decode(), ' '.join(flags),
                                             _try_get_compiler_version(self.path))
        except subprocess.TimeoutExpired:
            message = (f"`tileiras` compiler exceeded timeout {timeout_sec}s. "
                       "Using a smaller tile size may reduce compilation time.")
            raise TileCompilerTimeoutError(message, ' '.join(flags),
                                           _try_get_compiler_version(self.path))


_PIP_TILEIRAS_PACKAGES = (
    "nvidia-cuda-tileiras",
    "nvidia-cuda-nvcc",
    "nvidia-nvvm",
)


def _get_major_minor(version_str: str) -> tuple[int, int]:
    parts = version_str.split(".")
    return int(parts[0]), int(parts[1])


def _find_pip_tileiras() -> Optional[str]:
    versions: dict[str, str] = {}
    for pkg in _PIP_TILEIRAS_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    majors_minors = {pkg: _get_major_minor(v) for pkg, v in versions.items()}
    unique = set(majors_minors.values())
    if len(unique) != 1:
        details = ", ".join(f"{pkg} {versions[pkg]}" for pkg in _PIP_TILEIRAS_PACKAGES)
        warnings.warn(
            f"Installed NVIDIA pip packages have mismatched versions ({details}). "
            "Falling back to system tileiras.",
            stacklevel=3,
        )
        return None

    try:
        import nvidia.cu13 as cu13_pkg
        cu13_root = cu13_pkg.__path__[0]
    except (ImportError, AttributeError, IndexError):
        logger.debug("Fail to get nvidia.cu13 package path.", exc_info=True)
        return None

    pip_bin_dir = os.path.join(cu13_root, "bin")
    res = shutil.which("tileiras", path=pip_bin_dir)
    if res is None:
        logger.debug("Fail to find tileiras under nvidia.cu13 path.")
        return None

    logger.debug(f"Found tileiras from pip package: {res}")
    return res


@cache
def _find_compiler_bin() -> _CompilerBinary:
    bin_path = os.environ.get('PATH', '')
    ld_path = os.environ.get('LD_LIBRARY_PATH', "") if not is_windows() else ""

    # search from nvidia-cuda-tileiras pip package
    logger.debug("Searching tileiras from nvidia pip package")
    res = _find_pip_tileiras()
    if res is not None:
        return _CompilerBinary(res, bin_path, ld_path, pass_cuda_home_var=False)

    # search under PATH
    logger.debug(f"Searching tileiras: {bin_path}")
    if (res := shutil.which("tileiras")):
        return _CompilerBinary(res, bin_path, ld_path, pass_cuda_home_var=True)

    # search under CUDA_HOME
    if (cuda_home := _get_cuda_home()):
        cuda_bin_path = os.path.join(cuda_home, 'bin')
        logger.debug(f"Searching tileiras: {cuda_bin_path}")
        if (res := shutil.which("tileiras", path=cuda_bin_path)):
            bin_path = bin_path + ":" + cuda_bin_path
            return _CompilerBinary(res, bin_path, ld_path, pass_cuda_home_var=True)

    # Try default CUDA Toolkit installation paths as a fallback
    res = _find_compiler_in_default_cuda_toolkit_paths()
    if res is not None:
        tileiras_path, bin_path = res
        return _CompilerBinary(tileiras_path, bin_path, ld_path, pass_cuda_home_var=False)

    cuda_home_var = "CUDA_PATH" if is_windows() else "CUDA_HOME"
    raise FileNotFoundError("'tileiras' compiler not found, "
                            "make sure it is available as a python package via "
                            "`pip install cuda-tile[tileiras]` or "
                            f"available in $PATH or ${cuda_home_var}/bin via system CTK (13.1+)"
                            " installation.")


_SUPPORTED_VERSIONS = [
    BytecodeVersion.V_13_1,
    BytecodeVersion.V_13_2,
    BytecodeVersion.V_13_3,
    BytecodeVersion.V_13_4,
]


def _all_bytecode_versions(allow_dev: bool = False) -> Sequence[BytecodeVersion]:
    return BytecodeVersion if allow_dev else _SUPPORTED_VERSIONS


@cache
def _get_max_supported_bytecode_version(temp_dir: str, allow_dev: bool = False) -> BytecodeVersion:
    binary = _find_compiler_bin()
    flags = ["--gpu-name", "sm_120"]
    for version in reversed(_all_bytecode_versions(allow_dev)):
        probe = bytearray()
        with bc.write_bytecode(num_functions=0, buf=probe, version=version):
            pass

        with tempfile.NamedTemporaryFile(suffix='.bytecode', prefix=f"probe{version}",
                                         dir=temp_dir, delete=False) as f_in, \
            tempfile.NamedTemporaryFile(suffix='.cubin', prefix=f"probe{version}",
                                        dir=temp_dir, delete=False) as f_out:
            f_in.write(probe)

        try:
            binary.run([f_in.name, "-o", f_out.name], flags)
        except TileCompilerError:
            continue

        return version

    warnings.warn("Failed to detect the maximum supported TileIR bytecode version;"
                  " falling back to 13.1.")
    return BytecodeVersion.V_13_1


def _tileiras_supports_remarks(temp_dir: str) -> bool:
    max_supported_version = _get_max_supported_bytecode_version(
        temp_dir, allow_dev=dev_features_enabled()
    )
    return max_supported_version >= BytecodeVersion.V_13_4


def _find_compiler_in_default_cuda_toolkit_paths() -> tuple[str, str] | None:
    binary_name = "tileiras.exe" if is_windows() else "tileiras"
    for toolkit_path in _get_default_cuda_toolkit_paths():
        bin_path = os.path.join(toolkit_path, "bin")
        p = os.path.join(bin_path, binary_name)
        if os.path.exists(p) and os.access(p, os.X_OK) and not os.path.isdir(p):
            return p, bin_path
    return None


def _get_default_cuda_toolkit_paths() -> list[str]:
    candidates = []

    if os.name == "nt":
        prefix = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
        regex = re.compile(r"[vV]([0-9]+)(\.[0-9]+)?")
    else:
        prefix = "/usr/local"
        regex = re.compile(r"cuda-([0-9]+)(\.[0-9]+)?")
        candidates.append((math.inf, math.inf, "cuda"))

    for subdir in os.listdir(prefix):
        m = re.fullmatch(regex, subdir)
        if m is None:
            continue
        major = int(m.group(1))
        minor = m.group(2)
        minor = math.inf if minor is None else int(minor[1:])
        candidates.append((major, minor, subdir))

    return [os.path.join(prefix, subdir)
            for _, _, subdir in reversed(sorted(candidates))]


def _try_get_compiler_version(compiler_bin) -> Optional[str]:
    try:
        res = subprocess.run([str(compiler_bin), "--version"],
                             check=True, capture_output=True, text=True)
        return res.stdout
    except Exception:
        return None


@cache
def _get_compiler_version_string() -> str | None:
    binary = _find_compiler_bin()
    version = _try_get_compiler_version(binary.path)
    return version


def format_sm_arch(major: int, minor: int) -> str:
    return f'sm_{major}{minor}'


@cache
def get_sm_arch(device_id: int = 0) -> str:
    return format_sm_arch(*get_compute_capability(device_id))


def compile_cubin(
        fname_bytecode: str,
        compiler_options: CompilerOptions,
        sm_arch: str,
        timeout_sec: Optional[float],
        remarks_output_file: str | os.PathLike | None = None) -> Path:
    binary = _find_compiler_bin()
    fname_cubin = Path(fname_bytecode).with_suffix(".cubin")
    effective_opt, use_device_debug = _tileiras_effective_opt_and_device_debug(
        compiler_options, sm_arch
    )

    args = [str(fname_bytecode), "-o", str(fname_cubin)]

    flags: list[str] = [
        "--gpu-name",
        sm_arch,
        f"-O{effective_opt}",
    ]
    if use_device_debug:
        flags.append("--device-debug")
    else:
        flags.append("--lineinfo")
    if remarks_output_file is not None:
        flags.extend([
            "--remark-format=yaml",
            "--remarks=all",
            f"--remarks-output-file={remarks_output_file}",
        ])

    binary.run(args, flags, timeout_sec)
    return fname_cubin
