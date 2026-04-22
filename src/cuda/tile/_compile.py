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
import traceback
from types import FunctionType
from typing import Optional, Sequence
import zipfile

from cuda.tile import int32
from cuda.tile._annotated_function import AnnotatedFunction, get_annotated_function
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._cext import get_compute_capability, TileContext, default_tile_context
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import (
    TileCompilerError,
    TileCompilerExecutionError,
    TileCompilerTimeoutError, FunctionDesc, Loc
)
from cuda.tile._ir import ir, hir
from cuda.tile._ir.ops import loosely_typed_const, flatten_block_parameters, tile_impl_registry
from cuda.tile._ir.type import TileTy, ArrayTy, ListTy, make_tile_ty
from cuda.tile._passes.ast2hir import get_function_hir
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
)

from cuda.tile._passes.dataflow_analysis import dataflow_analysis
from cuda.tile._passes.check_dtype_support import check_dtype_support
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.propagate_divby import add_divby_pass
from cuda.tile._passes.token_order import token_order_pass
from cuda.tile._cache import cache_key, cache_lookup, cache_store, evict_lru
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
from cuda.tile._version import __version__ as cutile_version
import cuda.tile._bytecode as bc
from cuda.tile.compilation._signature import KernelSignature, ParameterConstraint, \
    ScalarConstraint, ArrayConstraint, ListConstraint, ConstantConstraint

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
                  param_constraints: Sequence[tuple[tuple[ir.Var, ...], ParameterConstraint]]):
    eliminate_assign_ops(func_body)
    dead_code_elimination_pass(func_body)
    dataflow_result = dataflow_analysis(func_body, param_constraints)

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


@dataclass
class _KernelParameters:
    aggregate_vars: Sequence[ir.Var]
    nonconstant_flat_vars: Sequence[tuple[tuple[ir.Var, ...], ParameterConstraint]]


def _create_kernel_parameters(parameter_constraints: Sequence[ParameterConstraint],
                              constant_parameter_mask: Sequence[bool],
                              parameter_names: Sequence[str],
                              parameter_locations: Sequence[Loc],
                              ir_ctx: ir.IRContext) -> _KernelParameters:
    aggregate_vars = []
    nonconstant_flat_vars = []
    for pos, (constraint, is_const, name, loc) in enumerate(
            zip(parameter_constraints, constant_parameter_mask,
                parameter_names, parameter_locations, strict=True)):
        if is_const:
            if not isinstance(constraint, ConstantConstraint):
                raise TypeError(f"Expected a ConstantConstraint for the constant parameter"
                                f" {name} at position {pos}, got '{type(constraint).__name__}'")
            var = loosely_typed_const(constraint.value, name=name)
        else:
            if isinstance(constraint, ScalarConstraint):
                ty = TileTy(constraint.dtype, ())
            elif isinstance(constraint, ArrayConstraint):
                ty = _get_array_ty(constraint)
            elif isinstance(constraint, ListConstraint):
                assert isinstance(constraint.element, ArrayConstraint)
                array_ty = _get_array_ty(constraint.element)
                ty = ListTy(array_ty)
            else:
                raise TypeError(f"Unexpected parameter descriptor type"
                                f" '{type(constraint).__name__}'"
                                f" for non-constant parameter `{name}` at position {pos}")
            var = ir_ctx.make_var(name, loc)
            var.set_type(ty)
            [flat_vars] = flatten_block_parameters([var])
            nonconstant_flat_vars.append((flat_vars, constraint))
        aggregate_vars.append(var)
    return _KernelParameters(aggregate_vars, nonconstant_flat_vars)


def _get_array_ty(param: ArrayConstraint):
    if param.index_dtype != int32:
        raise NotImplementedError("Only int32 is currently supported as array index type")

    for static_stride, bound in zip(param.stride_constant, param.stride_lower_bound_incl,
                                    strict=True):
        if static_stride is not None:
            continue
        if bound is None or bound < 0:
            raise NotImplementedError("Negative strides are currently not supported:"
                                      " please specify stride_lower_bound_incl=0")

    return ArrayTy(make_tile_ty(param.dtype, ()),
                   shape=(None,) * param.ndim,
                   strides=param.stride_constant)


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

    for i, func_ir in enumerate(final_ir):
        artifacts[f"{func_name}.{i}.cutileir"] = f"{func_ir.body.to_string(include_loc=False)}\n"

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


class _IrKeeper:
    def __init__(self,
                 ann_func: AnnotatedFunction,
                 func_hir: hir.Function,
                 signatures: Sequence[KernelSignature],
                 bytecode_version: bc.BytecodeVersion,
                 sm_arch: str,
                 log_cutile_ir: bool,
                 keep_all: bool):
        self.ann_func = ann_func
        self._func_hir = func_hir
        self.signatures = signatures
        self.bytecode_version = bytecode_version
        self.sm_arch = sm_arch
        self._log_cutile_ir = log_cutile_ir
        self.final_ir: list[ir.Block | None] | None = [None] * len(signatures) if keep_all else None

    @property
    def num_signatures(self):
        return len(self.signatures)

    def get_final_ir(self, signature_index: int) -> ir.Block:
        if self.final_ir is None or self.final_ir[signature_index] is None:
            sig = self.signatures[signature_index]
            param_names = tuple(self.ann_func.pysig.parameters.keys())
            ir_ctx = ir.IRContext(log_ir_on_error=self._log_cutile_ir,
                                  tileiras_version=self.bytecode_version)
            with ir.Builder(ir_ctx, self._func_hir.body.loc) as ir_builder:
                params = _create_kernel_parameters(sig.parameters,
                                                   self.ann_func.constant_parameter_mask,
                                                   param_names,
                                                   self._func_hir.param_locs,
                                                   ir_ctx)
                with tile_impl_registry.as_current():
                    hir2ir(self._func_hir, params.aggregate_vars, ir_ctx)

            func_body = ir.Block(ir_ctx, self._func_hir.body.loc)
            func_body.params = sum((vars for vars, _ in params.nonconstant_flat_vars), ())
            func_body.extend(ir_builder.ops)

            _transform_ir(func_body, self.bytecode_version, params.nonconstant_flat_vars)

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


def _get_bytecode(ir_keeper: _IrKeeper,
                  compiler_options: CompilerOptions,
                  anonymize_debug_info: bool) -> bytearray:
    bytecode_buf = bytearray()

    with bc.write_bytecode(num_functions=ir_keeper.num_signatures,
                           buf=bytecode_buf, version=ir_keeper.bytecode_version) as writer:
        for i in range(ir_keeper.num_signatures):
            func_body = ir_keeper.get_final_ir(i)
            symbol = ir_keeper.signatures[i].symbol
            generate_bytecode_for_kernel(func_body, symbol, compiler_options, ir_keeper.sm_arch,
                                         writer, anonymize_debug_attr=anonymize_debug_info)
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

    if sm_arch is None:
        sm_arch = get_sm_arch()

    if bytecode_version is None:
        bytecode_version = _get_max_supported_bytecode_version(context.config.temp_dir,
                                                               allow_dev=dev_features_enabled())

    func_hir = get_function_hir(ann_func.pyfunc, entry_point=True)
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
        opt_level = compiler_options.specialize_for_target(sm_arch).opt_level
        key = cache_key(compiler_ver, sm_arch, opt_level, bytecode_buf)
        cubin = cache_lookup(cache_dir, key)
        if cubin is not None:
            ret.cubin = cubin
            return ret

    # Compile MLIR module and generate cubin
    with tempfile.NamedTemporaryFile(suffix='.bytecode', prefix=func_desc.name,
                                     dir=context.config.temp_dir, delete=False) as f:
        f.write(bytecode_buf)
        f.flush()

        try:
            cubin_file = compile_cubin(f.name, compiler_options, sm_arch,
                                       timeout_sec=context.config.compiler_timeout_sec)
        except TileCompilerError as e:
            if context.config.enable_crash_dump:
                anonymized_bytecode = _get_bytecode(ir_keeper, compiler_options,
                                                    anonymize_debug_info=True)

                _compiler_crash_dump(ir_keeper.final_ir, func_desc.name,
                                     anonymized_bytecode, e.message,
                                     e.compiler_flags, e.compiler_version)

            raise e
        ret.cubin = Path(cubin_file).read_bytes()

    if cache_dir is not None and key is not None:
        cache_store(cache_dir, key, ret.cubin)
        evict_lru(cache_dir, context.config.cache_size_limit)

    return ret


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


@cache
def get_sm_arch() -> str:
    major, minor = get_compute_capability()
    return f'sm_{major}{minor}'


def compile_cubin(
        fname_bytecode: str,
        compiler_options: CompilerOptions,
        sm_arch: str,
        timeout_sec: Optional[float]) -> Path:
    binary = _find_compiler_bin()
    fname_cubin = Path(fname_bytecode).with_suffix(".cubin")
    compiler_hints = compiler_options.specialize_for_target(sm_arch)

    args = [str(fname_bytecode), "-o", str(fname_cubin)]

    flags = [
        "--gpu-name",
        sm_arch,
        f"-O{compiler_hints.opt_level}",
        "--lineinfo"
    ]

    binary.run(args, flags, timeout_sec)
    return fname_cubin
