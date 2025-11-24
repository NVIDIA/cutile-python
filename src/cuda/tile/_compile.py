# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
from typing import Callable, Optional
import zipfile

from cuda.tile._ast2ir import get_function_ir
from cuda.tile._cext import get_compute_capability, TileContext, default_tile_context
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._exception import (
    TileCompilerError,
    TileCompilerExecutionError,
    TileCompilerTimeoutError,
)
from cuda.tile._ir import ir
from cuda.tile._passes.code_motion import hoist_loop_invariants
from cuda.tile._passes.loop_split import split_loops
from cuda.tile._passes.rewrite_patterns import rewrite_patterns
from cuda.tile._debug import (
    CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER,
    CUDA_TILE_DUMP_BYTECODE,
    CUDA_TILE_DUMP_TILEIR,
)

from cuda.tile._passes.alias_analysis import alias_analysis_pass
from cuda.tile._passes.typeinfer import infer_types_pass
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.token_order import token_order_pass
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
from cuda.tile._version import __version__ as cutile_version
import cuda.tile._bytecode as bc


logger = logging.getLogger(__name__)


class TileLibrary:
    def __init__(self, func_name, fname_cubin, bytecode, final_ir: ir.Function):
        self.func_name = func_name
        self.fname_cubin = fname_cubin
        self.bytecode = bytecode
        self.final_ir = final_ir


# Create a global lock
_compiler_lock = threading.RLock()


def global_compiler_lock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _compiler_lock:
            return func(*args, **kwargs)
    return wrapper


def _get_final_ir(pyfunc, args) -> ir.Function:
    ir_ctx = ir.IRContext()
    func_ir: ir.Function = get_function_ir(pyfunc, ir_ctx, call_site=None)
    ir_args = func_ir.bind_arguments(args, get_constant_annotations(pyfunc))
    infer_types_pass(func_ir, ir_args, pyfunc)
    dead_code_elimination_pass(func_ir)

    if not CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER:
        alias_result = alias_analysis_pass(func_ir)
        token_order_pass(func_ir, alias_result)

    rewrite_patterns(func_ir)

    # Loop invariant code motion needs to run after the token order pass.
    # Otherwise, it may incorrectly hoist load operations out of the loop.
    hoist_loop_invariants(func_ir)

    split_loops(func_ir.root_block)
    dead_code_elimination_pass(func_ir)
    return func_ir


def _log_mlir(bytecode_buf):
    try:
        from cuda.tile_internal import _internal_cext
    except ImportError:
        print("Can't print MLIR because the internal extension is missing", file=sys.stderr)
        return

    try:
        text = _internal_cext.bytecode_to_mlir_text(bytecode_buf)
    except Exception:
        print("Failed to print MLIR", file=sys.stderr)
        traceback.print_exc()
        return

    print(f"Lowering\n==== TILEIR MLIR module ====\n\n{text}", file=sys.stderr)


def _compiler_crash_dump(func_ir,
                         bytecode_generator,
                         error_msg,
                         compiler_flags,
                         compiler_version):
    debug_info = (
        f"error:\n{error_msg}\n\n"
        f"compiler flags:\n{compiler_flags}\n\n"
        f"compiler version:\n{compiler_version or 'Unkown'}\n\n"
        f"cutile version:\n{cutile_version}\n"
    )

    # Anonymize debug attributes in the bytecode
    bytecode_buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
        bytecode_generator(writer, anonymize_debug_attr=True)

    artifacts = {
        f"{func_ir.qualname}.bytecode": bytes(bytecode_buf),
        f"{func_ir.qualname}.cutileir": f"{func_ir.to_string(include_loc=False)}\n",
        "debug_info.txt": debug_info,
    }

    timestamp = datetime.datetime.now().timestamp()
    zip_filename = os.path.abspath(f"crash_dump_{func_ir.qualname}_{timestamp}.zip")
    print(f"Dumping crash artifacts to {zip_filename}\n", file=sys.stderr)

    with zipfile.ZipFile(zip_filename, "w") as z:
        for filename, content in artifacts.items():
            z.writestr(filename, content)


@global_compiler_lock
def compile_tile(pyfunc,
                 args,
                 compiler_options: CompilerOptions,
                 context: TileContext = default_tile_context) -> TileLibrary:
    func_ir = _get_final_ir(pyfunc, args)

    if 'CUTILEIR' in context.config.log_keys:
        code = (f"==== CuTile IR for {func_ir.qualname}==== \n\n"
                f"{func_ir.to_string(include_loc=False)}\n\n")
        print(f'\n{code}', file=sys.stderr)

    sm_arch = get_sm_arch()

    bytecode_generator = functools.partial(generate_bytecode_for_kernel,
                                           func_ir, compiler_options, sm_arch)

    bytecode_buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
        bytecode_generator(writer, anonymize_debug_attr=False)

    if 'TILEIR' in context.config.log_keys:
        _log_mlir(bytecode_buf)

    if CUDA_TILE_DUMP_BYTECODE is not None:
        if not os.path.exists(CUDA_TILE_DUMP_BYTECODE):
            os.makedirs(CUDA_TILE_DUMP_BYTECODE)
        base_filename = os.path.basename(func_ir.loc.filename.split(".")[0])
        path = os.path.join(CUDA_TILE_DUMP_BYTECODE,
                            f"{base_filename}.ln{func_ir.loc.line}.cutile")
        print(f"Dumping TILEIR bytecode to file: {path}", file=sys.stderr)
        with open(path, "wb") as f:
            f.write(bytecode_buf)

    # Write MLIR module to file
    if CUDA_TILE_DUMP_TILEIR is not None:
        try:
            from cuda.tile_internal._internal_cext import bytecode_to_mlir_text
            mlir_text = bytecode_to_mlir_text(bytecode_buf)
            if not os.path.exists(CUDA_TILE_DUMP_TILEIR):
                os.makedirs(CUDA_TILE_DUMP_TILEIR)
            base_filename = os.path.basename(func_ir.loc.filename.split(".")[0])
            path = os.path.join(
                CUDA_TILE_DUMP_TILEIR, f"{base_filename}.ln{func_ir.loc.line}.cuda_tile.mlir"
            )
            print(f"Dumping TILEIR MLIR module to file:{path}", file=sys.stderr)
            with open(path, "w") as f:
                print(mlir_text, file=f)
        except ImportError:
            print("Can't print MLIR because the internal extension is missing", file=sys.stderr)

    # Compile MLIR module and generate cubin
    with tempfile.NamedTemporaryFile(suffix='.bytecode', prefix=func_ir.qualname,
                                     dir=context.config.temp_dir, delete=False) as f:
        f.write(bytecode_buf)
        f.flush()

        try:
            cubin_file = compile_cubin(f.name, compiler_options, sm_arch,
                                       timeout_sec=context.config.compiler_timeout_sec)
        except TileCompilerError as e:
            if context.config.enable_crash_dump:
                _compiler_crash_dump(func_ir, bytecode_generator, e.message,
                                     e.compiler_flags, e.compiler_version)

            raise e

    return TileLibrary(func_ir.qualname, cubin_file, bytecode_buf, func_ir)


# Adapter between compile_tile() and kernel/TileDispatcher
@dataclass
class CompileCallback:
    pyfunc: Callable
    compiler_options: CompilerOptions

    def __call__(self, pyfunc_args, tile_context):
        lib = compile_tile(self.pyfunc, pyfunc_args, self.compiler_options, tile_context)
        return str(lib.fname_cubin), lib.func_name


def is_windows() -> bool:
    return sys.platform == "win32"


def _get_cuda_home() -> Optional[str]:
    if is_windows():
        if (ret := os.environ.get("CUDA_PATH")):
            return ret
    return os.environ.get("CUDA_HOME")


def _local_deps_dir():
    import cuda.tile
    package_dir = os.path.dirname(os.path.abspath(cuda.tile.__file__))
    return os.path.join(package_dir, '_deps')


@cache
def _find_compiler_bin() -> tuple[str, str, str]:
    # search under cuda/tile/_deps
    res = None
    bin_path = os.environ.get('PATH', '')
    ld_path = os.environ.get('LD_LIBRARY_PATH', "") if not is_windows() else ""

    deps_bin_dir = os.path.join(_local_deps_dir(), 'bin')
    deps_lib_dir = os.path.join(_local_deps_dir(), 'lib')
    if os.path.exists(deps_bin_dir):
        logger.debug(f"Searching tileiras: {deps_bin_dir}")
        if (res := shutil.which("tileiras", path=deps_bin_dir)):
            bin_path = deps_bin_dir + ":" + bin_path
            ld_path = deps_lib_dir + ":" + ld_path
            return res, bin_path, ld_path

    # search under PATH
    logger.debug(f"Searching tileiras: {bin_path}")
    if (res := shutil.which("tileiras")):
        return res, bin_path, ld_path

    # search under CUDA_HOME
    if (cuda_home := _get_cuda_home()):
        cuda_bin_path = os.path.join(cuda_home, 'bin')
        logger.debug(f"Searching tileiras: {cuda_bin_path}")
        if (res := shutil.which("tileiras", path=cuda_bin_path)):
            bin_path = bin_path + ":" + cuda_bin_path
            return res, bin_path, ld_path

    cuda_home_var = "CUDA_PATH" if is_windows() else "CUDA_HOME"
    raise FileNotFoundError(f"'tileiras' compiler not found, "
                            f"make sure it is available in $PATH or ${cuda_home_var}/bin")


def _try_get_compiler_version(compiler_bin) -> Optional[str]:
    try:
        res = subprocess.run([str(compiler_bin), "--version"],
                             check=True, capture_output=True, text=True)
        return res.stdout
    except Exception:
        return None


@cache
def get_sm_arch() -> str:
    major, minor = get_compute_capability()
    return f'sm_{major}{minor}'


def compile_cubin(
        fname_bytecode: str,
        compiler_options: CompilerOptions,
        sm_arch: str,
        timeout_sec: Optional[int]) -> Path:
    compiler_bin, bin_path, ld_path = _find_compiler_bin()
    fname_cubin = Path(fname_bytecode).with_suffix(".cubin")
    compiler_hints = compiler_options.specialize_for_target(sm_arch)

    command = [
        str(compiler_bin),
        str(fname_bytecode),
        "-o",
        str(fname_cubin),
    ]

    flags = [
        "--gpu-name",
        sm_arch,
        f"-O{compiler_hints.opt_level}",
        "--lineinfo"
    ]

    logger.debug(f"Invoke tile compiler: {' '.join(command + flags)}\n"
                 f"LD_LIBRARY_PATH:{ld_path}\n"
                 f"PATH:{bin_path}")
    try:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = ld_path
        env['PATH'] = bin_path
        subprocess.run(command + flags, env=env, check=True, capture_output=True,
                       timeout=timeout_sec)
    except subprocess.CalledProcessError as e:
        raise TileCompilerExecutionError(e.returncode, e.stderr.decode(), ' '.join(flags),
                                         _try_get_compiler_version(compiler_bin))
    except subprocess.TimeoutExpired:
        message = (f"`tileiras` compiler exceeded timeout {timeout_sec}s. "
                   "Using a smaller tile size may reduce compilation time.")
        raise TileCompilerTimeoutError(message, ' '.join(flags),
                                       _try_get_compiler_version(compiler_bin))

    return fname_cubin
