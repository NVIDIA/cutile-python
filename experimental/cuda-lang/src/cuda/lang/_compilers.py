# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os.path
import shutil
import subprocess
from functools import cache
import importlib.metadata
from importlib.metadata import PackageNotFoundError
from tempfile import TemporaryDirectory
from typing import Iterable

from cuda.tile._compile import _get_cuda_home, is_windows, \
    _find_file, _get_default_cuda_toolkit_paths
from cuda.tile._exception import CompilerExecutionError, Loc
from cuda.tile._cext import NVVM


class PtxCompiler:
    def __init__(self, ptxas_path: str):
        self._ptxas_path = ptxas_path

    @staticmethod
    @cache
    def get() -> "PtxCompiler":
        return PtxCompiler(_find_ptxas())

    def compile(self, ptx: bytes | str, gpu_name: str) -> bytes:
        if isinstance(ptx, str):
            ptx = ptx.encode()

        with TemporaryDirectory(prefix="ptxas-scratch") as tmpdir:
            input_path = os.path.join(tmpdir, "input.ptx")
            output_path = os.path.join(tmpdir, "output.cubin")
            with open(input_path, "wb") as f:
                f.write(ptx)

            argv = [self._ptxas_path,
                    input_path,
                    "-o", output_path,
                    "--gpu-name", gpu_name]
            try:
                subprocess.run(argv, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                raise CompilerExecutionError(e.returncode, e.stderr.decode(), Loc.unknown(),
                                             compiler_flags=' '.join(argv),
                                             compiler_version=None)

            with open(output_path, "rb") as f:
                return f.read()


def compile_nvvm(bitcode: bytes, arch: str) -> bytes:
    nvvm, libdevice = get_nvvm_and_libdevice()
    options = ["-arch=" + arch]
    try:
        program = nvvm.create_program()
        program.add_module(bitcode, "main")
        program.add_module(libdevice, "libdevice")
        return program.compile(options)
    except RuntimeError as e:
        raise CompilerExecutionError(
            return_code=None,
            message=str(e),
            loc=Loc.unknown(),
            compiler_flags=" ".join(options),
            compiler_version=None,
        ) from e


# TODO: isolate in a subprocess
@cache
def get_nvvm_and_libdevice() -> tuple[NVVM, bytes]:
    nvvm_so_path, libdevice_path = _find_nvvm_and_libdevice()
    with open(libdevice_path, "rb") as f:
        libdevice = f.read()
    return NVVM(nvvm_so_path), libdevice


def _find_nvvm_compiler_file(name: str) -> str | None:
    compiler_dir = os.environ.get("CUDA_LANG_NVVM_COMPILER_DIR")
    if not compiler_dir:
        return None
    path = os.path.join(compiler_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CUDA_LANG_NVVM_COMPILER_DIR is missing required file: {path}")
    return path


def _find_nvvm_and_libdevice() -> tuple[str, str]:
    nvvm_name = "nvvm.dll" if is_windows() else "libnvvm.so"
    nvvm_path = _find_nvvm_compiler_file(nvvm_name)
    if nvvm_path is not None:
        return nvvm_path, _find_nvvm_compiler_file("libdevice.10.bc")

    # Prefer bundled libnvvm when no compiler directory is configured.
    included_lib_name = "nvvm.dll" if is_windows() else "libnvvm.so"
    bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")
    included_path = os.path.join(bin_path, included_lib_name)
    included_libdevice_path = os.path.join(bin_path, "libdevice.10.bc")
    if os.path.exists(included_path) and os.path.exists(included_libdevice_path):
        return included_path, included_libdevice_path

    if is_windows():
        lib_names = ["nvvm64_40_0.dll"]
    else:
        lib_names = ["libnvvm.so.4"]

    path = _find_file_in_python_package("nvidia-nvvm", lib_names)
    if path is not None:
        libdevice_path = _find_file_in_python_package("nvidia-nvvm", ["libdevice.10.bc"])
        if libdevice_path is not None:
            return path, libdevice_path

    for prefix in _get_default_cuda_toolkit_paths():
        res = _find_file([prefix], ["nvvm/lib64", "lib64", "nvvm/bin", "nvvm/bin/x64"],
                         lib_names, require_executable=False)
        if res is not None:
            nvvm_so_path, _ = res
            res = _find_file([prefix], ["nvvm/libdevice"], ["libdevice.10.bc"],
                             require_executable=False)
            if res is not None:
                libdevice_path, _ = res
                return nvvm_so_path, libdevice_path

    cuda_home_var = "CUDA_PATH" if is_windows() else "CUDA_HOME"
    raise FileNotFoundError(
        f"'nvvm' dynamic library not found. Please provide it in one of the following ways:\n"
        f"  - Install 'nvidia-nvvm' package from PyPI: `pip install nvidia-nvvm`;\n"
        f"  - Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads).\n"
        f"    If non-standard installation path is used, make sure ${cuda_home_var} is defined.\n")


def _find_ptxas() -> str:
    binary_name = "ptxas.exe" if is_windows() else "ptxas"
    path = _find_nvvm_compiler_file(binary_name)
    if path is not None:
        return path

    # Prefer bundled ptxas when no compiler directory is configured.
    included_lib_name = "ptxas.exe" if is_windows() else "ptxas"
    included_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "bin", included_lib_name)
    if os.path.exists(included_path):
        return included_path

    binary_name = "ptxas.exe" if is_windows() else "ptxas"
    path = _find_file_in_python_package("nvidia-cuda-nvcc", [binary_name])
    if path is not None:
        return path

    path = shutil.which("ptxas")
    if path is not None:
        return path

    cuda_paths = _get_default_cuda_toolkit_paths()
    res = _find_file(cuda_paths, ["bin"], [binary_name], require_executable=True)
    if res is not None:
        path, _ = res
        return path

    cuda_home_var = "CUDA_PATH" if is_windows() else "CUDA_HOME"
    raise FileNotFoundError(
        f"'ptxas' compiler not found. Please provide it in one of the following ways:\n"
        f"  - Put 'ptxas' in the current $PATH;\n"
        f"  - Install 'nvidia-cuda-nvcc' package from PyPI: `pip install nvidia-cuda-nvcc`;\n"
        f"  - Install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads).\n"
        f"    If non-standard installation path is used, make sure ${cuda_home_var} is defined.\n")


def _get_all_cuda_toolkit_paths() -> list[str]:
    cuda_paths = []
    cuda_home = _get_cuda_home()
    if cuda_home is not None:
        cuda_paths.append(cuda_home)
    cuda_paths.extend(_get_default_cuda_toolkit_paths())
    return cuda_paths


def _find_file_in_python_package(distribution_name: str,
                                 file_basenames: Iterable[str]) -> str | None:
    try:
        files = importlib.metadata.files(distribution_name)
    except PackageNotFoundError:
        return None

    for f in files:
        if f.name in file_basenames:
            return str(f.locate())
    return None
