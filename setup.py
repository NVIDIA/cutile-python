# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from distutils import file_util
from setuptools import setup, find_namespace_packages, Distribution
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import sys


project_root = os.path.dirname(os.path.abspath(__file__))
is_windows = sys.platform == "win32"


class CudaTileDistribution(Distribution):
    global_options = Distribution.global_options + [
        ('include-cuda-lang', None, 'Include cuda.lang and its native compiler'),
    ]

    def __init__(self, attrs=None):
        self.include_cuda_lang = False
        super().__init__(attrs)

    def parse_command_line(self):
        result = super().parse_command_line()
        # Configure packages and extensions before any build commands are finalized.
        if self.include_cuda_lang:
            lang_src = os.path.join("experimental", "cuda-lang", "src")
            self.packages = [*self.packages, *find_namespace_packages(
                where=lang_src, include=["cuda.lang", "cuda.lang.*"])]
            self.package_dir = dict(self.package_dir, **{
                "cuda.lang": os.path.join(lang_src, "cuda", "lang"),
            })
            self.ext_modules.append(Extension("cuda.lang._host_jit", []))
        return result


class BuildExtWithCmake(build_ext):
    user_options = build_ext.user_options + [
        ('disable-internal', None, 'Disable building internal extension'),
        ('minimal-internal', None, 'Enable building minimal internal extension'),
        ('enable-dev-features', None, 'Enable development-only features'),
        ('custom-static-libnvvm=', None, 'Include a custom NVVM built from a static library'),
        ('custom-libdevice=', None, 'Include a custom libdevice object'),
        ('custom-ptxas=', None, 'Include a custom ptxas binary'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.disable_internal = False
        self.minimal_internal = False
        self.enable_dev_features = False
        self.custom_static_libnvvm = None
        self.custom_libdevice = None
        self.custom_ptxas = None

    def finalize_options(self):
        super().finalize_options()

    def _make(self, build_dir: str, build_type: str, parallel: int):
        if is_windows:
            self.spawn(["msbuild", f"{build_dir}/cuda-tile-python.sln",
                        f"-maxcpucount:{parallel}",
                        f"/p:Configuration={build_type}",
                        "/t:_cext"])
        else:
            self.spawn(["cmake", "--build", build_dir, "--parallel", str(parallel)])
        # TODO: ideally, we should "make install" the library somewhere, so that CMake removes
        #   any build RPATHs etc. But I'll leave that for another day.

    def _cmake(self, build_dir: str, build_type: str, dlpack_path: str, xla_path: str):
        cmake_cmd = ["cmake", "-B", build_dir, project_root,
                     f"-DDLPACK_PATH={dlpack_path}",
                     f"-DXLA_PATH={xla_path}",
                     f"-DCMAKE_BUILD_TYPE={build_type}",
                     f"-DPython_EXECUTABLE={sys.executable}",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        if self.disable_internal:
            cmake_cmd.append("-DDISABLE_INTERNAL=1")
        if self.minimal_internal:
            cmake_cmd.append("-DMINIMAL_INTERNAL=1")
        if self.enable_dev_features:
            cmake_cmd.append("-DENABLE_DEV_FEATURES=1")
        if self.custom_static_libnvvm is not None:
            cmake_cmd.append(f"-DCUSTOM_STATIC_LIBNVVM={self.custom_static_libnvvm}")
        if self.custom_libdevice is not None:
            cmake_cmd.append(f"-DCUSTOM_LIBDEVICE={self.custom_libdevice}")
        if self.custom_ptxas is not None:
            cmake_cmd.append(f"-DCUSTOM_PTXAS={self.custom_ptxas}")
        self.spawn(cmake_cmd)

    def run(self):
        build_dir = os.getenv("CUDA_TILE_CEXT_BUILD_DIR")
        if build_dir is None or build_dir == "":
            if self.editable_mode:
                build_dir = os.path.join(project_root, "build")
            else:
                build_dir = self.build_temp

        build_type = "Debug" if self.debug else "Release"
        dlpack_path = os.getenv("CUDA_TILE_CMAKE_DLPACK_PATH", "")
        xla_path = os.getenv("CUDA_TILE_CMAKE_XLA_PATH", "")
        parallel = 1 if self.parallel is None else self.parallel
        self._cmake(build_dir, build_type, dlpack_path, xla_path)
        self._make(build_dir, build_type, parallel)

        for ext in self.extensions:
            src_dir = _get_csrc_dir(ext.name)
            ext_name = _get_build_lib_filename(ext.name)
            if is_windows:
                ext_build_path = os.path.join(build_dir, src_dir, build_type, ext_name)
            else:
                ext_build_path = os.path.join(build_dir, src_dir, ext_name)
            ext_path = self.get_ext_fullpath(ext.name)
            # Create a symlink to the build directory if in editable mode, otherwise copy
            link = "sym" if self.editable_mode else None
            file_util.copy_file(ext_build_path, ext_path, update=1, link=link)

        if self.distribution.include_cuda_lang:
            bin_dir = os.path.join(os.path.dirname(
                self.get_ext_fullpath("cuda.lang._host_jit")), "bin")
            os.makedirs(bin_dir, exist_ok=True)
            file_util.copy_file(os.path.join(build_dir, "internal", "mlir2cubin", "mlir2cubin"),
                                os.path.join(bin_dir, "mlir2cubin"),
                                update=1, link="sym" if self.editable_mode else None)


def _get_csrc_dir(ext_name: str):
    if ext_name == "cuda.lang._host_jit":
        return os.path.join("internal", "host_jit")
    prefix = "cuda.tile._"
    assert ext_name.startswith(prefix)
    return ext_name[len(prefix):]


def _get_build_lib_filename(ext_name: str):
    if ext_name == "cuda.lang._host_jit":
        return "_host_jit.so"
    name = ext_name.split(".")[-1]
    if is_windows:
        return f"{name}.dll"
    else:
        return f"lib{name}.so"


setup(
    distclass=CudaTileDistribution,
    ext_modules=[
        Extension("cuda.tile._cext", []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    )
)
