# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from distutils import file_util
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import os
import sys


project_root = os.path.dirname(os.path.abspath(__file__))
is_windows = sys.platform == "win32"


class BuildExtWithCmake(build_ext):
    user_options = build_ext.user_options + [
        ('disable-internal', None, 'Disable building internal extension'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.disable_internal = False

    def finalize_options(self):
        super().finalize_options()

    def _make(self, build_dir: str, build_type: str, parallel: int):
        if is_windows:
            self.spawn(["msbuild", f"{build_dir}/cuda-tile-python.sln",
                        f"-maxcpucount:{parallel}",
                        f"/p:Configuration={build_type}",
                        "/t:_cext"])
        else:
            self.spawn(["make", "-C", build_dir, "-j", str(parallel)])
        # TODO: ideally, we should "make install" the library somewhere, so that CMake removes
        #   any build RPATHs etc. But I'll leave that for another day.

    def _cmake(self, build_dir: str, build_type: str, dlpack_path: str):
        cmake_cmd = ["cmake", "-B", build_dir, project_root,
                     f"-DDLPACK_PATH={dlpack_path}",
                     f"-DCMAKE_BUILD_TYPE={build_type}",
                     "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
        if self.disable_internal:
            cmake_cmd.append("-DDISABLE_INTERNAL=1")
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
        parallel = 1 if self.parallel is None else self.parallel
        self._cmake(build_dir, build_type, dlpack_path)
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
            file_util.copy_file(ext_build_path, ext_path, update=1, link=link,
                                dry_run=self.dry_run)


def _get_csrc_dir(ext_name: str):
    prefix = "cuda.tile._"
    assert ext_name.startswith(prefix)
    return ext_name[len(prefix):]


def _get_build_lib_filename(ext_name: str):
    name = ext_name.split(".")[-1]
    if is_windows:
        return f"{name}.dll"
    else:
        return f"lib{name}.so"


setup(
    ext_modules=[
        Extension("cuda.tile._cext", []),
    ],
    cmdclass=dict(
        build_ext=BuildExtWithCmake,
    )
)
