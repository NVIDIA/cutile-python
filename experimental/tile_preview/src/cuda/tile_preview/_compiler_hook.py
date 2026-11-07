# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extra environment hook for tileiras when linking tile libraries.

WAR for a tileiras bug (fixed in 13.5U1): tileiras resolves its toolkit path ($CUDA_HOME, else the
parent of its own directory) and loads nvJitLink only under the unversioned name, in a fixed list
of directories that does not match the layout of the CUDA pip wheels. Linking a tile library
(``-l <lib>.tilelib``) therefore fails in a pip-only environment.

* Linux: tileiras looks for ``lib{64,}/libnvJitLink.so``, but the wheels ship only the SONAME file
  (e.g. ``libnvJitLink.so.13``).
* Windows x64: tileiras looks for ``nvJitLink_<ver>.dll`` in ``bin\\x64``, ``bin``,
  ``nvJitLink\\bin`` and ``bin\\nvJitLink\\bin``, but the win_amd64 wheel ships
  ``bin\\x86_64\\nvJitLink_<ver>.dll``.

The hook points tileiras at a temporary mirror of the pip toolkit root in which every entry is
linked to the original and the missing nvJitLink name is added:

* Linux: ``lib`` is a real directory of per-file symlinks plus a ``libnvJitLink.so`` alias.
* Windows: ``bin`` is a real directory of per-entry links plus a ``bin\\x64\\nvJitLink_<ver>.dll``
  link. Directories are linked with junctions and files with hard links.

TODO: remove once tileiras 13.5 Update 1 is the minimum supported version.
"""

import contextlib
import glob
import logging
import os
import platform
import subprocess
import sys
import tempfile
import warnings

logger = logging.getLogger(__name__)

# Maximum length of a cmd.exe command line.
_CMD_MAX_LEN = 8191


def _is_windows():
    return sys.platform == "win32"


def _is_windows_x64():
    return _is_windows() and platform.machine().lower() in ("amd64", "x86_64")


def _find_versioned_nvjitlink_linux(root):
    return sorted(glob.glob(os.path.join(root, "lib", "libnvJitLink.so.[0-9]*")), key=len)


def _find_nvjitlink_win(root):
    return sorted(glob.glob(os.path.join(root, "bin", "x86_64", "nvJitLink*.dll")), key=len)


def _linux_needs_workaround(root):
    if any(os.path.exists(os.path.join(root, d, "libnvJitLink.so")) for d in ("lib64", "lib")):
        return False
    if not _find_versioned_nvjitlink_linux(root):
        logger.debug(f"No nvJitLink SONAME file under {root}/lib; skipping toolkit shim")
        return False
    return True


def _windows_needs_workaround(root):
    if not _is_windows_x64():
        return False
    # Directories in which tileiras 13.5 searches for nvJitLink*.dll.
    search_dirs = (
        os.path.join("bin", "x64"),
        "bin",
        os.path.join("nvJitLink", "bin"),
        os.path.join("bin", "nvJitLink", "bin"),
    )
    if any(glob.glob(os.path.join(root, d, "nvJitLink*.dll")) for d in search_dirs):
        return False
    if not _find_nvjitlink_win(root):
        logger.debug(f"No nvJitLink DLL under {root}\\bin\\x86_64; skipping toolkit shim")
        return False
    return True


def _needs_nvjitlink_workaround(compiler_path):
    root = _toolkit_root(compiler_path)

    if _is_windows():
        return _windows_needs_workaround(root)
    else:
        return _linux_needs_workaround(root)


def _toolkit_root(compiler_path):
    return os.path.dirname(os.path.dirname(os.path.realpath(compiler_path)))


def _populate_toolkit_shim_linux(compiler_path, shim):
    root = _toolkit_root(compiler_path)
    lib = os.path.join(root, "lib")
    for name in os.listdir(root):
        if name != "lib":
            os.symlink(os.path.join(root, name), os.path.join(shim, name))
    shim_lib = os.path.join(shim, "lib")
    os.mkdir(shim_lib)
    for name in os.listdir(lib):
        os.symlink(os.path.join(lib, name), os.path.join(shim_lib, name))
    os.symlink(_find_versioned_nvjitlink_linux(root)[0], os.path.join(shim_lib, "libnvJitLink.so"))


def _link_entries(src_dir, dst_dir, junctions, skip=()):
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            # Queue junctions to be created in a single Windows native command
            junctions.append((src, dst))
        else:
            os.link(src, dst)


def _create_junctions(shim, junctions):
    """Create all junctions with a single cmd.exe run."""
    commands = [f'mklink /J "{dst}" "{src}"' for src, dst in junctions]
    cmdline = f'cmd /s /c "{" && ".join(commands)}"'
    if len(cmdline) < _CMD_MAX_LEN:
        subprocess.run(cmdline, check=True, capture_output=True)
        return

    # Too long for one command line: run a temporary batch file instead.
    bat = os.path.join(shim, "mklink.bat")
    with open(bat, "w", encoding="oem", newline="\r\n") as f:
        f.write("@echo off\n")
        for command in commands:
            f.write(command.replace("%", "%%") + " >nul || exit /b 1\n")
    subprocess.run(["cmd", "/c", bat], check=True, capture_output=True)
    os.remove(bat)


def _populate_toolkit_shim_windows(compiler_path, shim):
    root = _toolkit_root(compiler_path)
    bin_dir = os.path.join(root, "bin")
    junctions = []
    _link_entries(root, shim, junctions, skip=("bin",))

    shim_bin = os.path.join(shim, "bin")
    os.mkdir(shim_bin)
    _link_entries(bin_dir, shim_bin, junctions, skip=("x64",))

    shim_x64 = os.path.join(shim_bin, "x64")
    os.mkdir(shim_x64)
    orig_x64 = os.path.join(bin_dir, "x64")
    if os.path.isdir(orig_x64):
        _link_entries(orig_x64, shim_x64, junctions)
    dll = _find_nvjitlink_win(root)[0]
    os.link(dll, os.path.join(shim_x64, os.path.basename(dll)))

    if junctions:
        _create_junctions(shim, junctions)


def _same_drive(a, b):
    return os.path.splitdrive(a)[0].lower() == os.path.splitdrive(b)[0].lower()


def _shim_parent_dir(compiler_path):
    # [Windows only] os.link() requires src and dst to be on the same drive.
    # Fallback to the root of the toolkit's drive if the temp dir is on a different drive.
    path = os.path.realpath(compiler_path)
    if not _is_windows() or _same_drive(tempfile.gettempdir(), path):
        return None
    return os.path.splitdrive(path)[0] + os.sep


@contextlib.contextmanager
def _tileiras_linker_env(*, compiler_path, version_output):
    """Yield extra environment variables for one tileiras invocation that links tile libraries."""
    if not _needs_nvjitlink_workaround(compiler_path):
        yield {}
        return

    with contextlib.ExitStack() as stack:
        try:
            shim = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="cutile-tileiras-", dir=_shim_parent_dir(compiler_path)))
            if _is_windows():
                _populate_toolkit_shim_windows(compiler_path, shim)
            else:
                _populate_toolkit_shim_linux(compiler_path, shim)
            env = {"CUDA_HOME": shim}
            logger.debug(f"tileiras toolkit shim for {compiler_path}: {shim}")
        except (OSError, subprocess.CalledProcessError) as e:
            # Fallback to tileiras and report tileiras fails to find nvJitLink error when
            # toolkit shim fails.
            warnings.warn(
                f"Cannot create the temporary toolkit directory that works around tileiras not "
                f"finding nvJitLink ({e}). Linking tile libraries may fail.",
                RuntimeWarning)
            env = {}
        yield env
