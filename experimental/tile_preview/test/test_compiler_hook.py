# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys

import pytest

from cuda.tile_preview import _compiler_hook
from cuda.tile_preview._compiler_hook import _tileiras_linker_env


# ============= Linux tests =============
linux_only = pytest.mark.skipif(sys.platform == "win32", reason="Linux layout")

_VERSION = "V13.5.37"


def _make_pip_root(tmp_path, libs):
    root = tmp_path / "cu13"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "nvvm" / "libdevice").mkdir(parents=True)
    tileiras = root / "bin" / "tileiras"
    tileiras.write_text("")
    tileiras.chmod(0o755)
    (root / "bin" / "ptxas").write_text("")
    for name in libs:
        (root / "lib" / name).write_text("")
    return root, str(tileiras)


@linux_only
def test_shim_mirrors_soname_only_layout(tmp_path):
    libs = ["libnvJitLink.so.13", "libnvvm.so.4", "libnvrtc.so.13"]
    root, tileiras = _make_pip_root(tmp_path, libs)
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        assert set(env) == {"CUDA_HOME"}
        shim = env["CUDA_HOME"]
        assert not shim.startswith(str(root))
        for name in ("bin", "nvvm"):
            assert os.readlink(os.path.join(shim, name)) == str(root / name)
        assert os.path.exists(os.path.join(shim, "bin", "ptxas"))
        for name in libs:
            assert os.readlink(os.path.join(shim, "lib", name)) == str(root / "lib" / name)
        assert os.readlink(os.path.join(shim, "lib", "libnvJitLink.so")) == \
            str(root / "lib" / "libnvJitLink.so.13")
        assert os.environ.get("CUDA_HOME") != shim
    assert not os.path.exists(shim)

    # Removing the shim must not touch the linked originals.
    assert (root / "nvvm" / "libdevice").exists()
    assert sorted(os.listdir(root / "lib")) == sorted(libs)


@linux_only
def test_shim_removed_on_error(tmp_path):
    _, tileiras = _make_pip_root(tmp_path, ["libnvJitLink.so.13"])
    with pytest.raises(RuntimeError):
        with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
            shim = env["CUDA_HOME"]
            raise RuntimeError
    assert not os.path.exists(shim)


@linux_only
@pytest.mark.parametrize("libs", [
    ["libnvJitLink.so", "libnvJitLink.so.13", "libnvvm.so.4"],
    ["libnvvm.so.4"],
])
def test_no_shim_lib_already_exists(tmp_path, libs):
    _, tileiras = _make_pip_root(tmp_path, libs)
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        assert env == {}


@linux_only
def test_link_fails(tmp_path, monkeypatch):
    _, tileiras = _make_pip_root(tmp_path, ["libnvJitLink.so.13", "libnvvm.so.4"])

    def fail(*args, **kwargs):
        raise OSError("Linking failed")

    monkeypatch.setattr(os, "symlink", fail)
    with pytest.warns(RuntimeWarning, match="Linking failed"):
        with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
            assert env == {}


# ============= Windows x64 tests =============
_machine = platform.machine().lower()
windows_x64_only = pytest.mark.skipif(
    not (sys.platform == "win32" and _machine in ("amd64", "x86_64")),
    reason="Windows x64 layout")


def _make_win_root(tmp_path, dll_path=("bin", "x86_64", "nvJitLink_130_0.dll")):
    root = tmp_path / "cu13"
    (root / "bin").mkdir(parents=True)
    (root / "nvvm" / "libdevice").mkdir(parents=True)
    (root / "bin" / "tileiras.exe").write_text("")
    (root / "bin" / "ptxas.exe").write_text("")
    if dll_path is not None:
        dll = root.joinpath(*dll_path)
        dll.parent.mkdir(parents=True, exist_ok=True)
        dll.write_text("")
    return root, str(root / "bin" / "tileiras.exe")


@windows_x64_only
def test_win_shim_exposes_dll_in_bin_x64(tmp_path):
    root, tileiras = _make_win_root(tmp_path)
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        assert set(env) == {"CUDA_HOME"}
        shim = env["CUDA_HOME"]
        assert os.path.exists(os.path.join(shim, "bin", "x64", "nvJitLink_130_0.dll"))
        assert os.path.exists(os.path.join(shim, "bin", "ptxas.exe"))
        assert os.path.exists(os.path.join(shim, "bin", "x86_64", "nvJitLink_130_0.dll"))
        assert os.path.exists(os.path.join(shim, "nvvm", "libdevice"))
        assert sorted(os.listdir(root / "bin")) == ["ptxas.exe", "tileiras.exe", "x86_64"]
    assert not os.path.exists(shim)

    # Removing the shim must not touch the linked originals.
    assert (root / "nvvm" / "libdevice").exists()
    assert (root / "bin" / "x86_64" / "nvJitLink_130_0.dll").exists()


@windows_x64_only
def test_win_shim_keeps_existing_bin_x64_entries(tmp_path):
    root, tileiras = _make_win_root(tmp_path)
    (root / "bin" / "x64").mkdir()
    (root / "bin" / "x64" / "other.dll").write_text("")
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        x64 = os.path.join(env["CUDA_HOME"], "bin", "x64")
        assert sorted(os.listdir(x64)) == ["nvJitLink_130_0.dll", "other.dll"]
    assert sorted(os.listdir(root / "bin" / "x64")) == ["other.dll"]


@windows_x64_only
@pytest.mark.parametrize("dll_path", [
    ("bin", "x64", "nvJitLink.dll"),
    ("bin", "nvJitLink.dll"),
    ("nvJitLink", "bin", "nvJitLink.dll"),
    ("bin", "nvJitLink", "bin", "nvJitLink.dll"),
    ("bin", "x64", "nvJitLink_130_0.dll"),
    ("bin", "nvJitLink_130_0.dll"),
    None,
])
# DLL already in a searched dir, or no wheel DLL at all.
def test_win_no_shim(tmp_path, dll_path):
    _, tileiras = _make_win_root(tmp_path, dll_path)
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        assert env == {}


@pytest.mark.skipif(
    not (sys.platform == "win32" and _machine == "arm64"), reason="Windows arm64 layout")
def test_win_no_shim_on_arm64(tmp_path):
    _, tileiras = _make_win_root(tmp_path)
    with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
        assert env == {}


@windows_x64_only
def test_win_link_fails(tmp_path, monkeypatch):
    _, tileiras = _make_win_root(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("Linking failed")

    monkeypatch.setattr(os, "link", fail)
    with pytest.warns(RuntimeWarning, match="Linking failed"):
        with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
            assert env == {}


@windows_x64_only
def test_win_temp_dir_cannot_be_created(tmp_path, monkeypatch):
    _, tileiras = _make_win_root(tmp_path)

    def fail(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(_compiler_hook.tempfile, "TemporaryDirectory", fail)
    with pytest.warns(RuntimeWarning, match="read-only"):
        with _tileiras_linker_env(compiler_path=tileiras, version_output=_VERSION) as env:
            assert env == {}
