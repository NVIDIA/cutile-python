# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from cuda.tile._library_lifecycle import nvshmem

# ================================
# NVSHMEM
# ================================


def _install_fake_nvshmem(monkeypatch):
    calls = []

    class NvshmemKernelObject:
        @staticmethod
        def from_handle(handle):
            calls.append(("from_handle", handle))
            return ("kernel_object", handle)

    core = types.ModuleType("nvshmem.core")
    core.NvshmemKernelObject = NvshmemKernelObject
    core.library_init = lambda obj: calls.append(("library_init", obj))
    core.library_finalize = lambda obj: calls.append(("library_finalize", obj))

    package = types.ModuleType("nvshmem")
    package.core = core
    monkeypatch.setitem(sys.modules, "nvshmem", package)
    monkeypatch.setitem(sys.modules, "nvshmem.core", core)
    return calls


def test_library_init(monkeypatch):
    calls = _install_fake_nvshmem(monkeypatch)

    nvshmem.library_init(123)

    assert calls == [
        ("from_handle", 123),
        ("library_init", ("kernel_object", 123)),
    ]


def test_library_finalize(monkeypatch):
    calls = _install_fake_nvshmem(monkeypatch)

    nvshmem.library_finalize(123)

    assert calls == [
        ("from_handle", 123),
        ("library_finalize", ("kernel_object", 123)),
    ]


def test_missing_nvshmem_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "nvshmem", None)
    monkeypatch.delitem(sys.modules, "nvshmem.core", raising=False)

    with pytest.raises(ImportError, match="require the nvshmem Python package"):
        nvshmem.library_init(123)
