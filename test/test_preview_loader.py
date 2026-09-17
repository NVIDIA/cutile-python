# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

from cuda.tile import _preview_loader


_mock_type = type("PreviewForeignCall", (), {})
_preview = types.SimpleNamespace(PreviewForeignCall=_mock_type)


def _missing_import(name):
    raise ModuleNotFoundError(name="cuda.tile_preview")


@pytest.mark.parametrize("mock_import,expected", [
    (_missing_import, None),
    (lambda name: _preview, _mock_type),
])
def test_import_preview_package(monkeypatch, mock_import, expected):
    calls = []

    def counted_import(name):
        calls.append(name)
        return mock_import(name)

    get_type = _preview_loader.get_preview_foreign_call_type
    get_type.cache_clear()
    monkeypatch.setattr(_preview_loader.importlib, "import_module", counted_import)
    try:
        assert get_type() is expected
        assert get_type() is expected  # cached
        assert len(calls) == 1
    finally:
        get_type.cache_clear()


def test_import_fails_on_dependency(monkeypatch):
    error = ModuleNotFoundError(name="preview_dependency")

    def fail_import(name):
        raise error

    get_type = _preview_loader.get_preview_foreign_call_type
    get_type.cache_clear()
    monkeypatch.setattr(_preview_loader.importlib, "import_module", fail_import)
    try:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            get_type()
        assert exc_info.value is error
    finally:
        get_type.cache_clear()


def test_import_fails_when_type_missing(monkeypatch):
    preview = types.SimpleNamespace()
    get_type = _preview_loader.get_preview_foreign_call_type
    get_type.cache_clear()
    monkeypatch.setattr(_preview_loader.importlib, "import_module", lambda name: preview)
    try:
        with pytest.raises(AttributeError):
            get_type()
    finally:
        get_type.cache_clear()
