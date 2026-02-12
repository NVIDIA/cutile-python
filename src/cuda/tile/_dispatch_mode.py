# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager

from ._exception import TileStaticEvalError
from ._ir.hir import StaticEvalKind


class DispatchMode:
    @staticmethod
    def get_current() -> "DispatchMode":
        return _current_mode.mode

    @contextmanager
    def as_current(self):
        old_mode = _current_mode.mode
        _current_mode.mode = self
        try:
            yield self
        finally:
            _current_mode.mode = old_mode

    def call_tile_function_from_host(self, func, args, kwargs):
        raise NotImplementedError()


class NormalMode(DispatchMode):
    def call_tile_function_from_host(self, func, args, kwargs):
        raise RuntimeError("Tile functions can only be called from tile code.")


class StaticEvalMode(DispatchMode):
    def __init__(self, kind: StaticEvalKind):
        self._kind = kind

    def call_tile_function_from_host(self, func, args, kwargs):
        from cuda.tile import static_eval, static_assert
        if func in (static_eval, static_assert):
            what = f"{func.__name__}() cannot be used"
        else:
            what = "Tile functions cannot be called"

        where = self._kind._value_
        raise TileStaticEvalError(f"{what} inside {where}.")


class _CurrentModeTL(threading.local):
    mode: DispatchMode = NormalMode()


_current_mode = _CurrentModeTL()
