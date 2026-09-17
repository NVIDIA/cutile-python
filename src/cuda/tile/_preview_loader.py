# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from functools import cache


@cache
def get_preview_foreign_call_type():
    try:
        preview = importlib.import_module("cuda.tile_preview")
    except ModuleNotFoundError as exc:
        if exc.name != "cuda.tile_preview":
            raise
        return None
    return preview.PreviewForeignCall
