# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def get_cache_dir_from_env() -> str | None:
    home_cache = os.path.join(os.path.expanduser("~"), ".cache")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", home_cache)
    else:
        base = os.environ.get("XDG_CACHE_HOME", home_cache)
    default = os.path.join(base, "cutile-python")
    env = os.environ.get("CUDA_TILE_CACHE_DIR", default)
    if env.strip().lower() in ("0", "off", "none", ""):
        return None
    return env
