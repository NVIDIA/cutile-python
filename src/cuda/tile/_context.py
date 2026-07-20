# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
from contextlib import contextmanager
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

from cutile_cache._env import get_cache_dir_from_env


@dataclass
class TileContextConfig:
    temp_dir: str
    compiler_timeout_sec: Optional[float]
    enable_crash_dump: bool
    cache_dir: Optional[str]
    cache_size_limit: int
    log_cutile_ir: bool = False
    log_tileir: bool = False


def init_context_config_from_env():
    config = TileContextConfig(
            temp_dir=get_temp_dir_from_env(),
            compiler_timeout_sec=get_compile_timeout_from_env(),
            enable_crash_dump=get_enable_crash_dump_from_env(),
            cache_dir=get_cache_dir_from_env(),
            cache_size_limit=get_cache_size_limit_from_env(),
            **get_log_keys_from_env(),
            )
    return config


def get_compile_timeout_from_env() -> Optional[float]:
    key = "CUDA_TILE_COMPILER_TIMEOUT_SEC"
    t = os.environ.get(key)
    if t is not None:
        t = float(t)
        if t <= 0:
            raise ValueError(f"Value of {key} must be positive")
    return t


# Map from CUDA_TILE_LOGS env variable value to TileContextConfig attribute name
_LOG_KEYS = {
    "CUTILEIR": "log_cutile_ir",
    "TILEIR": "log_tileir"
}


def get_log_keys_from_env() -> dict[str, bool]:
    env = os.environ.get('CUDA_TILE_LOGS', "")
    ret = dict()
    for x in env.split(","):
        x = x.upper().strip()
        if len(x) == 0:
            continue
        try:
            attr_name = _LOG_KEYS[x]
        except KeyError:
            raise RuntimeError(f"Unexpected value {x} in CUDA_TILE_LOGS, "
                               f"supported values are {list(_LOG_KEYS.keys())}")
        ret[attr_name] = True
    return ret


def _clean_tmp_dir(dir: str):
    shutil.rmtree(dir, ignore_errors=True)


def get_temp_dir_from_env() -> str:
    dir = os.environ.get('CUDA_TILE_TEMP_DIR', "")
    if dir == "":
        dir = tempfile.mkdtemp()
        atexit.register(_clean_tmp_dir, dir)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def get_enable_crash_dump_from_env() -> bool:
    key = "CUDA_TILE_ENABLE_CRASH_DUMP"
    env = os.environ.get(key, "0").lower()
    return env in ("1", "true", "yes", "on")


def get_cache_size_limit_from_env() -> int:
    return int(os.environ.get("CUDA_TILE_CACHE_SIZE", 1 << 31))  # 2GB


@contextmanager
def compiler_timeout(timeout_sec: float):
    """Context manager that temporarily sets the compiler timeout.

    .. note::
        This function is not thread-safe. It modifies global state
        shared by all threads.

    Example::

        with ct.compiler_timeout(10):
            ct.launch(stream, grid, kernel, args)
    """
    from cuda.tile._cext import default_tile_context
    old = default_tile_context.config.compiler_timeout_sec
    default_tile_context.config.compiler_timeout_sec = timeout_sec
    try:
        yield
    finally:
        default_tile_context.config.compiler_timeout_sec = old
