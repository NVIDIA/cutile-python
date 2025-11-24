# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class TileContextConfig:
    temp_dir: str
    log_keys: list[str]
    compiler_timeout_sec: Optional[int]
    enable_crash_dump: bool


def init_context_config_from_env():
    config = TileContextConfig(
            temp_dir=get_temp_dir_from_env(),
            log_keys=get_log_keys_from_env(),
            compiler_timeout_sec=get_compile_timeout_from_env(),
            enable_crash_dump=get_enable_crash_dump_from_env()
            )
    return config


def get_compile_timeout_from_env() -> Optional[int]:
    key = "CUDA_TILE_COMPILER_TIMEOUT_SEC"
    t = os.environ.get(key)
    if t is not None:
        t = int(t)
        if t <= 0:
            raise ValueError(f"Value of {key} must be positive")
    return t


def get_log_keys_from_env() -> list[str]:
    KEYS = {"CUTILEIR", "TILEIR"}
    env = os.environ.get('CUDA_TILE_LOGS', "")
    ret = []
    for x in env.split(","):
        x = x.upper().strip()
        if len(x) == 0:
            continue
        if x not in KEYS:
            raise RuntimeError(f"Unexpected value {x} in CUDA_TILE_LOGS, "
                               f"supported values are {KEYS}")
        ret.append(x)
    return ret


def _clean_tmp_dir(dir: str):
    shutil.rmtree(dir, ignore_errors=True)


def get_temp_dir_from_env() -> str:
    dir = os.environ.get('CUDA_TILE_TEMP_DIR', "")
    if dir == "":
        dir = tempfile.mkdtemp()
        atexit.register(_clean_tmp_dir, dir)
    return dir


def get_enable_crash_dump_from_env() -> bool:
    key = "CUDA_TILE_ENABLE_CRASH_DUMP"
    env = os.environ.get(key, "0").lower()
    return env in ("1", "true", "yes", "on")
