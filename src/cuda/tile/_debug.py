# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# Comma separated string for selective debug logging
# values are case in-sensitive
CUDA_TILE_LOG_KEYS = {"CUTILEIR", "TILEIR"}


def parse_cuda_tile_logs_env():
    env = os.environ.get('CUDA_TILE_LOGS', "")
    ret = []
    for x in env.split(","):
        x = x.upper().strip()
        if len(x) == 0:
            continue
        if x not in CUDA_TILE_LOG_KEYS:
            raise RuntimeError(f"Unexpected value {x} in CUDA_TILE_LOGS, "
                               f"supported values are {CUDA_TILE_LOG_KEYS}")
        ret.append(x)
    return ret


CUDA_TILE_LOGS = parse_cuda_tile_logs_env()

CUDA_TILE_DUMP_TILEIR = os.environ.get('CUDA_TILE_DUMP_TILEIR', None)
CUDA_TILE_DUMP_BYTECODE = os.environ.get('CUDA_TILE_DUMP_BYTECODE', None)
CUDA_TILE_TEMP_DIR = os.environ.get('CUDA_TILE_TEMP_DIR', None)


CUDA_TILE_TESTING_DISABLE_DIV = (
    os.environ.get("CUDA_TILE_TESTING_DISABLE_DIV", "0") == "1")


CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER = (
    os.environ.get("CUDA_TILE_TESTING_DISABLE_TOKEN_ORDER", "0") == "1")


logger = logging.getLogger(__name__)
