# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    log_hir: bool = False
    log_ir: bool = False
    log_flattened_ir: bool = False
    log_mlir: bool = False
    log_nvvm: bool = False
    log_ptx: bool = False


_LOG_KEYS = {
    'HIR': 'log_hir',
    'IR': 'log_ir',
    'FLATIR': 'log_flattened_ir',
    'MLIR': 'log_mlir',
    'NVVM': 'log_nvvm',
    'PTX': 'log_ptx',
}


_config: LoggingConfig | None = None


def get_log_flags() -> LoggingConfig:
    global _config
    if _config is not None:
        return _config

    env = [key.strip() for key in os.environ.get('CUDA_LANG_LOGS', "").upper().split(",") if key]
    _config = LoggingConfig()
    for key in env:
        if key in _LOG_KEYS:
            setattr(_config, _LOG_KEYS[key], True)
        else:
            raise RuntimeError(
                f"Unknown logging key {key} in CUDA_LANG_LOGS, "
                f"expected one of {list(_LOG_KEYS.keys())}"
            )
    return _config
