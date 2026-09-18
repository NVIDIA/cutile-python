# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

from cuda.tile._by_target import ByTarget, UNSPECIFIED


@dataclass(frozen=True)
class CompilerOptions:
    num_ctas: None | int | ByTarget[int] = None
    occupancy: None | int | ByTarget[int] = None
    opt_level: int | ByTarget[int] = 3
    num_worker_warps: None | int | ByTarget[int] = None

    def __post_init__(self):
        for field in dataclasses.fields(self):
            validator = globals()[f"_validate_{field.name}"]
            value = getattr(self, field.name)
            if isinstance(value, ByTarget):
                for target_val in value._by_target.values():
                    validator(target_val)
                if value._default is not UNSPECIFIED:
                    validator(value._default)
            else:
                validator(value)

    def hints_by_target(self) -> Mapping[str, Mapping[str, Any]]:
        res: dict[str, dict[str, Any]] = defaultdict(dict)
        for field in dataclasses.fields(CompilerOptions):
            value = getattr(self, field.name)
            if isinstance(value, ByTarget):
                for target_name, v in value._by_target.items():
                    res[target_name][field.name] = v
                default = value._default if value._default is not UNSPECIFIED else field.default
            else:
                default = value
            res["default"][field.name] = default
        return dict(res)

    def opt_level_for_target(self, target_name: str) -> int:
        opt_level = self.opt_level
        if isinstance(opt_level, ByTarget):
            if target_name in opt_level._by_target:
                opt_level = opt_level._by_target[target_name]
            elif opt_level._default is not UNSPECIFIED:
                opt_level = opt_level._default
            else:
                opt_level = CompilerOptions.__dataclass_fields__["opt_level"].default
        assert isinstance(opt_level, int)
        return opt_level


def _validate_num_ctas(num_ctas: None | int):
    if num_ctas is not None:
        if num_ctas > 16 or num_ctas < 1:
            raise ValueError(f'num_ctas should be [1, 16], got {num_ctas}')
        if (num_ctas & (num_ctas - 1)) != 0:
            raise ValueError(f'num_ctas should be power of 2, got {num_ctas}')


def _validate_occupancy(occupancy: None | int):
    if occupancy is not None:
        if occupancy < 1 or occupancy > 32:
            raise ValueError(f'occupancy should be [1, 32], got {occupancy}')


def _validate_opt_level(opt_level: None | int):
    if opt_level is not None:
        if opt_level < 0 or opt_level > 3:
            raise ValueError(f'opt_level should be [0, 3], got {opt_level}')


def _validate_num_worker_warps(num_worker_warps: None | int):
    if num_worker_warps is not None:
        if num_worker_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f'num_worker_warps should be a power of 2 in [1, 32],'
                             f' got {num_worker_warps}')
