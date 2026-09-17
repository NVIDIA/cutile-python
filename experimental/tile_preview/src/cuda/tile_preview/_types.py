# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tilelib:
    path: str | Path
    version: str

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)):
            raise TypeError("Tilelib path must be a string or pathlib.Path")
        resolved = Path(self.path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Tilelib is not a file: {resolved}")
        object.__setattr__(self, "path", str(resolved))
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Tilelib version must be a non-empty string")
