# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import setup

# cuda-tile-preview is released together with cuda-tile under the same version.
version = (Path(__file__).parent / "src/cuda/tile_preview/VERSION").read_text().strip()
setup(install_requires=[f"cuda-tile=={version}"])
