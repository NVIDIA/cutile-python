# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cuda.lang", reason="Skipping cuda-lang test: module not found")
