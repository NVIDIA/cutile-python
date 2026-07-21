# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path
import subprocess
import sys
from .util import require_blackwell_cc100

TUTORIAL_ROOT = Path(__file__).parent.parent / "tutorial"
TUTORIALS = tuple(sorted(TUTORIAL_ROOT.rglob("*.py")))


@require_blackwell_cc100()
@pytest.mark.parametrize("tutorial", TUTORIALS, ids=str)
def test_tutorial(tutorial):
    subprocess.run([sys.executable, tutorial], check=True)
