# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._passes.ir2mlir.location import ir_loc_to_mlir_location
from cuda.tile._exception import Loc


def test_ir_location_conversion():
    location = Loc(
        filename="kernel.py",
        line=2,
        col=3,
        last_line=2,
        end_col=7,
    )

    assert str(ir_loc_to_mlir_location(location)) == '"kernel.py":2:4 to :7'


def test_ir_call_site_location_conversion():
    caller = Loc(filename="kernel.py", line=12, col=8)
    callee = Loc(filename="helper.py", line=4, col=4, call_site=caller)

    location = ir_loc_to_mlir_location(callee)

    assert str(location) == 'callsite("helper.py":4:5 at "kernel.py":12:9)'


def test_unknown_ir_location_conversion():
    assert ir_loc_to_mlir_location(Loc.unknown()) is None
