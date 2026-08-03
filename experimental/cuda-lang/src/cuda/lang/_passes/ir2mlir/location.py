# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang import _mlir as mlir
from cuda.tile._exception import Loc


def ir_loc_to_mlir_location(loc: Loc) -> mlir.Location | None:
    if loc.is_unknown() or loc.filename is None:
        return None

    start_column = loc.col + 1
    end_line = loc.last_line if loc.last_line is not None else loc.line
    end_column = loc.end_col if loc.end_col is not None else start_column
    source_loc = mlir.FileLineColRange(
        filename=mlir.StringAttr(value=loc.filename),
        start_line=loc.line,
        start_column=start_column,
        end_line=end_line,
        end_column=end_column,
    )

    if loc.call_site is None:
        return source_loc

    caller = ir_loc_to_mlir_location(loc.call_site)
    if caller is None:
        return source_loc
    return mlir.CallSiteLoc(callee=source_loc, caller=caller)
