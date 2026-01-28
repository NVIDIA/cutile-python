# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.ir import Block
from cuda.tile._ir.type import TileTy, ArrayTy
from cuda.tile._datatype import float8_e4m3fn, float8_e5m2, DType
from cuda.tile._exception import TileUnsupportedFeatureError

FLOAT8_DTYPES = (float8_e4m3fn, float8_e5m2)


def check_ampere_fp8(root_block: Block, sm_arch: str) -> None:
    # Technically sm_89 (Ada Lovelace) supports FP8, but tileiras doesn't have support for it yet.
    if not sm_arch.startswith("sm_8"):
        return

    for op in root_block.traverse():
        for var in op.all_inputs():
            ty = var.try_get_type()
            dtype = None
            if isinstance(ty, (TileTy, ArrayTy)):
                dtype = ty.dtype
            elif isinstance(ty, DType):
                dtype = ty
            if dtype in FLOAT8_DTYPES:
                raise TileUnsupportedFeatureError(
                    "float8 dtype is not supported on Ampere or Ada Lovelace (sm_8*) architecture",
                    loc=op.loc
                )
