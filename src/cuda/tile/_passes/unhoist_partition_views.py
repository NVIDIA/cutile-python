# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from cuda.tile._ir.ir import Block, Mapper, Operation
from cuda.tile._ir.ops import MakePartitionView, TileLoad, TileStore, NumTiles


def unhoist_partition_views(root_block: Block):
    def_info: dict[str, tuple[Operation, Block]] = {}
    _unhoist(root_block, def_info)


def _unhoist(block: Block, def_info: dict[str, tuple[Operation, Block]]):
    new_block = block.empty_like_self()
    for op in block:
        if isinstance(op, (TileLoad, TileStore, NumTiles)):
            view_def, def_block = def_info[op.view.name]
            if isinstance(view_def, MakePartitionView) and def_block is not block:
                mapper = Mapper(block.ctx)
                new_block.append(view_def.clone(mapper))
                op = dataclasses.replace(op, view=mapper.get_var(op.view))

        for nested in op.nested_blocks:
            _unhoist(nested, def_info)

        new_block.append(op)
        for v in op.result_vars:
            def_info[v.name] = (op, block)

    block[:] = new_block.detach_all()
