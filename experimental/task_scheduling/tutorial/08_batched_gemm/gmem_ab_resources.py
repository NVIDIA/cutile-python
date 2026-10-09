# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GMEM coordinate resources for BatchedGemm operands A and B.

The head publishes the expert and local token limit once per work tile. CUDA
Lang passes those values explicitly to the loop instead of mutating resources.
"""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts


def nonnegative_div(value, divisor: int):
    if divisor > 0 and divisor & (divisor - 1) == 0:
        return value >> cl.int32(divisor.bit_length() - 1)
    return value // cl.int32(divisor)


def nonnegative_mod(value, divisor: int):
    if divisor > 0 and divisor & (divisor - 1) == 0:
        return value & cl.int32(divisor - 1)
    return value % cl.int32(divisor)


def _local_tile_limit(raw_limit, token_tile, tile_rows):
    local_limit = raw_limit - token_tile * cl.int32(tile_rows)
    # These conditionals are traced device control flow, not Python min/max.
    if local_limit < cl.int32(0):  # noqa: PLR1730
        local_limit = cl.int32(0)
    if local_limit > cl.int32(tile_rows):  # noqa: PLR1730
        local_limit = cl.int32(tile_rows)
    return local_limit


def _load_tile_metadata(stage_info, cfg):
    tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
    token_tile = tile_coord_n if cfg.is_swap_ab else tile_coord_m
    token_rows = cfg.tile_n if cfg.is_swap_ab else cfg.tile_m
    tasks_inputs = stage_info.context.tasks_inputs
    tile_expert_idx = tasks_inputs.tile_idx_view[token_tile]
    tile_mn_limit = _local_tile_limit(
        tasks_inputs.mn_limit_view[token_tile], token_tile, token_rows
    )
    return tile_expert_idx, tile_mn_limit


@dataclass(kw_only=True, eq=False)
class GmemAResource(ts.MemoryResource):
    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_coords_state(stage_info):
        pass

    @ts.consumer_work(outputs=5)
    @staticmethod
    def compute_a_coords_head(stage_info, cfg):
        expert_idx, mn_limit = _load_tile_metadata(stage_info, cfg)
        tile_coord_m, _, _ = stage_info.work_tile.tile_idx
        coord_a_k = cl.int32(0)
        coord_a_mn = tile_coord_m * cl.int32(cfg.tile_m)
        coord_a_l = expert_idx if cfg.is_swap_ab else cl.int32(0)
        return coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit

    @ts.consumer_work(outputs=5)
    @staticmethod
    def compute_a_coords_loop(
        stage_info, coord_a_mn, coord_a_l, expert_idx, mn_limit, cfg
    ):
        coord_a_k = stage_info.loop_offset * cl.int32(cfg.tile_k)
        return coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit


@dataclass(kw_only=True, eq=False)
class GmemBResource(ts.MemoryResource):
    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_coords_state(stage_info):
        pass

    @ts.consumer_work(outputs=4)
    @staticmethod
    def compute_b_coords_head(stage_info, cfg):
        expert_idx, mn_limit = _load_tile_metadata(stage_info, cfg)
        _, tile_coord_n, _ = stage_info.work_tile.tile_idx
        coord_b_k = cl.int32(0)
        coord_b_mn = tile_coord_n * cl.int32(cfg.tile_n)
        coord_b_l = cl.int32(0) if cfg.is_swap_ab else expert_idx
        return coord_b_k, coord_b_mn, coord_b_l, mn_limit

    @ts.consumer_work(outputs=4)
    @staticmethod
    def compute_b_coords_loop(stage_info, coord_b_mn, coord_b_l, mn_limit, cfg):
        coord_b_k = stage_info.loop_offset * cl.int32(cfg.tile_k)
        return coord_b_k, coord_b_mn, coord_b_l, mn_limit

    @ts.consumer_work(outputs=4)
    @staticmethod
    def compute_b_coords_prefetch(
        stage_info, coord_b_mn, coord_b_l, mn_limit, prefetch_idx, cfg
    ):
        return cl.int32(prefetch_idx * cfg.tile_k), coord_b_mn, coord_b_l, mn_limit
