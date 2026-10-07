# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Q/K/V TMA staging and the page-table producer resource."""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts

from .helpers_common import (
    kv_tile_idx, q_row_token_and_local_head, smem_array, smem_descriptor,
    transform_ragged_coords, work_coords,
)


@dataclass(kw_only=True, eq=False)
class SmemQResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, offset, cfg):
        q_group, kv_head, batch = work_coords(stage_info)
        inputs = stage_info.context.tasks_inputs
        q_smem = smem_array(stage_info, offset, cfg.q_dtype, cfg.q_stages * 1024)
        if cfg.use_variable_seqlens_q:
            token, head = q_row_token_and_local_head(q_group, cl.int32(0), cfg)
        if cl.elect_sync():
            for chunk in cl.static_iter(range(2)):
                coords = (chunk * 64, q_group * 8, kv_head, 0, batch)
                if cfg.use_variable_seqlens_q:
                    coords = transform_ragged_coords(
                        (chunk * 64, kv_head * cfg.heads_q_per_kv + head,
                         inputs.q_token_offset + token),
                        cfg.q_tokens_per_cta, inputs.seq_len_q - token,
                    )
                cl.copy_async_bulk_tensor_global_to_shared(
                    inputs.tma_q_desc,
                    coords,
                    q_smem.pointer(stage_info.stage_idx * 1024 + chunk * 512),
                    stage_info.barrier,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def q_desc(stage_info, offset, cfg):
        q_smem = smem_array(stage_info, offset, cfg.q_dtype, cfg.q_stages * 1024)
        return smem_descriptor(q_smem.pointer(stage_info.stage_idx * 1024), 1024)


@dataclass(kw_only=True, eq=False)
class SmemPageOffsetsKvResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def load_page_offsets(stage_info, offset, inst_idx, section, is_v, cfg):
        inputs = stage_info.context.tasks_inputs
        tile = kv_tile_idx(stage_info, inst_idx, section, is_v, cfg)
        if not cfg.use_variable_seqlens_q:
            tile = cl.minimum(tile, (inputs.seq_len - 1) // 128)
        fragments = 128 // cfg.num_tokens_per_page
        logical_page = tile * fragments
        window = logical_page // 32 * 32
        if cfg.use_variable_seqlens_q:
            window = (logical_page >> 5) << 5
        lane = cl.thread_index(0) % 32
        begin = inputs.page_begin
        count = inputs.page_count
        page_idx = cl.minimum(window + lane, count - 1)
        page_id = inputs.paged_kv_indices[begin + page_idx]
        cache = smem_array(stage_info, offset, cl.int32, cfg.page_offsets_stages * 32)
        cache[stage_info.stage_idx * 32 + lane] = page_id

    @ts.consumer_work(outputs=1)
    @staticmethod
    def page_ids(stage_info, offset, inst_idx, section, is_v, cfg):
        inputs = stage_info.context.tasks_inputs
        tile = kv_tile_idx(stage_info, inst_idx, section, is_v, cfg)
        if not cfg.use_variable_seqlens_q:
            tile = cl.minimum(tile, (inputs.seq_len - 1) // 128)
        fragments = 128 // cfg.num_tokens_per_page
        cache = smem_array(stage_info, offset, cl.int32, cfg.page_offsets_stages * 32)
        begin = stage_info.stage_idx * 32 + (tile * fragments) % 32
        if cfg.use_variable_seqlens_q:
            begin = stage_info.stage_idx * 32 + ((tile * fragments) & 31)
        return cl.Vector(*tuple(cache[begin + i] for i in cl.static_iter(range(fragments))))


@dataclass(kw_only=True, eq=False)
class SmemKvResource(ts.MemoryResource):
    @ts.producer_work
    @staticmethod
    def tma_load(stage_info, page_ids, offset, is_v, cfg):
        _, kv_head, _ = work_coords(stage_info)
        inputs = stage_info.context.tasks_inputs
        desc = inputs.tma_v_desc if is_v else inputs.tma_k_desc
        smem = smem_array(stage_info, offset, cfg.kv_dtype, cfg.kv_stages * 16384)
        if cl.elect_sync():
            for fragment in cl.static_iter(range(128 // cfg.num_tokens_per_page)):
                for chunk in cl.static_iter(range(2)):
                    cl.copy_async_bulk_tensor_global_to_shared(
                        desc,
                        (chunk * 64, 0, kv_head, page_ids[fragment]),
                        smem.pointer(
                            stage_info.stage_idx * 16384
                            + chunk * 8192
                            + fragment * cfg.num_tokens_per_page * 64
                        ),
                        stage_info.barrier,
                    )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def kv_desc(stage_info, offset, cfg):
        smem = smem_array(stage_info, offset, cfg.kv_dtype, cfg.kv_stages * 16384)
        return smem_descriptor(smem.pointer(stage_info.stage_idx * 16384), 16384)
