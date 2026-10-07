# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for the D128, Q8 SwapsMmaAb decode port."""

from dataclasses import dataclass, fields

import cuda.lang as cl

from cuda.tile import DType


@dataclass(frozen=True)
class FmhaDecodeConfig:
    headdim: int = 128
    tile_size_q: int = 8
    tile_size_kv: int = 128
    num_insts_kv: int = 2
    q_dtype: DType = cl.float16
    kv_dtype: DType = cl.float16
    out_dtype: DType = cl.float16
    q_stages: int = 2
    kv_stages: int = 4
    o_stages: int = 2
    page_offsets_stages: int = 6
    num_tokens_per_page: int = 32
    softmax0_warp_idx: int = 0
    softmax1_warp_idx: int = 4
    correction_warp_idx: int = 8
    mma_warp_idx: int = 12
    load_warp_idx: int = 13
    scheduler_warp_idx: int = 13
    page_offsets_warp_idx: int = 14
    clc_load_warp_idx: int = 15
    threads_per_cta: int = 512
    tmem_s_cols: int = 8
    tmem_stats_cols: int = 32
    tmem_o_cols: int = 8
    tmem_alloc_cols: int = 128
    window_left: int = -1
    mask_type: str = "dense"
    use_persistent_scheduler: bool = False
    groups_tokens_heads_q: bool = False
    use_variable_seqlens_q: bool = False
    max_seq_len_q: int = 1
    heads_q_per_kv: int = 0

    @property
    def q_tokens_per_cta(self):
        return self.tile_size_q // self.heads_q_per_kv if self.groups_tokens_heads_q else 1

    def num_q_ctas(self, head_ratio):
        if self.groups_tokens_heads_q:
            tokens = self.tile_size_q // head_ratio
            return (self.max_seq_len_q + tokens - 1) // tokens
        return self.max_seq_len_q * ((head_ratio + self.tile_size_q - 1) // self.tile_size_q)


def validate_config(cfg):
    defaults = FmhaDecodeConfig()
    for field in fields(cfg):
        if field.name not in (
            "num_tokens_per_page", "mask_type", "window_left",
            "use_persistent_scheduler", "groups_tokens_heads_q",
            "q_dtype", "kv_dtype", "out_dtype",
            "use_variable_seqlens_q", "max_seq_len_q", "heads_q_per_kv",
        ):
            if getattr(cfg, field.name) != getattr(defaults, field.name):
                raise ValueError(
                    f"{field.name} is fixed at {getattr(defaults, field.name)} "
                    "in this specialization"
                )
    if (cfg.headdim, cfg.tile_size_q, cfg.tile_size_kv, cfg.num_insts_kv) != (128, 8, 128, 2):
        raise ValueError("This port currently implements D128/Q8/KV128 with two K/V instances")
    if cfg.q_dtype not in (cl.float16, cl.bfloat16):
        raise ValueError("Q/K/V/O dtype must be float16 or bfloat16")
    if not cfg.q_dtype == cfg.kv_dtype == cfg.out_dtype:
        raise ValueError("Q, K, V, and O must use the same dtype")
    if cfg.num_tokens_per_page not in (16, 32, 64, 128):
        raise ValueError("page size must be 16, 32, 64, or 128")
    if cfg.mask_type not in ("dense", "causal"):
        raise ValueError("mask_type must be dense or causal")
    if cfg.window_left < -1 or (cfg.window_left >= 0 and cfg.mask_type != "causal"):
        raise ValueError("window_left requires causal attention and must be >= -1")
    if not isinstance(cfg.max_seq_len_q, int) or cfg.max_seq_len_q < 1:
        raise ValueError("max_seq_len_q must be a positive integer")
    if not cfg.use_variable_seqlens_q and cfg.max_seq_len_q != 1:
        raise ValueError("multiple query tokens currently require packed Q and qo_indptr")
    if not 0 <= cfg.heads_q_per_kv <= 32:
        raise ValueError("heads_q_per_kv must be 0 (inferred) or an integer in [1,32]")
