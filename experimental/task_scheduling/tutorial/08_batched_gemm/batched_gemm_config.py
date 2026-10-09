# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the supported BF16 BatchedGemm CUDA Lang paths.

Supports one-CTA expert GEMM, static/CLC scheduling, BF16/FP16 output, and
ordinary SwiGLU in either operand orientation. Activations are pre-expanded;
output stores are direct global stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, make_dataclass
from enum import IntEnum

import cuda.lang as cl


class BatchMode(IntEnum):
    BATCH_N = 0
    BATCH_M = 1


class TileScheduler(IntEnum):
    STATIC = 0
    PERSISTENT = 1


class ActKind(IntEnum):
    NONE = 0
    SWIGLU = 1


class DType(IntEnum):
    FP32 = 1
    FP16 = 2
    BF16 = 3


DTYPE_BITS = {
    int(DType.FP32): 32,
    int(DType.FP16): 16,
    int(DType.BF16): 16,
}


def dtype_bits(dtype_kind: object) -> int:
    """Return the logical bit width for a supported DType value."""
    kind = int(dtype_kind)
    if kind not in DTYPE_BITS:
        raise ValueError(f"Unsupported dtype kind: {kind}")
    return DTYPE_BITS[kind]


@dataclass(kw_only=True)
class BatchedGemmConfig:
    """Static policy for the BF16 expert GEMM and ordinary SwiGLU kernels.

    Plain integer fields retain the source's names. BF16 operands, FP32
    accumulation, and a one-CTA cluster are the only supported MMA path.
    Warp indices/counts and threads_per_cta are derived by compute_warp_layout;
    they are not constructor options. Defaults describe a supported FC2 tile.
    """

    # Shape and dtype. Tile N also selects the MMA N extent.
    cluster_m: int = 1
    epi_tile_m: int = 128
    epi_tile_n: int = 8
    mma_k: int = 16
    mma_m: int = 128
    mma_n: int = 8
    tile_k: int = 64
    tile_m: int = 128
    tile_n: int = 8
    dtype_a: int = int(DType.BF16)
    dtype_acc: int = int(DType.FP32)
    dtype_b: int = int(DType.BF16)
    dtype_c: int = int(DType.BF16)

    # Pipeline depths. C scratch remains reserved to match the source footprint,
    # including when the epilogue uses direct global stores.
    num_stages_a: int = 5
    num_stages_b: int = 5
    num_stages_c_smem: int = 2
    num_stages_tmem_acc: int = 2
    num_stages_workid: int = 3

    # Per-thread register budgets; zero A/B overrides select load_regs.
    epilogue_regs: int = 160
    load_a_regs: int = 0
    load_b_regs: int = 0
    load_regs: int = 48
    mma_regs: int = 48
    padding_regs: int = 48
    workid_regs: int = 48

    # BATCH_M: expanded activations A, expert weights B, row-major output.
    # BATCH_N: expert weights A, expanded activations B, M-major SwiGLU output.
    act_kind: int = int(ActKind.NONE)
    batch_mode: int = int(BatchMode.BATCH_M)
    tile_scheduler: int = int(TileScheduler.STATIC)
    transpose_mma_output: int = 0
    use_unroll_loop_2x_for_mma: int = 0

    # Derived task ownership, in the source order: epilogue, MMA, A, B, work-id.
    epilogue_warp_idx: int = field(default=0, init=False)
    load_a_warp_idx: int = field(default=5, init=False)
    load_b_warp_idx: int = field(default=6, init=False)
    mma_warp_idx: int = field(default=4, init=False)
    padding_warp_idx: int = field(default=7, init=False)
    workid_warp_idx: int = field(default=7, init=False)
    num_epilogue_warps: int = field(default=4, init=False)
    num_load_a_warps: int = field(default=1, init=False)
    num_load_b_warps: int = field(default=1, init=False)
    num_mma_warps: int = field(default=1, init=False)
    num_padding_warps: int = field(default=1, init=False)
    num_workid_warps: int = field(default=0, init=False)
    threads_per_cta: int = field(default=256, init=False)

    def __post_init__(self) -> None:
        if self.dtype_acc != int(DType.FP32):
            raise ValueError(f"BatchedGemm accumulators must be FP32, got {self.dtype_acc}")

    @property
    def dtype_a_bits(self) -> int:
        return dtype_bits(self.dtype_a)

    @property
    def dtype_b_bits(self) -> int:
        return dtype_bits(self.dtype_b)

    @property
    def dtype_c_bits(self) -> int:
        return dtype_bits(self.dtype_c)

    @property
    def dtype_a_smem_bits(self) -> int:
        return self.dtype_a_bits

    @property
    def num_bytes_a_per_stage(self) -> int:
        return self.tile_m * self.tile_k * self.dtype_a_smem_bits // 8

    @property
    def num_bytes_b_smem_per_stage(self) -> int:
        return self.tile_n * self.tile_k * self.dtype_b_bits // 8

    @property
    def num_bytes_a_tma_per_stage(self) -> int:
        return self.num_bytes_a_per_stage

    @property
    def num_bytes_b_tma_per_stage(self) -> int:
        return self.num_bytes_b_smem_per_stage

    @property
    def num_bytes_c_smem_scratch(self) -> int:
        return self.epi_tile_m * self.epi_tile_n * self.dtype_c_bits // 8 * self.num_stages_c_smem

    @property
    def tmem_c_cols_per_stage(self) -> int:
        return self.tile_n

    @property
    def tmem_required_cols(self) -> int:
        return self.tmem_c_cols_per_stage * self.num_stages_tmem_acc

    @property
    def has_cluster(self) -> bool:
        return self.cluster_m > 1

    @property
    def split_b_across_ctas(self) -> bool:
        return self.has_cluster

    @property
    def cta_group(self):
        return cl.CTAGroup.CTA_1

    @property
    def is_persistent(self) -> bool:
        return self.tile_scheduler == int(TileScheduler.PERSISTENT)

    @property
    def is_swap_ab(self) -> bool:
        return self.batch_mode == int(BatchMode.BATCH_N)

    @property
    def load_a_task_regs(self) -> int:
        return self.load_a_regs if self.load_a_regs > 0 else self.load_regs

    @property
    def load_b_task_regs(self) -> int:
        return self.load_b_regs if self.load_b_regs > 0 else self.load_regs


@dataclass(frozen=True)
class TaskShape:
    name: str
    count_field: str
    index_field: str


@dataclass(frozen=True)
class WarpLayout:
    task_order: tuple[str, ...]


_WARP_TASKS = {
    task.name: task
    for task in (
        TaskShape("epilogue", "num_epilogue_warps", "epilogue_warp_idx"),
        TaskShape("load_a", "num_load_a_warps", "load_a_warp_idx"),
        TaskShape("load_b", "num_load_b_warps", "load_b_warp_idx"),
        TaskShape("mma", "num_mma_warps", "mma_warp_idx"),
        TaskShape("workid", "num_workid_warps", "workid_warp_idx"),
    )
}


def _pack_warp_layout(cfg: BatchedGemmConfig, layout: WarpLayout) -> None:
    warp_idx = 0
    for task_name in layout.task_order:
        task = _WARP_TASKS[task_name]
        setattr(cfg, task.index_field, warp_idx)
        warp_idx += int(getattr(cfg, task.count_field))
    total_warps = ((warp_idx + 3) // 4) * 4
    cfg.padding_warp_idx = warp_idx
    cfg.num_padding_warps = total_warps - warp_idx
    cfg.threads_per_cta = total_warps * 32


def compute_warp_layout(cfg: BatchedGemmConfig) -> None:
    """Assign BF16 tasks in order: epilogue, MMA, A/B loads, then work-id."""
    cfg.num_epilogue_warps = 4
    cfg.num_mma_warps = 1
    cfg.num_load_a_warps = 1
    cfg.num_load_b_warps = 1
    cfg.num_workid_warps = 1 if cfg.is_persistent else 0
    _pack_warp_layout(cfg, WarpLayout(("epilogue", "mma", "load_a", "load_b", "workid")))


def _validate_binary_config_option(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value}")


def validate_config(
    cfg: BatchedGemmConfig,
    problem_mnk: tuple[int, int, int] | None = None,
) -> None:
    """Reject unsupported dtype/layout/shape policies before schedule capture."""
    if cfg.dtype_a != int(DType.BF16) or cfg.dtype_b != int(DType.BF16):
        raise ValueError("BatchedGemm requires BF16 A and B operands")
    if cfg.dtype_acc != int(DType.FP32):
        raise ValueError("BatchedGemm accumulators must be FP32")
    if cfg.dtype_c not in (int(DType.BF16), int(DType.FP16)):
        raise ValueError("dtype_c must be BF16 or FP16")
    if cfg.batch_mode not in (int(BatchMode.BATCH_N), int(BatchMode.BATCH_M)):
        raise ValueError(f"batch_mode must be BATCH_N or BATCH_M, got {cfg.batch_mode}")
    if cfg.act_kind not in (int(ActKind.NONE), int(ActKind.SWIGLU)):
        raise ValueError(f"act_kind must be NONE or SWIGLU, got {cfg.act_kind}")
    if cfg.is_swap_ab and cfg.act_kind != int(ActKind.SWIGLU):
        raise ValueError("swap-AB requires SwiGLU")
    _validate_binary_config_option("transpose_mma_output", cfg.transpose_mma_output)
    if cfg.transpose_mma_output != int(cfg.is_swap_ab):
        raise ValueError("transpose_mma_output must match the batch_mode-selected swap_ab path")
    if cfg.tile_scheduler not in (int(TileScheduler.STATIC), int(TileScheduler.PERSISTENT)):
        raise ValueError(f"tile_scheduler must be STATIC or PERSISTENT, got {cfg.tile_scheduler}")
    if cfg.cluster_m != 1:
        raise ValueError("Only cluster_m=1 is supported")
    if cfg.tile_m != 128 or cfg.mma_m != cfg.tile_m:
        raise ValueError("BF16 BatchedGemm requires tile_m=mma_m=128")
    if cfg.tile_n not in (8, 16, 32, 64, 128, 192, 256) or cfg.mma_n != cfg.tile_n:
        raise ValueError("tile_n=mma_n must be 8/16/32/64/128/192/256")
    if cfg.mma_k != 16:
        raise ValueError("BF16 MMA requires mma_k=16")
    if cfg.tile_k < 64 or cfg.tile_k % 64:
        raise ValueError("BF16 K-box TMA requires tile_k to be a positive multiple of 64")
    if cfg.epi_tile_m <= 0 or (cfg.is_swap_ab and cfg.epi_tile_m != 128):
        raise ValueError("epi_tile_m must be positive; swap-AB requires epi_tile_m=128")
    if cfg.epi_tile_n < 8 or cfg.epi_tile_n % 8:
        raise ValueError("epi_tile_n must be a positive multiple of 8")
    if cfg.tile_n % cfg.epi_tile_n:
        raise ValueError("epi_tile_n must divide tile_n")
    t2r_repx = cfg.epi_tile_n // (8 if cfg.is_swap_ab else 4)
    if t2r_repx & (t2r_repx - 1):
        raise ValueError("epi_tile_n defines non-power-of-two TMEM fragments")
    _validate_binary_config_option("use_unroll_loop_2x_for_mma", cfg.use_unroll_loop_2x_for_mma)
    stage_fields = ["num_stages_a", "num_stages_b", "num_stages_c_smem", "num_stages_tmem_acc"]
    if cfg.is_persistent:
        stage_fields.append("num_stages_workid")
    for name in stage_fields:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive, got {getattr(cfg, name)}")
    if cfg.tmem_required_cols > 512:
        raise ValueError("TMEM footprint exceeds the 512-column tcgen05 allocation limit")
    if problem_mnk is not None:
        m, n, k = problem_mnk
        if m <= 0 or m % cfg.tile_m or n <= 0 or n % cfg.tile_n or k <= 0 or k % cfg.tile_k:
            raise ValueError("BatchedGemm requires positive tile-aligned M, N, K")


def _normalize_config_overrides(overrides: dict) -> dict:
    valid_fields = {item.name for item in fields(BatchedGemmConfig) if item.init}
    unknown = sorted(key for key in overrides if key not in valid_fields)
    if unknown:
        raise TypeError(f"Unknown BatchedGemmConfig option(s): {', '.join(unknown)}")
    return dict(overrides)


def make_config(**overrides) -> BatchedGemmConfig:
    """Construct a config using only supported, caller-controlled options."""
    return BatchedGemmConfig(**_normalize_config_overrides(overrides))


# Keep source property names at the host/device boundary. Resource sizes and
# CTA-group values are resolved once, before device work is compiled.
_DEVICE_DERIVED_FIELDS = (
    "num_bytes_a_per_stage",
    "num_bytes_b_smem_per_stage",
    "dtype_a_smem_bits",
    "tmem_c_cols_per_stage",
    "has_cluster",
    "split_b_across_ctas",
    "cta_group",
)
DeviceBatchedGemmConfig = make_dataclass(
    "DeviceBatchedGemmConfig",
    [(item.name, int) for item in fields(BatchedGemmConfig)]
    + [(name, object) for name in _DEVICE_DERIVED_FIELDS],
    namespace={
        "__module__": __name__,
        **{
            name: value
            for name, value in vars(BatchedGemmConfig).items()
            if name not in _DEVICE_DERIVED_FIELDS
            and (
                isinstance(value, property)
                or (callable(value) and not name.startswith("__"))
            )
        },
    },
    frozen=True,
    kw_only=True,
)


def freeze_config(cfg: BatchedGemmConfig):
    """Snapshot an assembled host config without retaining mutable device state."""
    return DeviceBatchedGemmConfig(
        **{item.name: getattr(cfg, item.name) for item in fields(cfg)},
        **{name: getattr(cfg, name) for name in _DEVICE_DERIVED_FIELDS},
    )
