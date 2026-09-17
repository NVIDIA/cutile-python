# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Resources for the query/head-paired D128 FMHA specializations.

This layer implements fixed-length or packed-ragged FP16 attention whose
structural axes are:

* SM100 direct, static-persistent, or CLC dynamic-persistent scheduling;
* query-paired or head-paired work tiles;
* dense, bottom-right causal, or causal left-window masking;
* two Q/K/V instances sharing one 128-column K/V tile; and
* D=128 CTA-group-1 tcgen05 QK and PV MMAs.
"""

from dataclasses import dataclass
import cuda.lang as cl

import task_scheduling as ts

try:
    from .stage import FmhaStage
except ImportError:
    from stage import FmhaStage


WARP_SIZE = 32
THREADS_PER_CTA = 16 * WARP_SIZE

SOFTMAX0_WARPS = (0, 1, 2, 3)
SOFTMAX1_WARPS = (4, 5, 6, 7)
CORRECTION_WARPS = (8, 9, 10, 11)
MMA_WARP = 12
LOAD_WARP = 13
EPILOGUE_WARP = 14
SCHEDULER_WARP = 15

Q_TILE_M = 128
KV_TILE_N = 128
HEAD_DIM = 128
ATTENTION_SCALE_LOG2 = 1.0 / (HEAD_DIM**0.5) * 1.4426950408889634
WORK_TILE_Q = 2 * Q_TILE_M
MMA_K = 16

Q_STAGES = 2
KV_STAGES = 3
O_STAGES = 2
STATS_STAGES = 1
SEQUENCE_STAGES = 1
SCHEDULER_STAGES = 1
SUPPORTED_SCHEDULERS = (
    "direct",
    "static_persistent",
    "clc_dynamic_persistent",
)
SUPPORTED_PAGE_SIZES = (16, 32, 64, 128)

Q_STAGE_ELEMENTS = Q_TILE_M * HEAD_DIM
KV_STAGE_ELEMENTS = KV_TILE_N * HEAD_DIM
O_STAGE_ELEMENTS = Q_TILE_M * HEAD_DIM
Q_STAGE_BYTES = Q_STAGE_ELEMENTS * 2
KV_STAGE_BYTES = KV_STAGE_ELEMENTS * 2
O_STAGE_BYTES = O_STAGE_ELEMENTS * 2
STATS_ROWS = 4 * WARP_SIZE
STATS_ELEMENTS = STATS_ROWS * 2
STATS_BYTES = STATS_ELEMENTS * 4

TMA_SLICES = 2
TMA_GRANULARITY = HEAD_DIM // TMA_SLICES
SMEM_ALIGNMENT = 128
# A 128B-swizzled FP16 tcgen05 atom spans eight 64-element row segments.
TCGEN05_SMEM_STRIDE_BYTES = 8 * TMA_GRANULARITY * 2

TMEM_COLUMNS = 512
TMEM_S0 = 0
TMEM_S1 = 128
TMEM_P0 = 32
TMEM_P1 = 160
TMEM_O0 = 256
TMEM_O1 = 384

SOFTMAX_REGISTERS = 176
CORRECTION_REGISTERS = 80
AUXILIARY_REGISTERS = 80

CLC_RESPONSE_BYTES = cl.cluster_launch_control_token.bitwidth // 8
CLUSTER_SHAPE = (1, 1, 1)
TMEM_SYNC_BARRIER = 2
TMEM_SYNC_THREADS = (len(CORRECTION_WARPS) + 1) * WARP_SIZE
RAGGED_LARGE_N = 1 << 30
RAGGED_XLARGE_N = 1 << 35
RAGGED_TMA_DIM_MAX = 1 << 31


# ---------------------------------------------------------------------------
# FmhaConfig -- kernel-wide configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaConfig:
    """Compile-time contract for the paired D128 specialization."""

    # Data types
    q_dtype: str = "fp16"
    k_dtype: str = "fp16"
    v_dtype: str = "fp16"
    o_dtype: str = "fp16"
    qk_acc_dtype: str = "fp32"
    pv_acc_dtype: str = "fp32"

    # Tile shapes
    qk_mma_tiler: tuple[int, int, int] = (Q_TILE_M, KV_TILE_N, HEAD_DIM)
    pv_mma_tiler: tuple[int, int, int] = (Q_TILE_M, HEAD_DIM, KV_TILE_N)
    epi_tile: tuple[int, int] = (Q_TILE_M, HEAD_DIM)
    num_qkv_instances: int = 2

    # Pipeline stages
    q_stage: int = Q_STAGES
    kv_stage: int = KV_STAGES
    mma_softmax_stage: int = 1
    has_tmem_p_pipeline: bool = False
    stats_via_smem: bool = True
    stage_scoped_tmem_stats: bool = False
    softmax_corr_stage: int = STATS_STAGES
    mma_corr_stage: int = O_STAGES

    # TMA copy granularity
    tma_copy_qkv_iters: int = TMA_SLICES
    tma_copy_q_granu_inner: int = TMA_GRANULARITY
    tma_copy_q_elements: int = Q_STAGE_ELEMENTS
    tma_copy_q_granu_elems: int = Q_STAGE_ELEMENTS // TMA_SLICES
    tma_copy_q_bytes: int = Q_STAGE_BYTES
    tma_copy_kv_granu_inner: int = TMA_GRANULARITY
    tma_copy_kv_elements: int = KV_STAGE_ELEMENTS
    tma_copy_kv_stage_iters: int = TMA_SLICES
    tma_copy_kv_granu_elems: int = KV_STAGE_ELEMENTS // TMA_SLICES
    tma_copy_kv_bytes: int = KV_STAGE_BYTES
    tma_copy_o_iters: int = TMA_SLICES
    tma_copy_o_granu_inner: int = TMA_GRANULARITY
    tma_copy_o_elements: int = O_STAGE_ELEMENTS
    tma_copy_o_stage_iters: int = TMA_SLICES
    tma_copy_o_granu_elems: int = O_STAGE_ELEMENTS // TMA_SLICES
    q_tile_m: int = Q_TILE_M
    kv_tile_n: int = KV_TILE_N

    # Warp assignments
    softmax0_warp_ids: tuple[int, ...] = SOFTMAX0_WARPS
    softmax1_warp_ids: tuple[int, ...] = SOFTMAX1_WARPS
    correction_warp_ids: tuple[int, ...] = CORRECTION_WARPS
    mma_warp_id: int = MMA_WARP
    load_warp_id: int = LOAD_WARP
    epilogue_warp_id: int = EPILOGUE_WARP
    empty_warp_id: int = SCHEDULER_WARP

    # Register budgets
    num_regs_softmax: int = SOFTMAX_REGISTERS
    num_regs_correction: int = CORRECTION_REGISTERS
    num_regs_other: int = AUXILIARY_REGISTERS

    # TMEM layout
    tmem_alloc_cols: int = TMEM_COLUMNS
    tmem_stats_cols: int = 4
    tmem_s0_offset: int = TMEM_S0
    tmem_s1_offset: int = TMEM_S1
    tmem_o0_offset: int = TMEM_O0
    tmem_o1_offset: int = TMEM_O1
    tmem_p0_offset: int = TMEM_P0
    tmem_p1_offset: int = TMEM_P1
    tmem_vec0_offset: int = 0
    tmem_vec1_offset: int = 128

    # SMEM shapes
    sO_stage_elements: int = O_STAGE_ELEMENTS
    sQ_shape: tuple[int, int] = (Q_STAGES, Q_STAGE_ELEMENTS)
    sK_shape: tuple[int, int] = (KV_STAGES, KV_STAGE_ELEMENTS)

    # Miscellaneous launch configuration
    buffer_align_bytes: int = 1024
    tmem_bar_id: int = TMEM_SYNC_BARRIER
    cluster_shape_mn: tuple[int, int] = (1, 1)
    block_warps: int = 16

    # Scheduler and attention mode
    scheduler: str = "clc_dynamic_persistent"
    h_r: int = 1
    is_causal: bool = False
    balance_causal_workload: bool = False
    num_seq_tiles: int = 0
    enable_skip_correction: bool = True
    has_varlen: bool = False
    has_uniform_varlen: bool = False
    uniform_seq_len_q: int = 0
    uniform_seq_len_k: int = 0
    head_paired: bool = False
    enable_early_tile_sum: bool = True
    seq_tile_n: int = KV_TILE_N
    tmem_x_load_s: int = 32
    has_q_offset: bool = False
    causal_single_kv_tile: bool = False
    window_size_left: int = 0
    fixed_dense_k_tail: int = 0
    packed_dense_k_mask: bool = True
    use_paged_kv: bool = False
    fuse_epilogue_into_correction: bool = False
    num_tokens_per_page: int = 32
    max_num_pages_per_seq_kv: int = 1
    page_offsets_num_warps: int = 1
    page_table_window_entries: int = 32

    @property
    def is_persistent(self) -> bool:
        return self.scheduler != "direct"

    @property
    def is_clc_dynamic(self) -> bool:
        return self.scheduler == "clc_dynamic_persistent"

    @property
    def single_qkv_instance(self) -> bool:
        return self.num_qkv_instances == 1

    @property
    def work_tile_q_heads(self) -> int:
        return 2 if self.head_paired else 1

    @property
    def work_tile_q_seq_tiles(self) -> int:
        return 1 if self.head_paired else self.num_qkv_instances

    @property
    def peer_q_head_stride(self) -> int:
        return 1 if self.head_paired else 0

    @property
    def peer_q_seq_tile_stride(self) -> int:
        return 0 if self.head_paired else 1

    @property
    def gmem_o_store_wait_after_write(self) -> bool:
        return self.head_paired

    @property
    def uses_causal_reversed_head_batch_seq_tile_order(self) -> bool:
        return self.is_causal and self.balance_causal_workload

    @property
    def uses_head_batch_seq_tile_order(self) -> bool:
        return self.uses_causal_reversed_head_batch_seq_tile_order

    @property
    def work_tile_coord_indices(self) -> tuple[int, int, int]:
        return (2, 0, 1) if self.uses_head_batch_seq_tile_order else (0, 1, 2)

    @property
    def has_tile_aligned_uniform_q_offset(self) -> bool:
        return (
            self.has_q_offset
            and self.has_uniform_varlen
            and self.uniform_seq_len_q % self.cta_tiler[0] == 0
            and (self.uniform_seq_len_k - self.uniform_seq_len_q) % self.kv_tile_n == 0
        )

    @property
    def skip_causal_invalid_peer0(self) -> bool:
        if (
            not self.is_causal
            or self.head_paired
            or (self.has_q_offset and not self.has_tile_aligned_uniform_q_offset)
            or self.causal_single_kv_tile
        ):
            return False
        peer0_tiles = (self.q_tile_m + self.kv_tile_n - 1) // self.kv_tile_n
        paired_tiles = (self.cta_tiler[0] + self.kv_tile_n - 1) // self.kv_tile_n
        return paired_tiles > peer0_tiles

    @property
    def needs_window_tail_left_mask(self) -> bool:
        """Return whether a sliding-window tail can cross its left edge."""
        return self.window_size_left > 0 and (
            self.has_varlen
            or self.has_q_offset
            or self.q_tile_m != self.kv_tile_n
            or self.window_size_left < self.q_tile_m - 1
        )

    @property
    def cta_tiler(self) -> tuple[int, int, int]:
        return (
            self.work_tile_q_seq_tiles * self.qk_mma_tiler[0],
            self.qk_mma_tiler[1],
            self.qk_mma_tiler[2],
        )

    @property
    def tmem_offsets(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.tmem_s0_offset,
            self.tmem_s1_offset,
            self.tmem_p0_offset,
            self.tmem_p1_offset,
            self.tmem_o0_offset,
            self.tmem_o1_offset,
        )

    @property
    def block_threads(self) -> int:
        return self.block_warps * WARP_SIZE

    @property
    def cluster_shape(self) -> tuple[int, int, int]:
        return (*self.cluster_shape_mn, 1)


def _tmem_pointer(base, lane_offset=0, column_offset=0):
    return cl.tcgen05_tmem_offset(
        base,
        lane_offset=lane_offset,
        column_offset=column_offset,
    )


def transform_ragged_coords(
    coords,
    *,
    ragged_dim_idx,
    ragged_box_size,
    ragged_extent,
):
    """Expand rank-3, axis-2 coordinates for ragged TMA."""
    box = cl.int32(ragged_box_size)
    ext = cl.int32(ragged_extent)
    is_neg = cl.int32(ext < cl.int32(0))
    ext = ext - ext * is_neg
    is_over = cl.int32(ext > box)
    ext = ext + (box - ext) * is_over
    ext_mod = ext % box
    dist = box - ext_mod
    d_mod = dist % box
    is_empty = cl.int32(ext == cl.int32(0))
    d_mod = d_mod + box * is_empty
    bal = cl.int32(0) - d_mod
    large_n = cl.int32(RAGGED_LARGE_N)

    orig = cl.int32(coords[ragged_dim_idx])
    return coords[0], coords[1], d_mod, large_n, orig + large_n + bal


def _smem_array(stage_info, smem_offset, dtype, shape):
    pointer = stage_info.context.smem_base.pointer() + smem_offset
    typed_pointer = cl.bitcast(
        pointer,
        cl.pointer_dtype(dtype, pointer.memory_space),
    )
    return cl.Array.from_parts(typed_pointer, shape)


def _qk_descriptor(pointer, fmha_config):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=pointer,
            leading_dimension_byte_offset=0,
            stride_dimension_byte_offset=(8 * fmha_config.tma_copy_q_granu_inner * 2),
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )


def _pv_descriptor(pointer, fmha_config):
    return cl.int64(
        cl.Tcgen05SharedMemoryDescriptor(
            matrix_start_address=pointer,
            leading_dimension_byte_offset=(
                fmha_config.tma_copy_q_bytes // fmha_config.tma_copy_qkv_iters
            ),
            stride_dimension_byte_offset=(8 * fmha_config.tma_copy_kv_granu_inner * 2),
            swizzle_mode=cl.SwizzleMode.SWIZZLE_128B,
        ).encode()
    )


def _qk_instruction_descriptor(fmha_config):
    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.float32,
        a_type=cl.float16,
        b_type=cl.float16,
        n=fmha_config.qk_mma_tiler[1],
        m=fmha_config.qk_mma_tiler[0],
    ).encode()


def _pv_instruction_descriptor(fmha_config):
    return cl.Tcgen05InstructionDescriptor(
        d_type=cl.float32,
        a_type=cl.float16,
        b_type=cl.float16,
        transpose_b=True,
        n=fmha_config.epi_tile[1],
        m=fmha_config.epi_tile[0],
    ).encode()


def _pack_output_values(values, scale, element_count):
    converted = (values[:element_count] * scale).astype(cl.float16)
    return cl.Vector(
        *tuple(
            converted[2 * word:2 * word + 2].reinterpret_as_scalar(cl.uint32)
            for word in cl.static_iter(range(element_count // 2))
        ),
        dtype=cl.uint32,
    )


def _store_output_values(
    o_stage,
    packed_words,
    row,
    column,
    element_count,
    fmha_config,
):
    """Store packed FP16 output values into the D128 two-slice TMA layout."""
    out_ptr = cl.bitcast(
        o_stage,
        cl.pointer_dtype(cl.uint32, cl.MemorySpace.SHARED),
    )
    slice_elements = fmha_config.tma_copy_o_granu_inner
    slice_bytes = slice_elements * 2
    swizzle_mask = (slice_bytes // 16 - 1) << 7
    slice_index = column // slice_elements
    column_in_slice = column % slice_elements
    for group in cl.static_iter(range(element_count // 8)):
        byte_offset = (
            slice_index * fmha_config.q_tile_m * slice_bytes
            + row * slice_bytes
            + column_in_slice * 2
            + group * 16
        )
        swizzled = byte_offset ^ (((byte_offset & swizzle_mask) >> 7) << 4)
        (out_ptr + swizzled // 4).store(
            packed_words[group * 4:group * 4 + 4],
            alignment=16,
        )


def _mask_packed_score_chunk(scores, key_base, seqlen_k):
    valid_in_chunk = cl.minimum(
        cl.maximum(seqlen_k - key_base, cl.int32(0)),
        cl.int32(32),
    )
    return cl.Vector(
        *tuple(
            scores[item]
            if cl.uint32(item) < cl.uint32(valid_in_chunk)
            else cl.float32(-float("inf"))
            for item in cl.static_iter(range(32))
        ),
        dtype=cl.float32,
    )


def _mask_score_chunk(scores, key_base, lower_bound, upper_bound):
    """Keep keys in the inclusive ``[lower_bound, upper_bound]`` interval."""
    left_invalid = cl.minimum(
        cl.maximum(lower_bound - key_base, cl.int32(0)),
        cl.int32(32),
    )
    right_valid = cl.minimum(
        cl.maximum(upper_bound - key_base + 1, cl.int32(0)),
        cl.int32(32),
    )
    return cl.Vector(
        *tuple(
            scores[item]
            if (
                (cl.uint32(item) >= cl.uint32(left_invalid))
                & (cl.uint32(item) < cl.uint32(right_valid))
            )
            else cl.float32(-float("inf"))
            for item in cl.static_iter(range(32))
        ),
        dtype=cl.float32,
    )


def _mask_score_chunk_left(scores, key_base, lower_bound):
    """Keep keys at or to the right of ``lower_bound``."""
    left_invalid = cl.minimum(
        cl.maximum(lower_bound - key_base, cl.int32(0)),
        cl.int32(32),
    )
    return cl.Vector(
        *tuple(
            scores[item]
            if cl.uint32(item) >= cl.uint32(left_invalid)
            else cl.float32(-float("inf"))
            for item in cl.static_iter(range(32))
        ),
        dtype=cl.float32,
    )


def bottom_right_window_tile_start(
    seq_coord,
    q_tile_m,
    kv_tile_n,
    q_offset,
    window_size_left,
):
    """Return the first K/V tile intersecting a bottom-right left window."""
    return cl.maximum(
        cl.int32(0),
        (seq_coord * q_tile_m + q_offset - window_size_left) // kv_tile_n,
    )


def bottom_right_window_max_tiles(q_tile_m, kv_tile_n, window_size_left):
    """Return the offset-independent maximum K/V span for one Q tile."""
    return (window_size_left + q_tile_m + 2 * kv_tile_n - 2) // kv_tile_n


def _reduce_row_max(chunks, row_max, fmha_config):
    row_values = cl.Vector(
        *tuple(
            chunks[chunk_idx][item]
            for chunk_idx in cl.static_iter(
                range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
            )
            for item in cl.static_iter(range(fmha_config.tmem_x_load_s))
        ),
        dtype=cl.float32,
    )
    old_row_max = row_max
    tile_row_max = row_values.reduce(cl.VectorReduction.max)
    row_max = cl.maximum(row_max, tile_row_max)
    if row_max == cl.float32(-float("inf")):
        row_max = cl.float32(0.0)
    return old_row_max, row_max


@dataclass(kw_only=True, eq=False)
class GmemQKVResource(ts.MemoryResource):
    """Global-memory Q/K/V source and per-work-tile coordinates.

    ``compute_coords`` routes coordinates as immutable work outputs.
    Compile-time mode data is passed separately as a static work-method
    argument, outside the loop-carried ``TasksInputs``.
    """

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=9,
    )
    @staticmethod
    def compute_coords(stage_info, fmha_config):
        """Resolve fixed or packed coordinates for downstream Q/K/V loads."""
        seq_idx, head_idx, batch_idx = fmha_config.work_tile_coord_indices
        tile_idx = stage_info.work_tile.tile_idx
        seq_coord = tile_idx[seq_idx]
        head_coord = tile_idx[head_idx]
        batch_coord = tile_idx[batch_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        kv_head_coord = head_coord * fmha_config.work_tile_q_heads // fmha_config.h_r
        seq_coord_q = (
            seq_coord * fmha_config.q_tile_m * fmha_config.work_tile_q_seq_tiles
        )
        cuseqlen_q = cl.int32(0)
        cuseqlen_k = cl.int32(0)
        seqlen_q = cl.int32(0)
        seqlen_k = cl.int32(0)
        q_offset = stage_info.context.tasks_inputs.q_offset
        if fmha_config.has_varlen:
            cum_seqlen_q = stage_info.context.tasks_inputs.cum_seqlen_q
            cum_seqlen_k = stage_info.context.tasks_inputs.cum_seqlen_k
            cuseqlen_q = cum_seqlen_q[batch_coord]
            cuseqlen_k = cum_seqlen_k[batch_coord]
            seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
            seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
            seq_coord_q = cuseqlen_q + seq_coord_q
            q_offset = seqlen_k - seqlen_q
        kv_tile_start = cl.int32(0)
        if fmha_config.window_size_left > 0:
            kv_tile_start = bottom_right_window_tile_start(
                seq_coord,
                fmha_config.q_tile_m,
                fmha_config.kv_tile_n,
                q_offset,
                fmha_config.window_size_left,
            )
        return (
            seq_coord_q,
            head_coord,
            kv_head_coord,
            batch_coord,
            cuseqlen_q,
            cuseqlen_k,
            seqlen_q,
            seqlen_k,
            kv_tile_start,
        )


@dataclass(kw_only=True, eq=False)
class SmemQResource(ts.MemoryResource):
    """SMEM Q tile buffer with a two-stage TMA-to-UMMA pipeline."""

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float16,
            fmha_config.sQ_shape,
        )

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptor_state(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float16,
            fmha_config.sQ_shape,
        )

    @ts.producer_work
    @staticmethod
    def tma_load(
        stage_info,
        sQ_array,
        seq_coord_q,
        head_coord,
        batch_coord,
        cuseqlen_q,
        seqlen_q,
        inst_idx,
        fmha_config,
    ):
        """Load one Q tile using fixed or packed coordinates."""
        q_head_coord = (
            head_coord * fmha_config.work_tile_q_heads
            + inst_idx * fmha_config.peer_q_head_stride
        )
        q_seq_offset = (
            seq_coord_q
            + inst_idx * fmha_config.peer_q_seq_tile_stride * fmha_config.q_tile_m
        )
        q_seq_extent = cl.int32(0)
        if fmha_config.has_varlen:
            q_seq_extent = cuseqlen_q + seqlen_q - q_seq_offset
        if cl.elect_sync():
            for i in cl.static_iter(range(fmha_config.tma_copy_qkv_iters)):
                d_offset = i * fmha_config.tma_copy_q_granu_inner
                q_coords = (d_offset, q_head_coord, q_seq_offset, batch_coord)
                if fmha_config.has_varlen:
                    q_coords = (d_offset, q_head_coord, q_seq_offset)
                    q_coords = transform_ragged_coords(
                        q_coords,
                        ragged_dim_idx=2,
                        ragged_box_size=fmha_config.q_tile_m,
                        ragged_extent=q_seq_extent,
                    )
                cl.copy_async_bulk_tensor_global_to_shared(
                    stage_info.context.tasks_inputs.tma_q_desc,
                    q_coords,
                    sQ_array.pointer(
                        (
                            stage_info.stage_idx,
                            i * fmha_config.tma_copy_q_granu_elems,
                        )
                    ),
                    stage_info.barrier,
                )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def q_desc(stage_info, sQ_array, inst_idx, fmha_config):
        return _qk_descriptor(
            sQ_array.pointer((inst_idx, 0)),
            fmha_config,
        )


@dataclass(kw_only=True, eq=False)
class SmemKVResource(ts.MemoryResource):
    """SMEM K/V tile buffer with a three-stage TMA-to-UMMA pipeline."""

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_load_state(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float16,
            fmha_config.sK_shape,
        )

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_descriptor_state(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float16,
            fmha_config.sK_shape,
        )

    @staticmethod
    def _tma_load(
        stage_info,
        sK_array,
        kv_head_coord,
        batch_coord,
        cuseqlen_k,
        kv_tile_start,
        is_v,
        fmha_config,
    ):
        tma_desc = stage_info.context.tasks_inputs.tma_k_desc
        if is_v:
            tma_desc = stage_info.context.tasks_inputs.tma_v_desc

        if fmha_config.use_paged_kv:
            tile_idx = kv_tile_start + stage_info.loop_offset
            pages_per_tile = fmha_config.kv_tile_n // fmha_config.num_tokens_per_page
            page_elements = (
                fmha_config.num_tokens_per_page * fmha_config.tma_copy_kv_granu_inner
            )
            page_table_offset = batch_coord * 2 * fmha_config.max_num_pages_per_seq_kv
            logical_page_idx = tile_idx * pages_per_tile
            if cl.elect_sync():
                # The elected lane reads the tile's page IDs
                # before issuing its page-fragment / head-dimension TMA copies.
                page_ids = tuple(
                    cl.int32(
                        stage_info.context.tasks_inputs.page_idx_kv[
                            page_table_offset + logical_page_idx + page_frag
                        ]
                    )
                    for page_frag in cl.static_iter(range(pages_per_tile))
                )
                for page_frag in cl.static_iter(range(pages_per_tile)):
                    page_id = page_ids[page_frag]
                    for i in cl.static_iter(range(fmha_config.tma_copy_kv_stage_iters)):
                        d_offset = i * fmha_config.tma_copy_kv_granu_inner
                        smem_offset = (
                            i * fmha_config.tma_copy_kv_granu_elems
                            + page_frag * page_elements
                        )
                        cl.copy_async_bulk_tensor_global_to_shared(
                            tma_desc,
                            (d_offset, 0, kv_head_coord, page_id),
                            sK_array.pointer((stage_info.stage_idx, smem_offset)),
                            stage_info.barrier,
                        )
            return

        seq_offset = (kv_tile_start + stage_info.loop_offset) * fmha_config.kv_tile_n
        if cl.elect_sync():
            seq_coord_kv = cuseqlen_k + seq_offset
            for i in cl.static_iter(range(fmha_config.tma_copy_kv_stage_iters)):
                d_offset = i * fmha_config.tma_copy_kv_granu_inner
                kv_coords = (d_offset, kv_head_coord, seq_coord_kv, batch_coord)
                if fmha_config.has_varlen:
                    kv_coords = (d_offset, kv_head_coord, seq_coord_kv)
                cl.copy_async_bulk_tensor_global_to_shared(
                    tma_desc,
                    kv_coords,
                    sK_array.pointer(
                        (
                            stage_info.stage_idx,
                            i * fmha_config.tma_copy_kv_granu_elems,
                        )
                    ),
                    stage_info.barrier,
                )

    @ts.producer_work
    @staticmethod
    def k_load(
        stage_info,
        sK_array,
        kv_head_coord,
        batch_coord,
        cuseqlen_k,
        kv_tile_start,
        fmha_config,
    ):
        SmemKVResource._tma_load(
            stage_info,
            sK_array,
            kv_head_coord,
            batch_coord,
            cuseqlen_k,
            kv_tile_start,
            False,
            fmha_config,
        )

    @ts.producer_work
    @staticmethod
    def v_load(
        stage_info,
        sK_array,
        kv_head_coord,
        batch_coord,
        cuseqlen_k,
        kv_tile_start,
        fmha_config,
    ):
        SmemKVResource._tma_load(
            stage_info,
            sK_array,
            kv_head_coord,
            batch_coord,
            cuseqlen_k,
            kv_tile_start,
            True,
            fmha_config,
        )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def k_desc(stage_info, sK_array, fmha_config):
        return _qk_descriptor(
            sK_array.pointer((stage_info.stage_idx, 0)),
            fmha_config,
        )

    @ts.consumer_work(outputs=1)
    @staticmethod
    def v_desc(stage_info, sK_array, fmha_config):
        return _pv_descriptor(
            sK_array.pointer((stage_info.stage_idx, 0)),
            fmha_config,
        )


@dataclass(kw_only=True, eq=False)
class TmemSPResource(ts.MemoryResource):
    """One query half's aliased score/probability TMEM region."""

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def cache_seqlen_k(stage_info):
        """Load the request-local packed K extent once per softmax work tile."""
        batch_coord = stage_info.work_tile.tile_idx[2]
        cum_seqlen_k = stage_info.context.tasks_inputs.cum_seqlen_k
        cuseqlen_k = cum_seqlen_k[batch_coord]
        return cum_seqlen_k[batch_coord + 1] - cuseqlen_k

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def cache_q_offset(stage_info, fmha_config):
        """Cache the bottom-right Q/K shift once per softmax work tile."""
        if not fmha_config.has_varlen:
            return stage_info.context.tasks_inputs.q_offset
        batch_idx = fmha_config.work_tile_coord_indices[2]
        batch_coord = stage_info.work_tile.tile_idx[batch_idx]
        cum_seqlen_q = stage_info.context.tasks_inputs.cum_seqlen_q
        cum_seqlen_k = stage_info.context.tasks_inputs.cum_seqlen_k
        seqlen_q = cum_seqlen_q[batch_coord + 1] - cum_seqlen_q[batch_coord]
        seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
        return seqlen_k - seqlen_q

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def load_scale_softmax_log2(stage_info):
        """Load the runtime softmax scale once before the K/V loop."""
        return stage_info.context.tasks_inputs.scale_softmax_log2[0]

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=2)
    @staticmethod
    def init_softmax_work_tile_state(stage_info):
        return (
            cl.float32(-float("inf")),
            cl.float32(0.0),
        )

    @ts.producer_work
    @staticmethod
    def qk_mma(stage_info, desc_q_base, desc_k_base, section, inst_idx, fmha_config):
        skip_invalid_peer0 = cl.bool_(False)
        if (
            fmha_config.skip_causal_invalid_peer0
            and inst_idx == 0
            and section == FmhaStage.Loop
        ):
            skip_invalid_peer0 = stage_info.loop_offset == stage_info.loop_end - 1
        if skip_invalid_peer0:
            return
        tasks_inputs = stage_info.context.tasks_inputs
        tmem_base = tasks_inputs.tmem_base
        score_column = (
            fmha_config.tmem_s0_offset if inst_idx == 0 else fmha_config.tmem_s1_offset
        )
        score_tmem = _tmem_pointer(tmem_base, column_offset=score_column)
        instruction = _qk_instruction_descriptor(fmha_config)
        qk_phases = fmha_config.qk_mma_tiler[2] // MMA_K
        phases_per_slice = qk_phases // fmha_config.tma_copy_qkv_iters
        step_bytes = MMA_K * 2
        slice_bytes = fmha_config.tma_copy_q_bytes // fmha_config.tma_copy_qkv_iters
        elected = cl.bool_(False)
        if not fmha_config.use_paged_kv:
            elected = cl.elect_sync()
        for kk in cl.static_iter(range(qk_phases)):
            byte_offset = kk * step_bytes
            if kk >= phases_per_slice:
                byte_offset = slice_bytes + (kk - phases_per_slice) * step_bytes
            descriptor_increment = byte_offset >> 4
            if fmha_config.use_paged_kv:
                # Elect a lane for each MMA in the paged pipeline.
                elected = cl.elect_sync()
            if elected:
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    score_tmem,
                    desc_q_base + descriptor_increment,
                    desc_k_base + descriptor_increment,
                    instruction,
                    accumulate=kk != 0,
                    cta_group=cl.CTAGroup.CTA_1,
                )

    @ts.producer_work
    @staticmethod
    def p_read(stage_info):
        return None

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def next_kv_tile_idx(stage_info):
        """Route the dynamic loop end to tail work."""
        return stage_info.loop_offset + 1

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def current_kv_tile_idx(stage_info):
        return stage_info.loop_offset

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_kv_tile_idx(stage_info):
        return cl.int32(0)

    @staticmethod
    def _load_s_chunks(stage_info, inst_idx, fmha_config):
        tasks_inputs = stage_info.context.tasks_inputs
        warp_in_group = stage_info.context.warp_index % len(
            fmha_config.softmax0_warp_ids
        )
        score_column = (
            fmha_config.tmem_s0_offset if inst_idx == 0 else fmha_config.tmem_s1_offset
        )
        row_tmem = _tmem_pointer(
            tasks_inputs.tmem_base,
            lane_offset=warp_in_group * WARP_SIZE,
            column_offset=score_column,
        )
        chunks = tuple(
            cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                _tmem_pointer(
                    row_tmem,
                    column_offset=chunk_idx * fmha_config.tmem_x_load_s,
                ),
                element_count=fmha_config.tmem_x_load_s,
                dtype=cl.float32,
            )
            for chunk_idx in cl.static_iter(
                range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
            )
        )
        cl.tcgen05_wait_load()
        return chunks

    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_row_max(stage_info, row_max, inst_idx, fmha_config):
        chunks = TmemSPResource._load_s_chunks(stage_info, inst_idx, fmha_config)
        old_row_max, row_max = _reduce_row_max(chunks, row_max, fmha_config)
        return old_row_max, row_max, chunks

    @ts.consumer_work(outputs=3)
    @staticmethod
    def packed_dense_k_masked_row_max(
        stage_info,
        row_max,
        seqlen_k,
        inst_idx,
        fmha_config,
    ):
        """Mask scores beyond the active packed request's K right edge."""
        raw_chunks = TmemSPResource._load_s_chunks(stage_info, inst_idx, fmha_config)

        key_tile_base = stage_info.loop_offset * fmha_config.kv_tile_n
        score0, score1, score2, score3 = raw_chunks
        # Only the request's final partial K tile executes the 128 element
        # predicates; all preceding tiles reduce the raw scores directly.
        if seqlen_k < key_tile_base + fmha_config.kv_tile_n:
            score0 = _mask_packed_score_chunk(score0, key_tile_base, seqlen_k)
            score1 = _mask_packed_score_chunk(
                score1,
                key_tile_base + fmha_config.tmem_x_load_s,
                seqlen_k,
            )
            score2 = _mask_packed_score_chunk(
                score2,
                key_tile_base + 2 * fmha_config.tmem_x_load_s,
                seqlen_k,
            )
            score3 = _mask_packed_score_chunk(
                score3,
                key_tile_base + 3 * fmha_config.tmem_x_load_s,
                seqlen_k,
            )
        chunks = (score0, score1, score2, score3)
        old_row_max, row_max = _reduce_row_max(chunks, row_max, fmha_config)
        return old_row_max, row_max, chunks

    @staticmethod
    def _query_row(stage_info, inst_idx, fmha_config):
        seq_idx = fmha_config.work_tile_coord_indices[0]
        seq_coord = stage_info.work_tile.tile_idx[seq_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        warp_in_group = stage_info.context.warp_index % len(
            fmha_config.softmax0_warp_ids
        )
        row_in_tile = warp_in_group * WARP_SIZE + cl.lane_index()
        peer_offset = inst_idx * fmha_config.peer_q_seq_tile_stride
        return (
            seq_coord * fmha_config.cta_tiler[0]
            + peer_offset * fmha_config.q_tile_m
            + row_in_tile
        )

    @ts.consumer_work(outputs=3)
    @staticmethod
    def left_masked_row_max(
        stage_info,
        row_max,
        q_offset,
        inst_idx,
        fmha_config,
    ):
        """Apply the left bound in head-paired sliding-window LOOP work."""
        chunks = TmemSPResource._load_s_chunks(stage_info, inst_idx, fmha_config)
        query_idx = TmemSPResource._query_row(stage_info, inst_idx, fmha_config)
        seq_idx = fmha_config.work_tile_coord_indices[0]
        seq_coord = stage_info.work_tile.tile_idx[seq_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        kv_tile_start = bottom_right_window_tile_start(
            seq_coord,
            fmha_config.q_tile_m,
            fmha_config.kv_tile_n,
            q_offset,
            fmha_config.window_size_left,
        )
        kv_base = (kv_tile_start + stage_info.loop_offset) * fmha_config.kv_tile_n
        lower_bound = query_idx + q_offset - fmha_config.window_size_left
        if fmha_config.has_varlen or fmha_config.has_q_offset:
            upper_bound = query_idx + q_offset
            chunks = tuple(
                _mask_score_chunk(
                    chunks[chunk_idx],
                    kv_base + chunk_idx * fmha_config.tmem_x_load_s,
                    lower_bound,
                    upper_bound,
                )
                for chunk_idx in cl.static_iter(
                    range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
                )
            )
        else:
            chunks = tuple(
                _mask_score_chunk_left(
                    chunks[chunk_idx],
                    kv_base + chunk_idx * fmha_config.tmem_x_load_s,
                    lower_bound,
                )
                for chunk_idx in cl.static_iter(
                    range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
                )
            )
        old_row_max, row_max = _reduce_row_max(chunks, row_max, fmha_config)
        return old_row_max, row_max, chunks

    @ts.consumer_work(outputs=3)
    @staticmethod
    def right_masked_row_max(
        stage_info,
        row_max,
        q_offset,
        kv_tile_idx,
        inst_idx,
        fmha_config,
    ):
        """Apply the head-paired causal right edge and optional left edge."""
        chunks = TmemSPResource._load_s_chunks(stage_info, inst_idx, fmha_config)
        query_idx = TmemSPResource._query_row(stage_info, inst_idx, fmha_config)
        seq_idx = fmha_config.work_tile_coord_indices[0]
        seq_coord = stage_info.work_tile.tile_idx[seq_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        kv_tile_start = cl.int32(0)
        if fmha_config.window_size_left > 0:
            kv_tile_start = bottom_right_window_tile_start(
                seq_coord,
                fmha_config.q_tile_m,
                fmha_config.kv_tile_n,
                q_offset,
                fmha_config.window_size_left,
            )
        kv_base = (kv_tile_start + kv_tile_idx) * fmha_config.kv_tile_n
        upper_bound = query_idx + q_offset
        if fmha_config.needs_window_tail_left_mask:
            lower_bound = query_idx + q_offset - fmha_config.window_size_left
            chunks = tuple(
                _mask_score_chunk(
                    chunks[chunk_idx],
                    kv_base + chunk_idx * fmha_config.tmem_x_load_s,
                    lower_bound,
                    upper_bound,
                )
                for chunk_idx in cl.static_iter(
                    range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
                )
            )
        else:
            chunks = tuple(
                _mask_packed_score_chunk(
                    chunks[chunk_idx],
                    kv_base + chunk_idx * fmha_config.tmem_x_load_s,
                    upper_bound + 1,
                )
                for chunk_idx in cl.static_iter(
                    range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
                )
            )
        old_row_max, row_max = _reduce_row_max(chunks, row_max, fmha_config)
        return old_row_max, row_max, chunks

    @ts.consumer_work(outputs=3)
    @staticmethod
    def query_paired_masked_row_max(
        stage_info,
        row_max,
        q_offset,
        kv_tile_idx,
        inst_idx,
        fmha_config,
    ):
        """Apply bottom-right causal masking to a query-paired score tile."""
        chunks = TmemSPResource._load_s_chunks(stage_info, inst_idx, fmha_config)
        kv_base = kv_tile_idx * fmha_config.kv_tile_n
        seq_idx = fmha_config.work_tile_coord_indices[0]
        seq_coord = stage_info.work_tile.tile_idx[seq_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        q_min = (
            q_offset
            + seq_coord * fmha_config.cta_tiler[0]
            + inst_idx * fmha_config.peer_q_seq_tile_stride * fmha_config.q_tile_m
        )
        k_max = kv_base + fmha_config.qk_mma_tiler[1] - 1
        if q_min <= k_max:
            query_idx = TmemSPResource._query_row(stage_info, inst_idx, fmha_config)
            score0, score1, score2, score3 = chunks
            upper_bound = query_idx + q_offset
            score0 = _mask_packed_score_chunk(score0, kv_base, upper_bound + 1)
            score1 = _mask_packed_score_chunk(
                score1,
                kv_base + fmha_config.tmem_x_load_s,
                upper_bound + 1,
            )
            score2 = _mask_packed_score_chunk(
                score2,
                kv_base + 2 * fmha_config.tmem_x_load_s,
                upper_bound + 1,
            )
            score3 = _mask_packed_score_chunk(
                score3,
                kv_base + 3 * fmha_config.tmem_x_load_s,
                upper_bound + 1,
            )
            chunks = (score0, score1, score2, score3)
        old_row_max, row_max = _reduce_row_max(chunks, row_max, fmha_config)
        return old_row_max, row_max, chunks

    @ts.consumer_work(outputs=2)
    @staticmethod
    def invalid_row_max(stage_info, row_max):
        """Advance peer0's synthetic invalid tail without loading TMEM."""
        row_max_safe = row_max
        if row_max == cl.float32(-float("inf")):
            row_max_safe = cl.float32(0.0)
        return row_max, row_max_safe

    @ts.consumer_work
    @staticmethod
    def invalid_exp2_p(stage_info, row_max):
        """The matching QK MMA is skipped, so no probability tile is written."""
        return None

    @ts.consumer_work(outputs=1)
    @staticmethod
    def exp2_p(
        stage_info,
        row_max,
        scale_softmax_log2,
        score_chunks,
        inst_idx,
        fmha_config,
    ):
        if fmha_config.use_paged_kv:
            return TmemSPResource._exp2_p_store(
                stage_info,
                row_max,
                scale_softmax_log2,
                score_chunks,
                inst_idx,
                fmha_config,
            )
        # Keep the score transform vector-valued for packed f32x2 FMAs;
        # spelling the subtraction and multiply per scalar makes LLVM emit
        # twice as many scalar arithmetic instructions.
        minus_row_max_scale = cl.fma(
            cl.float32(0.0) - row_max,
            scale_softmax_log2,
            cl.float32(0.0),
        )
        probability_inputs = tuple(
            cl.fma(
                chunk,
                scale_softmax_log2,
                minus_row_max_scale,
            )
            for chunk in cl.static_iter(score_chunks)
        )
        probabilities = tuple(
            cl.Vector(
                *tuple(
                    cl.exp2(
                        chunk[item],
                        flush_to_zero=True,
                    )
                    for item in cl.static_iter(range(fmha_config.tmem_x_load_s))
                ),
                dtype=cl.float32,
            )
            for chunk in cl.static_iter(probability_inputs)
        )

        packed_first = cl.Vector(
            *tuple(
                probabilities[chunk_idx][2 * pair:2 * pair + 2]
                .astype(cl.float16)
                .reinterpret_as_scalar(cl.int32)
                for chunk_idx in cl.static_iter(range(2))
                for pair in cl.static_iter(range(fmha_config.tmem_x_load_s // 2))
            ),
            dtype=cl.int32,
        )
        packed_second = cl.Vector(
            *tuple(
                probabilities[chunk_idx][2 * pair:2 * pair + 2]
                .astype(cl.float16)
                .reinterpret_as_scalar(cl.int32)
                for chunk_idx in cl.static_iter(range(2, 4))
                for pair in cl.static_iter(range(fmha_config.tmem_x_load_s // 2))
            ),
            dtype=cl.int32,
        )

        warp_in_group = stage_info.context.warp_index % len(
            fmha_config.softmax0_warp_ids
        )
        p_column = (
            fmha_config.tmem_p0_offset if inst_idx == 0 else fmha_config.tmem_p1_offset
        )
        p_tmem = _tmem_pointer(
            stage_info.context.tasks_inputs.tmem_base,
            lane_offset=warp_in_group * WARP_SIZE,
            column_offset=p_column,
        )
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            p_tmem,
            packed_first,
        )
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            _tmem_pointer(
                p_tmem,
                column_offset=fmha_config.tmem_x_load_s,
            ),
            packed_second,
        )
        cl.tcgen05_wait_store()

        local_sum_pair_0 = cl.Vector(0.0, 0.0, dtype=cl.float32)
        local_sum_pair_1 = cl.Vector(0.0, 0.0, dtype=cl.float32)
        for chunk_idx in cl.static_iter(
            range(fmha_config.kv_tile_n // fmha_config.tmem_x_load_s)
        ):
            for elem_idx in cl.static_iter(range(0, fmha_config.tmem_x_load_s, 2)):
                pair_idx = chunk_idx * (fmha_config.tmem_x_load_s // 2) + elem_idx // 2
                if pair_idx % 2 == 0:
                    local_sum_pair_0 = cl.add(
                        local_sum_pair_0,
                        probabilities[chunk_idx][elem_idx:elem_idx + 2],
                        rounding_mode=cl.RoundingMode.RN,
                        flush_to_zero=False,
                    )
                else:
                    local_sum_pair_1 = cl.add(
                        local_sum_pair_1,
                        probabilities[chunk_idx][elem_idx:elem_idx + 2],
                        rounding_mode=cl.RoundingMode.RN,
                        flush_to_zero=False,
                    )
        local_sum_pair = cl.add(
            local_sum_pair_0,
            local_sum_pair_1,
            rounding_mode=cl.RoundingMode.RN,
            flush_to_zero=False,
        )
        return local_sum_pair[0] + local_sum_pair[1]

    @staticmethod
    def _exp2_p_store(
        stage_info,
        row_max,
        scale_softmax_log2,
        score_chunks,
        inst_idx,
        fmha_config,
    ):
        """Use packed FMA/exp2/early-sum operations for paged KV."""
        warp_in_group = stage_info.context.warp_index % len(
            fmha_config.softmax0_warp_ids
        )
        p_column = (
            fmha_config.tmem_p0_offset if inst_idx == 0 else fmha_config.tmem_p1_offset
        )
        tmem_p_addr = _tmem_pointer(
            stage_info.context.tasks_inputs.tmem_base,
            lane_offset=warp_in_group * WARP_SIZE,
            column_offset=p_column,
        )
        tmem_x = fmha_config.tmem_x_load_s
        num_chunks = fmha_config.kv_tile_n // tmem_x
        p_packing_ratio = cl.float32.bitwidth // cl.float16.bitwidth
        scale = scale_softmax_log2
        minus_row_max_scale = cl.fma(
            cl.float32(0.0) - row_max, scale, cl.float32(0.0)
        )
        local_sum_pair_0 = cl.Vector(0.0, 0.0, dtype=cl.float32)
        local_sum_pair_1 = cl.Vector(0.0, 0.0, dtype=cl.float32)
        s_data = ()
        for chunk_idx in cl.static_iter(range(num_chunks)):
            p_vals = ()
            for elem_idx in cl.static_iter(range(0, tmem_x, 2)):
                fma_pair = cl.fma(
                    score_chunks[chunk_idx][elem_idx:elem_idx + 2],
                    scale,
                    minus_row_max_scale,
                )
                p0 = cl.exp2(fma_pair[0], flush_to_zero=True)
                p1 = cl.exp2(fma_pair[1], flush_to_zero=True)
                pair_idx = chunk_idx * (tmem_x // 2) + elem_idx // 2
                if pair_idx % 2 == 0:
                    local_sum_pair_0 = cl.add(
                        local_sum_pair_0,
                        cl.Vector(p0, p1, dtype=cl.float32),
                        rounding_mode=cl.RoundingMode.RN,
                        flush_to_zero=False,
                    )
                else:
                    local_sum_pair_1 = cl.add(
                        local_sum_pair_1,
                        cl.Vector(p0, p1, dtype=cl.float32),
                        rounding_mode=cl.RoundingMode.RN,
                        flush_to_zero=False,
                    )
                p_vals += (p0, p1)
            s_data += (cl.Vector(*p_vals, dtype=cl.float32),)
        for pair_idx in cl.static_iter(range(num_chunks // p_packing_ratio)):
            store_fragment = cl.Vector(
                *tuple(
                    s_data[pair_idx * p_packing_ratio + slice_idx][elem_idx:elem_idx + 2]
                    .astype(cl.float16)
                    .reinterpret_as_scalar(cl.int32)
                    for slice_idx in cl.static_iter(range(p_packing_ratio))
                    for elem_idx in cl.static_iter(range(0, tmem_x, 2))
                ),
                dtype=cl.int32,
            )
            cl.tcgen05_store(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                _tmem_pointer(tmem_p_addr, column_offset=pair_idx * tmem_x),
                store_fragment,
            )
        local_sum_pair = cl.add(
            local_sum_pair_0,
            local_sum_pair_1,
            rounding_mode=cl.RoundingMode.RN,
            flush_to_zero=False,
        )
        tile_sum = local_sum_pair[0] + local_sum_pair[1]
        cl.tcgen05_wait_store()
        return tile_sum

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def softmax_aux_reduce(
        stage_info,
        old_row_max,
        row_max,
        row_sum,
        p_chunk,
        scale_softmax_log2,
    ):
        alpha = cl.exp2(
            (old_row_max - row_max) * scale_softmax_log2,
            flush_to_zero=True,
        )
        return row_sum * alpha + p_chunk

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def softmax_aux_identity(stage_info, row_max):
        return row_max


@dataclass(kw_only=True, eq=False)
class TmemStatsResource(ts.MemoryResource):
    """One-stage SMEM correction-stat handoff for 128 query rows."""

    @staticmethod
    def _view(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float32,
            (len(fmha_config.softmax0_warp_ids) * WARP_SIZE * 2,),
        )

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_store_state(stage_info, smem_offset, fmha_config):
        return TmemStatsResource._view(stage_info, smem_offset, fmha_config)

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_read_state(stage_info, smem_offset, fmha_config):
        return TmemStatsResource._view(stage_info, smem_offset, fmha_config)

    @ts.producer_work
    @staticmethod
    def store_vec(
        stage_info,
        stats_smem,
        old_row_max,
        row_max,
        row_sum,
        final_stats,
        fmha_config,
    ):
        row = cl.thread_index(0) % fmha_config.q_tile_m
        stat0 = old_row_max
        if final_stats:
            stat0 = row_sum
        (stats_smem.pointer() + row * 2).store(
            cl.Vector(stat0, row_max, dtype=cl.float32),
            alignment=8,
        )

    @ts.consumer_work(outputs=4)
    @staticmethod
    def read_vec(
        stage_info,
        stats_smem,
        scale_softmax_log2,
        final_stats,
        fmha_config,
    ):
        row = cl.thread_index(0) % fmha_config.q_tile_m
        values = (stats_smem.pointer() + row * 2).load(
            count=2,
            alignment=8,
        )
        old_row_max = values[0]
        new_row_max = values[1]
        row_sum = cl.float32(0.0)
        scale = cl.float32(1.0)
        if final_stats:
            row_sum = values[0]
            old_row_max = new_row_max
        else:
            scale = cl.exp2(
                scale_softmax_log2 * (old_row_max - new_row_max),
                flush_to_zero=True,
            )
        return old_row_max, new_row_max, row_sum, scale

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def load_scale_softmax_log2(stage_info):
        """Load the runtime softmax scale once before the correction loop."""
        return stage_info.context.tasks_inputs.scale_softmax_log2[0]

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def load_output_scale(stage_info):
        """Load the runtime output scale once before the correction loop."""
        return stage_info.context.tasks_inputs.output_scale[0]


@dataclass(kw_only=True, eq=False)
class TmemOResource(ts.MemoryResource):
    """Two-stage O accumulator and correction resource."""

    @ts.producer_work
    @staticmethod
    def pv_mma(
        stage_info,
        desc_v_base,
        section,
        inst_idx,
        tail_kv_tile_end,
        fmha_config,
    ):
        """PV MMA: P*V -> O (double-buffered O0/O1)."""
        writes_o0 = section == FmhaStage.Head or (
            section == FmhaStage.Loop and inst_idx == 1
        )
        skip_o0_invalid = cl.bool_(False)
        if (
            fmha_config.skip_causal_invalid_peer0
            and writes_o0
            and section == FmhaStage.Loop
        ):
            skip_o0_invalid = stage_info.loop_offset == stage_info.loop_end - 1
        if skip_o0_invalid:
            return

        tasks_inputs = stage_info.context.tasks_inputs
        tmem_base = tasks_inputs.tmem_base
        output_column = (
            fmha_config.tmem_o0_offset if writes_o0 else fmha_config.tmem_o1_offset
        )
        p_column = (
            fmha_config.tmem_p0_offset if writes_o0 else fmha_config.tmem_p1_offset
        )
        output_tmem = _tmem_pointer(tmem_base, column_offset=output_column)
        instruction = _pv_instruction_descriptor(fmha_config)
        accumulate = cl.bool_(False)
        if section == FmhaStage.Loop and inst_idx == 0:
            # O1 is initialized by the first steady-state iteration.
            accumulate = stage_info.loop_offset > 0
        elif section == FmhaStage.Loop:
            # O0 was initialized by HEAD.
            accumulate = cl.bool_(True)
        elif section == FmhaStage.Tail:
            # If N==1, TAIL is O1's first write; otherwise LOOP initialized it.
            accumulate = tail_kv_tile_end > 0
        elected = cl.elect_sync()
        for kk in cl.static_iter(range(fmha_config.kv_tile_n // MMA_K)):
            p_tmem = _tmem_pointer(
                tmem_base,
                column_offset=p_column + kk * 8,
            )
            v_increment = (MMA_K * fmha_config.tma_copy_kv_granu_inner * 2 >> 4) * kk
            if elected:
                cl.tcgen05_mma(
                    cl.Tcgen05MMAKind.F16,
                    output_tmem,
                    p_tmem,
                    desc_v_base + v_increment,
                    instruction,
                    accumulate=accumulate,
                    cta_group=cl.CTAGroup.CTA_1,
                )
            accumulate = cl.bool_(True)

    @ts.consumer_work
    @staticmethod
    def correct(
        stage_info,
        vec_old_max,
        vec_new_max,
        vec_scale,
        inst_idx,
        fmha_config,
    ):
        # Online softmax needs to rescale the accumulated O tile only when
        # at least one row in this warp observed a new maximum. Avoiding the
        # TMEM load/scale/store for
        # the common no-op case is especially important for long K domains.
        tasks_inputs = stage_info.context.tasks_inputs
        should_rescale = cl.vote_ballot_sync(vec_old_max != vec_new_max) != 0
        # This location requires cross-thread TMEM ordering. The generic
        # pipeline fence is disabled so TMA waits and ordinary TMEM consumers
        # do not inherit unnecessary fences.
        cl.tcgen05_fence_after_thread_sync()
        if not should_rescale:
            return

        warp_in_group = (
            stage_info.context.warp_index - fmha_config.correction_warp_ids[0]
        )
        output_column = (
            fmha_config.tmem_o0_offset if inst_idx == 0 else fmha_config.tmem_o1_offset
        )
        output_tmem = _tmem_pointer(
            tasks_inputs.tmem_base,
            lane_offset=warp_in_group * WARP_SIZE,
            column_offset=output_column,
        )
        for column in cl.static_iter(range(0, fmha_config.epi_tile[1], 16)):
            tile = _tmem_pointer(output_tmem, column_offset=column)
            values = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                tile,
                element_count=16,
                dtype=cl.float32,
            )
            cl.tcgen05_wait_load()
            cl.tcgen05_store(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                tile,
                values * vec_scale,
            )
        cl.tcgen05_wait_store()


@dataclass(kw_only=True, eq=False)
class SmemOResource(ts.MemoryResource):
    """One query half's corrected output staging tile."""

    @staticmethod
    def _view(stage_info, smem_offset, fmha_config):
        return _smem_array(
            stage_info,
            smem_offset,
            cl.float16,
            (fmha_config.sO_stage_elements,),
        )

    @ts.producer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_store_state(stage_info, smem_offset, fmha_config):
        return SmemOResource._view(stage_info, smem_offset, fmha_config)

    @ts.consumer_work(
        work_attrs=ts.WorkAttr.AUXILIARY,
        outputs=1,
    )
    @staticmethod
    def init_output_state(stage_info, smem_offset, fmha_config):
        return SmemOResource._view(stage_info, smem_offset, fmha_config)

    @ts.producer_work
    @staticmethod
    def store_o(
        stage_info,
        sO_array,
        vec_row_sum,
        output_scale,
        inst_idx,
        fmha_config,
    ):
        tasks_inputs = stage_info.context.tasks_inputs
        warp_in_group = (
            stage_info.context.warp_index - fmha_config.correction_warp_ids[0]
        )
        row = warp_in_group * WARP_SIZE + cl.lane_index()
        output_column = (
            fmha_config.tmem_o0_offset if inst_idx == 0 else fmha_config.tmem_o1_offset
        )
        output_tmem = _tmem_pointer(
            tasks_inputs.tmem_base,
            lane_offset=warp_in_group * WARP_SIZE,
            column_offset=output_column,
        )
        scale = output_scale / vec_row_sum
        tmem_x = 16
        for column in cl.static_iter(range(0, fmha_config.epi_tile[1], tmem_x)):
            values = cl.tcgen05_load(
                cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
                _tmem_pointer(output_tmem, column_offset=column),
                element_count=tmem_x,
                dtype=cl.float32,
            )
            cl.tcgen05_wait_load()
            _store_output_values(
                sO_array.pointer(),
                _pack_output_values(values, scale, tmem_x),
                row,
                column,
                tmem_x,
                fmha_config,
            )
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.shared_block(),
        )

    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_output_coords(stage_info, fmha_config):
        seq_idx, head_idx, batch_idx = fmha_config.work_tile_coord_indices
        tile_idx = stage_info.work_tile.tile_idx
        seq_coord = tile_idx[seq_idx]
        head_coord = tile_idx[head_idx]
        batch_coord = tile_idx[batch_idx]
        if fmha_config.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = fmha_config.num_seq_tiles - seq_coord - 1
        seq_coord_q = (
            seq_coord * fmha_config.q_tile_m * fmha_config.work_tile_q_seq_tiles
        )
        return head_coord, batch_coord, seq_coord_q


@dataclass(kw_only=True, eq=False)
class GmemOResource(ts.MemoryResource):
    """TMA store sink for one staged 128-row output tile."""

    @ts.producer_work
    @staticmethod
    def tma_store(
        stage_info,
        sO_array,
        head_coord,
        batch_coord,
        seq_coord_q,
        inst_idx,
        fmha_config,
    ):
        """Store one O tile using fixed or packed coordinates."""
        head_coord = (
            head_coord * fmha_config.work_tile_q_heads
            + inst_idx * fmha_config.peer_q_head_stride
        )
        seq_offset_o = (
            seq_coord_q
            + inst_idx * fmha_config.peer_q_seq_tile_stride * fmha_config.q_tile_m
        )
        q_seq_extent = cl.int32(0)
        should_store = True
        if fmha_config.has_varlen:
            cum_seqlen_q = stage_info.context.tasks_inputs.cum_seqlen_q
            cuseqlen_q = cum_seqlen_q[batch_coord]
            seq_end = cum_seqlen_q[batch_coord + 1]
            seq_offset_o = cuseqlen_q + seq_offset_o
            q_seq_extent = seq_end - seq_offset_o
            should_store = seq_offset_o < seq_end
        if should_store:
            if cl.elect_sync():
                for i in cl.static_iter(range(fmha_config.tma_copy_o_iters)):
                    d_offset = i * fmha_config.tma_copy_o_granu_inner
                    o_coords = (
                        d_offset,
                        head_coord,
                        seq_offset_o,
                        batch_coord,
                    )
                    if fmha_config.has_varlen:
                        o_coords = (d_offset, head_coord, seq_offset_o)
                        o_coords = transform_ragged_coords(
                            o_coords,
                            ragged_dim_idx=2,
                            ragged_box_size=fmha_config.q_tile_m,
                            ragged_extent=q_seq_extent,
                        )
                    cl.copy_async_bulk_tensor_shared_to_global(
                        sO_array.pointer(i * fmha_config.tma_copy_o_granu_elems),
                        stage_info.context.tasks_inputs.tma_o_desc,
                        o_coords,
                    )
            # Query-paired mode can leave stores in flight. Head-paired mode
            # waits before its per-head stage is reused.
            cl.copy_async_bulk_commit_group()
            if fmha_config.gmem_o_store_wait_after_write:
                cl.copy_async_bulk_wait_group(0, read=True)


@dataclass(kw_only=True, eq=False)
class S0S1SequenceResource(ts.MemoryResource):
    """S0-to-S1 sequencer that serializes the two softmax P stores."""
