# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and memory resources for the SM120 FMHA context kernel."""

from dataclasses import dataclass

import cuda.lang as cl

import task_scheduling as ts


WARP_SIZE = 32
SUPPORTED_TILES = (128, 64)
SUPPORTED_HEAD_DIMS = tuple(range(16, 257, 16))
K_STAGE_COUNT = 1
V_STAGE_COUNT = 1
LOAD_WARPS = 1
PADDING_WARPS = 3
LOAD_REGISTERS = 40
COMPUTE_REGISTERS = 232
PADDING_REGISTERS = 40
COMPUTE_SYNC_BARRIER = 1
ELEMENT_BYTES = 2
BUFFER_ALIGNMENT = 128
MMA_M = 16
MMA_N = 8
MMA_K = 16
SM120_SMEM_CAPACITY_BYTES = 101376


@dataclass(frozen=True)
class FmhaConfig:
    """Compile-time specialization parameters."""

    head_dim: int
    dtype: str = "fp16"
    is_causal: bool = False
    q_tile: int = 128
    kv_tile: int = 128
    use_causal_head_fast_grid: bool = False
    k_stage: int = K_STAGE_COUNT
    v_stage: int = V_STAGE_COUNT
    k_smem_offset: int = 0
    v_smem_offset: int = 0

    def __post_init__(self):
        if self.head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError("head_dim must be a multiple of 16 in [16, 256]")
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError("dtype must be fp16 or bf16")
        if self.q_tile not in SUPPORTED_TILES:
            raise ValueError(f"q_tile must be one of {SUPPORTED_TILES}")
        if self.kv_tile not in SUPPORTED_TILES:
            raise ValueError(f"kv_tile must be one of {SUPPORTED_TILES}")
        if self.k_stage <= 0 or self.v_stage <= 0:
            raise ValueError("K/V pipeline stage counts must be positive")
        if self.head_dim * ELEMENT_BYTES % 32:
            raise ValueError("the head dimension must form a 32-byte TMA row")

    @property
    def num_compute_warps(self):
        return 8 if self.q_tile == 128 else 4

    @property
    def num_load_warps(self):
        return LOAD_WARPS

    @property
    def num_padding_warps(self):
        return PADDING_WARPS

    @property
    def compute_regs(self):
        return COMPUTE_REGISTERS

    @property
    def load_regs(self):
        return LOAD_REGISTERS

    @property
    def padding_regs(self):
        return PADDING_REGISTERS

    @property
    def block_threads(self):
        return (
            self.num_compute_warps + self.num_load_warps + self.num_padding_warps
        ) * WARP_SIZE

    @property
    def load_warp_index(self):
        return self.num_compute_warps

    @property
    def padding_warp_index(self):
        return self.num_compute_warps + self.num_load_warps

    @property
    def tile_elements(self):
        return self.kv_tile * self.head_dim

    @property
    def tma_copy_kv_bytes(self):
        return self.tile_elements * ELEMENT_BYTES

    @property
    def swizzle_chunk_bytes(self):
        row_bytes = self.head_dim * ELEMENT_BYTES
        if row_bytes % 128 == 0:
            return 128
        if row_bytes % 64 == 0:
            return 64
        return 32

    @property
    def tma_swizzle_chunk_elems(self):
        return self.swizzle_chunk_bytes // ELEMENT_BYTES

    @property
    def tma_swizzle_chunks(self):
        return self.head_dim // self.tma_swizzle_chunk_elems


def nvvm_threadquad_reduction_max(value):
    other, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 2)
    value = cl.maximum(value, other)
    other, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 1)
    return cl.maximum(value, other)


def nvvm_threadquad_reduction_sum(value):
    other, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 2)
    value += other
    other, _ = cl.shuffle_sync(cl.ShuffleKind.XOR, value, 1)
    return value + other


def ptx_mma_m16n8k16_f32(a, b, c, is_bf16):
    ab_tag = "bf16" if cl.ensure_constant(is_bf16) else "f16"
    result = cl._inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32."
        + ab_tag
        + "."
        + ab_tag
        + ".f32 {$0,$1,$2,$3}, "
        "{$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        *tuple(cl.float32 for _ in cl.static_iter(range(4))),
        *tuple(a[item] for item in cl.static_iter(range(4))),
        *tuple(b[item] for item in cl.static_iter(range(2))),
        *tuple(c[item] for item in cl.static_iter(range(4))),
    )
    return cl.Vector(*result, dtype=cl.float32)


def get_swizzled_col(row, column, row_stride, element_bytes):
    row_stride_bytes = row_stride * element_bytes
    if cl.ensure_constant(row_stride_bytes % 128 == 0):
        chunk_bytes, swizzle_bits, row_shift = 128, 3, 0
    elif cl.ensure_constant(row_stride_bytes % 64 == 0):
        chunk_bytes, swizzle_bits, row_shift = 64, 2, 1
    else:
        chunk_bytes, swizzle_bits, row_shift = 32, 1, 2
    chunk_size = chunk_bytes // element_bytes
    # FP16 and BF16 both place the XOR at the eight-element (16-byte) boundary.
    swizzle_base = 3
    chunk = column // chunk_size
    column_in_chunk = column % chunk_size
    bit_mask = (1 << swizzle_bits) - 1
    return chunk * chunk_size + (
        column_in_chunk ^ (((row >> row_shift) & bit_mask) << swizzle_base)
    )


def pack_to_i32(src, dtype):
    """Pack a pair of 16-bit values into one 32-bit register."""
    return src.astype(dtype).reinterpret_as_scalar(cl.int32)


def _make_smem_view(stage_info, offset, tile_elements):
    base = stage_info.context.smem_base.pointer()
    pointer = cl.bitcast(
        base + offset,
        cl.pointer_dtype(cl.uint16, base.memory_space),
    )
    return cl.Array.from_parts(pointer, tile_elements)


@dataclass(kw_only=True, eq=False)
class GmemQKVResource(ts.MemoryResource):
    """Resolve the CTA's batch and head coordinates."""

    @staticmethod
    def get_tile_coords(stage_info, use_head_fast_grid):
        """Return ``(physical_tile_q, batch, head)`` for the launch grid."""
        inputs = stage_info.context.tasks_inputs
        tile_x, tile_y, tile_z = stage_info.work_tile.tile_idx
        if cl.ensure_constant(use_head_fast_grid):
            tile_q_idx = tile_x // inputs.num_heads_q
            head_coord = tile_x - tile_q_idx * inputs.num_heads_q
            return tile_q_idx, tile_y, head_coord
        return tile_x, tile_y, tile_z

    @staticmethod
    def get_tile_q_idx(stage_info, q_tile, is_causal, use_head_fast_grid):
        """Return the logical Q tile index for the current work tile."""
        inputs = stage_info.context.tasks_inputs
        tile_q_idx, _, _ = GmemQKVResource.get_tile_coords(
            stage_info, use_head_fast_grid
        )
        if cl.ensure_constant(is_causal):
            num_q_tiles = cl.cdiv(inputs.seqlen_q, q_tile)
            tile_q_idx = num_q_tiles - tile_q_idx - 1
        return tile_q_idx

    @staticmethod
    def get_kv_tile_count(
        stage_info, q_tile, kv_tile, is_causal, use_head_fast_grid
    ):
        """Return the number of K/V tiles processed by this Q tile."""
        inputs = stage_info.context.tasks_inputs
        kv_tile_count = inputs.num_kv_tiles
        if cl.ensure_constant(is_causal):
            tile_q_idx = GmemQKVResource.get_tile_q_idx(
                stage_info, q_tile, is_causal, use_head_fast_grid
            )
            if cl.ensure_constant(q_tile == kv_tile):
                causal_n = tile_q_idx + 1
            else:
                max_q_row = tile_q_idx * q_tile + q_tile - 1
                causal_n = max_q_row // kv_tile + 1
            kv_tile_count = cl.minimum(kv_tile_count, causal_n)
        return kv_tile_count

    @staticmethod
    def get_kv_tile_idx(
        stage_info,
        loop_offset,
        q_tile,
        kv_tile,
        is_causal,
        use_head_fast_grid,
    ):
        """Map a forward loop offset to a right-to-left K/V tile index."""
        kv_tile_count = stage_info.loop_end
        if kv_tile_count is None:
            kv_tile_count = GmemQKVResource.get_kv_tile_count(
                stage_info,
                q_tile,
                kv_tile,
                is_causal,
                use_head_fast_grid,
            )
        return kv_tile_count - 1 - loop_offset

    @ts.consumer_work(outputs=3)
    @staticmethod
    def compute_coords(stage_info, q_tile, is_causal, use_head_fast_grid):
        """Resolve and publish the current batch, head, and Q row."""
        _, batch_coord, head_coord = GmemQKVResource.get_tile_coords(
            stage_info, use_head_fast_grid
        )
        tile_q_idx = GmemQKVResource.get_tile_q_idx(
            stage_info, q_tile, is_causal, use_head_fast_grid
        )
        return batch_coord, head_coord, tile_q_idx * q_tile


@dataclass(kw_only=True, eq=False)
class SmemKResource(ts.MemoryResource):
    """Stage K and host the QK MMA."""

    @staticmethod
    def _init_smem_state(stage_info, smem_offset, tile_elements):
        return _make_smem_view(stage_info, smem_offset, tile_elements)

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_load_state(stage_info, smem_offset, tile_elements):
        return SmemKResource._init_smem_state(
            stage_info, smem_offset, tile_elements
        )

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_compute_state(stage_info, tile_elements):
        return SmemKResource._init_smem_state(stage_info, 0, tile_elements)

    @ts.producer_work
    @staticmethod
    def tma_load(
        stage_info,
        k_smem,
        batch_coord,
        head_coord,
        q_tile,
        kv_tile,
        is_causal,
        use_causal_head_fast_grid,
    ):
        inputs = stage_info.context.tasks_inputs
        kv_sequence = (
            GmemQKVResource.get_kv_tile_idx(
                stage_info,
                stage_info.loop_offset,
                q_tile,
                kv_tile,
                is_causal,
                use_causal_head_fast_grid,
            )
            * kv_tile
        )
        if cl.lane_index() == 0:
            cl.copy_async_bulk_tensor_global_to_shared(
                inputs.tma_k_desc,
                (0, kv_sequence, 0, head_coord, batch_coord),
                k_smem.pointer(),
                stage_info.barrier,
            )

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def load_q(
        stage_info,
        head_dim,
        q_tile,
        num_compute_warps,
        is_causal,
        use_causal_head_fast_grid,
    ):
        inputs = stage_info.context.tasks_inputs
        lane = cl.lane_index()
        warp = stage_info.context.warp_index
        rows_per_warp = q_tile // num_compute_warps
        tile_q_idx = GmemQKVResource.get_tile_q_idx(
            stage_info,
            q_tile,
            is_causal,
            use_causal_head_fast_grid,
        )
        q_row_base = tile_q_idx * q_tile
        sequence_stride = inputs.num_heads_q * head_dim
        _, batch_coord, head_coord = GmemQKVResource.get_tile_coords(
            stage_info, use_causal_head_fast_grid
        )
        q_head_offset = (
            batch_coord * inputs.seqlen_q * sequence_stride
            + head_coord * head_dim
        )
        q_warp_row = warp * rows_per_warp
        q_registers = cl.Vector(
            *tuple(cl.int32(0) for _ in cl.static_iter(range(head_dim // 4))),
            dtype=cl.int32,
        )
        lane_row = lane // 4
        lane_column = (lane % 4) * 2
        for d_fragment in cl.static_iter(range(head_dim // MMA_K)):
            column = d_fragment * MMA_K + lane_column
            register_offset = d_fragment * 4
            coordinates = (
                (lane_row, column),
                (lane_row + 8, column),
                (lane_row, column + 8),
                (lane_row + 8, column + 8),
            )
            for item in cl.static_iter(range(4)):
                local_row, local_column = coordinates[item]
                global_row = q_row_base + q_warp_row + local_row
                packed = cl.int32(0)
                if global_row < inputs.seqlen_q and local_column < head_dim:
                    linear_offset = (
                        q_head_offset + global_row * sequence_stride + local_column
                    )
                    pair = (inputs.q + linear_offset).load(count=2, alignment=4)
                    packed = pair.reinterpret_as_scalar(cl.int32)
                q_registers = q_registers.with_item(register_offset + item, packed)
        return q_registers

    @ts.consumer_work(outputs=1)
    @staticmethod
    def qk_mma(
        stage_info,
        k_smem,
        q_registers,
        head_dim,
        kv_tile,
        tma_swizzle_chunk_elems,
        is_bf16,
    ):
        lane = cl.lane_index()
        k_fragments = kv_tile // MMA_N
        scores = cl.Vector(
            *tuple(cl.float32(0.0) for _ in cl.static_iter(range(k_fragments * 4))),
            dtype=cl.float32,
        )
        lane_div8 = lane // 8
        lane_mod8 = lane % 8
        k_row_in_pair = (lane // 16) * 8 + lane_mod8
        k_column_in_pair = (lane_div8 % 2) * 8
        chunk_elements = tma_swizzle_chunk_elems
        for k_pair in cl.static_iter(range(k_fragments // 2)):
            for d_fragment in cl.static_iter(range(head_dim // MMA_K)):
                k_row = k_pair * 16 + k_row_in_pair
                k_column = d_fragment * 16 + k_column_in_pair
                k_chunk = k_column // chunk_elements
                k_column_in_chunk = k_column % chunk_elements
                physical_row = k_chunk * kv_tile + k_row
                pointer = k_smem.pointer(
                    physical_row * chunk_elements
                    + get_swizzled_col(
                        physical_row,
                        k_column_in_chunk,
                        chunk_elements,
                        ELEMENT_BYTES,
                    )
                )
                k_values = cl.load_matrix(
                    pointer,
                    shape=cl.MatrixLoadShape.M8N8,
                    count=4,
                    transpose=False,
                )
                q_offset = d_fragment * 4
                score_offset = k_pair * 8
                low = ptx_mma_m16n8k16_f32(
                    q_registers[q_offset: q_offset + 4],
                    k_values[0:2],
                    scores[score_offset: score_offset + 4],
                    is_bf16,
                )
                high = ptx_mma_m16n8k16_f32(
                    q_registers[q_offset: q_offset + 4],
                    k_values[2:4],
                    scores[score_offset + 4: score_offset + 8],
                    is_bf16,
                )
                for item in cl.static_iter(range(4)):
                    scores = scores.with_item(score_offset + item, low[item])
                    scores = scores.with_item(score_offset + 4 + item, high[item])
        return scores


@dataclass(kw_only=True, eq=False)
class SmemVResource(ts.MemoryResource):
    """Stage V and host online softmax plus PV MMA."""

    @staticmethod
    def _init_smem_state(stage_info, smem_offset, tile_elements):
        return _make_smem_view(stage_info, smem_offset, tile_elements)

    @staticmethod
    def _init_work_tile_state(head_dim):
        accumulator = cl.Vector(
            *tuple(
                cl.float32(0.0) for _ in cl.static_iter(range((head_dim // MMA_N) * 4))
            ),
            dtype=cl.float32,
        )
        row_max = cl.Vector(
            cl.float32(-float("inf")),
            cl.float32(-float("inf")),
            dtype=cl.float32,
        )
        row_sum = cl.Vector(cl.float32(0.0), cl.float32(0.0), dtype=cl.float32)
        return accumulator, row_max, row_sum

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_load_state(stage_info, smem_offset, tile_elements):
        return SmemVResource._init_smem_state(
            stage_info, smem_offset, tile_elements
        )

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=1)
    @staticmethod
    def init_compute_state(stage_info, tile_elements, tma_copy_kv_bytes):
        return SmemVResource._init_smem_state(
            stage_info, tma_copy_kv_bytes, tile_elements
        )

    @ts.consumer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=3)
    @staticmethod
    def init_compute_work_tile_state(stage_info, head_dim):
        _ = stage_info
        return SmemVResource._init_work_tile_state(head_dim)

    @ts.producer_work
    @staticmethod
    def tma_load(
        stage_info,
        v_smem,
        batch_coord,
        head_coord,
        q_tile,
        kv_tile,
        is_causal,
        use_causal_head_fast_grid,
    ):
        inputs = stage_info.context.tasks_inputs
        kv_sequence = (
            GmemQKVResource.get_kv_tile_idx(
                stage_info,
                stage_info.loop_offset,
                q_tile,
                kv_tile,
                is_causal,
                use_causal_head_fast_grid,
            )
            * kv_tile
        )
        if cl.lane_index() == 0:
            cl.copy_async_bulk_tensor_global_to_shared(
                inputs.tma_v_desc,
                (0, kv_sequence, 0, head_coord, batch_coord),
                v_smem.pointer(),
                stage_info.barrier,
            )

    @staticmethod
    def _softmax_pv_body(
        stage_info,
        v_smem,
        scores,
        accumulator,
        row_max,
        row_sum,
        loop_offset,
        with_correction,
        check_mask,
        head_dim,
        q_tile,
        kv_tile,
        num_compute_warps,
        tma_swizzle_chunk_elems,
        is_bf16,
        is_causal,
        use_causal_head_fast_grid,
    ):
        inputs = stage_info.context.tasks_inputs
        lane = cl.lane_index()
        warp = stage_info.context.warp_index
        rows_per_warp = q_tile // num_compute_warps
        q_warp_row = warp * rows_per_warp
        q_sequence_base = GmemQKVResource.get_tile_q_idx(
            stage_info,
            q_tile,
            is_causal,
            use_causal_head_fast_grid,
        ) * q_tile
        kv_sequence_base = (
            GmemQKVResource.get_kv_tile_idx(
                stage_info,
                loop_offset,
                q_tile,
                kv_tile,
                is_causal,
                use_causal_head_fast_grid,
            )
            * kv_tile
        )
        k_fragments = kv_tile // MMA_N
        d_fragments = head_dim // MMA_N

        for row_half in cl.static_iter(range(2)):
            score_item_low = row_half * 2
            score_item_high = score_item_low + 1
            q_local_row = q_warp_row + lane // 4 + row_half * 8
            q_global_row = q_sequence_base + q_local_row
            current_max = cl.float32(-float("inf"))
            for k_fragment in cl.static_iter(range(k_fragments)):
                score_offset = k_fragment * 4
                score0 = scores[score_offset + score_item_low]
                score1 = scores[score_offset + score_item_high]
                if cl.ensure_constant(check_mask):
                    valid_limit = inputs.seqlen_k
                    if cl.ensure_constant(is_causal):
                        valid_limit = cl.minimum(q_global_row + 1, inputs.seqlen_k)
                    key0 = kv_sequence_base + k_fragment * 8 + 2 * (lane % 4)
                    if not key0 < valid_limit:
                        score0 = cl.float32(-float("inf"))
                    if not key0 + 1 < valid_limit:
                        score1 = cl.float32(-float("inf"))
                    scores = scores.with_item(score_offset + score_item_low, score0)
                    scores = scores.with_item(score_offset + score_item_high, score1)
                current_max = cl.maximum(current_max, cl.maximum(score0, score1))
            current_max = nvvm_threadquad_reduction_max(current_max)

            previous_max = row_max[row_half]
            new_max = (
                cl.maximum(previous_max, current_max)
                if cl.ensure_constant(with_correction)
                else current_max
            )
            row_max = row_max.with_item(row_half, new_max)
            exponent_max = new_max
            if cl.ensure_constant(check_mask):
                if exponent_max == cl.float32(-float("inf")):
                    exponent_max = cl.float32(0.0)
            negative_max_scaled = -(exponent_max * inputs.softmax_scale_log2)
            tile_sum = cl.float32(0.0)
            for k_fragment in cl.static_iter(range(k_fragments)):
                score_offset = k_fragment * 4
                score0 = scores[score_offset + score_item_low]
                score1 = scores[score_offset + score_item_high]
                probability_input = cl.fma(
                    cl.Vector(score0, score1, dtype=cl.float32),
                    inputs.softmax_scale_log2,
                    negative_max_scaled,
                )
                probability0 = cl.exp2(probability_input[0], flush_to_zero=True)
                probability1 = cl.exp2(probability_input[1], flush_to_zero=True)
                tile_sum += probability0 + probability1
                probability_pair = cl.Vector(probability0, probability1)
                probability_dtype = (
                    cl.bfloat16 if cl.ensure_constant(is_bf16) else cl.float16
                )
                packed = pack_to_i32(probability_pair, probability_dtype)
                scores = scores.with_item(
                    score_offset + row_half * 2,
                    cl.bitcast(packed, cl.float32),
                )
            tile_sum = nvvm_threadquad_reduction_sum(tile_sum)

            if cl.ensure_constant(with_correction):
                if cl.ensure_constant(check_mask):
                    old_scale = cl.float32(1.0)
                    if new_max > cl.float32(-float("inf")):
                        old_scale = cl.exp2(
                            (previous_max - new_max) * inputs.softmax_scale_log2,
                            flush_to_zero=True,
                        )
                else:
                    old_scale = cl.exp2(
                        (previous_max - new_max) * inputs.softmax_scale_log2,
                        flush_to_zero=True,
                    )
                row_sum = row_sum.with_item(
                    row_half, row_sum[row_half] * old_scale + tile_sum
                )
                for d_fragment in cl.static_iter(range(d_fragments)):
                    offset = d_fragment * 4 + row_half * 2
                    corrected = accumulator[offset: offset + 2] * old_scale
                    accumulator = accumulator.with_item(offset, corrected[0])
                    accumulator = accumulator.with_item(offset + 1, corrected[1])
            else:
                row_sum = row_sum.with_item(row_half, tile_sum)

        lane_div8 = lane // 8
        lane_mod8 = lane % 8
        v_row_in_pair = (lane_div8 % 2) * 8 + lane_mod8
        v_column_in_pair = (lane_div8 // 2) * 8
        chunk_elements = tma_swizzle_chunk_elems
        for v_fragment in cl.static_iter(range(kv_tile // MMA_K)):
            for d_pair in cl.static_iter(range(d_fragments // 2)):
                v_row = v_fragment * 16 + v_row_in_pair
                v_column = d_pair * 16 + v_column_in_pair
                v_chunk = v_column // chunk_elements
                v_column_in_chunk = v_column % chunk_elements
                physical_row = v_chunk * kv_tile + v_row
                pointer = v_smem.pointer(
                    physical_row * chunk_elements
                    + get_swizzled_col(
                        physical_row,
                        v_column_in_chunk,
                        chunk_elements,
                        ELEMENT_BYTES,
                    )
                )
                v_values = cl.load_matrix(
                    pointer,
                    shape=cl.MatrixLoadShape.M8N8,
                    count=4,
                    transpose=True,
                )
                probability_offset = 2 * v_fragment * 4
                p = cl.Vector(
                    cl.bitcast(scores[probability_offset], cl.int32),
                    cl.bitcast(scores[probability_offset + 2], cl.int32),
                    cl.bitcast(scores[probability_offset + 4], cl.int32),
                    cl.bitcast(scores[probability_offset + 6], cl.int32),
                    dtype=cl.int32,
                )
                accumulator_offset = d_pair * 8
                low = ptx_mma_m16n8k16_f32(
                    p,
                    v_values[0:2],
                    accumulator[accumulator_offset: accumulator_offset + 4],
                    is_bf16,
                )
                high = ptx_mma_m16n8k16_f32(
                    p,
                    v_values[2:4],
                    accumulator[accumulator_offset + 4: accumulator_offset + 8],
                    is_bf16,
                )
                for item in cl.static_iter(range(4)):
                    accumulator = accumulator.with_item(
                        accumulator_offset + item, low[item]
                    )
                    accumulator = accumulator.with_item(
                        accumulator_offset + 4 + item, high[item]
                    )
        return accumulator, row_max, row_sum

    @ts.consumer_work(outputs=3)
    @staticmethod
    def softmax_pv(
        stage_info,
        v_smem,
        scores,
        accumulator,
        row_max,
        row_sum,
        loop_offset,
        head_dim,
        q_tile,
        kv_tile,
        num_compute_warps,
        tma_swizzle_chunk_elems,
        is_bf16,
        is_causal,
        use_causal_head_fast_grid,
    ):
        return SmemVResource._softmax_pv_body(
            stage_info,
            v_smem,
            scores,
            accumulator,
            row_max,
            row_sum,
            loop_offset,
            False,
            True,
            head_dim,
            q_tile,
            kv_tile,
            num_compute_warps,
            tma_swizzle_chunk_elems,
            is_bf16,
            is_causal,
            use_causal_head_fast_grid,
        )

    @ts.consumer_work(outputs=3)
    @staticmethod
    def softmax_pv_with_correction(
        stage_info,
        v_smem,
        scores,
        accumulator,
        row_max,
        row_sum,
        head_dim,
        q_tile,
        kv_tile,
        num_compute_warps,
        tma_swizzle_chunk_elems,
        is_bf16,
        is_causal,
        use_causal_head_fast_grid,
    ):
        mask_steps = cl.cdiv(q_tile, kv_tile) if cl.ensure_constant(is_causal) else 1
        if stage_info.loop_offset < mask_steps:
            return SmemVResource._softmax_pv_body(
                stage_info,
                v_smem,
                scores,
                accumulator,
                row_max,
                row_sum,
                stage_info.loop_offset,
                True,
                True,
                head_dim,
                q_tile,
                kv_tile,
                num_compute_warps,
                tma_swizzle_chunk_elems,
                is_bf16,
                is_causal,
                use_causal_head_fast_grid,
            )
        return SmemVResource._softmax_pv_body(
            stage_info,
            v_smem,
            scores,
            accumulator,
            row_max,
            row_sum,
            stage_info.loop_offset,
            True,
            False,
            head_dim,
            q_tile,
            kv_tile,
            num_compute_warps,
            tma_swizzle_chunk_elems,
            is_bf16,
            is_causal,
            use_causal_head_fast_grid,
        )


@dataclass(kw_only=True, eq=False)
class GmemOResource(ts.MemoryResource):
    """Normalize and store the output tile."""

    @ts.producer_work
    @staticmethod
    def epilogue_and_store(
        stage_info,
        accumulator,
        row_sum,
        head_dim,
        q_tile,
        num_compute_warps,
        tile_elements,
        is_bf16,
        is_causal,
        use_causal_head_fast_grid,
    ):
        inputs = stage_info.context.tasks_inputs
        k_smem = _make_smem_view(stage_info, 0, tile_elements)
        lane = cl.lane_index()
        warp = stage_info.context.warp_index
        lane_div8 = lane // 8
        lane_mod8 = lane % 8
        lane_div16 = lane // 16
        d_fragment_pairs = head_dim // 16
        sequence_stride = inputs.num_heads_q * head_dim
        _, batch_coord, head_coord = GmemQKVResource.get_tile_coords(
            stage_info, use_causal_head_fast_grid
        )
        output_head_offset = (
            batch_coord * inputs.seqlen_q * sequence_stride
            + head_coord * head_dim
        )

        cl.barrier_sync_block_aligned(
            number_of_threads=num_compute_warps * WARP_SIZE,
            barrier_id=COMPUTE_SYNC_BARRIER,
        )
        inverse0 = cl._nvvm.rcp_approx_ftz_f(row_sum[0])
        inverse1 = cl._nvvm.rcp_approx_ftz_f(row_sum[1])
        inverse = cl.Vector(
            inverse0,
            inverse0,
            inverse1,
            inverse1,
            inverse0,
            inverse0,
            inverse1,
            inverse1,
            dtype=cl.float32,
        )
        for d_pair in cl.static_iter(range(d_fragment_pairs)):
            offset = d_pair * 8
            scaled = accumulator[offset: offset + 8] * inverse
            if cl.ensure_constant(is_bf16):
                packed = scaled.astype(cl.bfloat16).reinterpret_as_vector(cl.int32, 4)
            else:
                packed = scaled.astype(cl.float16).reinterpret_as_vector(cl.int32, 4)
            shared_pointer = k_smem.pointer(
                (warp * d_fragment_pairs + d_pair) * (16 * 16) + lane * 8
            )
            cl.store_matrix(
                shared_pointer,
                packed,
                shape=cl.MatrixStoreShape.M8N8,
                transpose=False,
            )

        store_row = lane_mod8 + (lane_div8 % 2) * 8
        store_column = lane_div16 * 8
        global_row = (
            GmemQKVResource.get_tile_q_idx(
                stage_info,
                q_tile,
                is_causal,
                use_causal_head_fast_grid,
            )
            * q_tile
            + warp * (q_tile // num_compute_warps)
            + store_row
        )
        for d_pair in cl.static_iter(range(d_fragment_pairs)):
            global_column = d_pair * 16 + store_column
            if global_row < inputs.seqlen_q and global_column < head_dim:
                shared_pointer = k_smem.pointer(
                    (warp * d_fragment_pairs + d_pair) * (16 * 16) + lane * 8
                )
                output = shared_pointer.load(count=8, alignment=16)
                if cl.ensure_constant(is_bf16):
                    output = output.reinterpret_as_vector(cl.bfloat16, 8)
                else:
                    output = output.reinterpret_as_vector(cl.float16, 8)
                linear_offset = (
                    output_head_offset
                    + global_row * sequence_stride
                    + global_column
                )
                (inputs.o + linear_offset).store(output, alignment=16)
