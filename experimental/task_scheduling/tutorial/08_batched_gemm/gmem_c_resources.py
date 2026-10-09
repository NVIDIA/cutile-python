# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GMEM output resource for the BF16 epilogue."""

from dataclasses import dataclass

import cuda.lang as cl
import task_scheduling as ts

from .gmem_ab_resources import _load_tile_metadata
from .batched_gemm_config import ActKind


@cl.function
def _apply_gated_activation(linear, gate):
    """Evaluate ordinary SwiGLU in FP32."""
    neg_gate_log2e = cl.float32(-1.4426950408889634) * gate
    sigmoid_gate = cl.truediv(cl.float32(1.0), cl.float32(1.0) + cl.exp2(neg_gate_log2e))
    return linear * gate * sigmoid_gate


@cl.function
def _packed_f32x2(ptx, lhs0, lhs1, rhs0, rhs1):
    """Apply a two-input packed f32x2 PTX operation to two scalar pairs."""
    lhs = cl.Vector(lhs0, lhs1, dtype=cl.float32).reinterpret_as_scalar(cl.uint64)
    rhs = cl.Vector(rhs0, rhs1, dtype=cl.float32).reinterpret_as_scalar(cl.uint64)
    (packed,) = cl._inline_ptx(ptx, cl.uint64, lhs, rhs)
    result = cl.Vector(packed, dtype=cl.uint64).reinterpret_as_vector(cl.float32, 2)
    return result[0], result[1]


@cl.function
def _fmul2(lhs0, lhs1, rhs0, rhs1):
    return _packed_f32x2(
        "mul.rn.ftz.f32x2 %0, %1, %2;", lhs0, lhs1, rhs0, rhs1
    )


@cl.function
def _fadd2(lhs0, lhs1, rhs0, rhs1):
    return _packed_f32x2(
        "add.rn.ftz.f32x2 %0, %1, %2;", lhs0, lhs1, rhs0, rhs1
    )


@cl.function
def _ffma2(lhs0, lhs1, rhs0, rhs1, add0, add1):
    lhs = cl.Vector(lhs0, lhs1, dtype=cl.float32).reinterpret_as_scalar(cl.uint64)
    rhs = cl.Vector(rhs0, rhs1, dtype=cl.float32).reinterpret_as_scalar(cl.uint64)
    add = cl.Vector(add0, add1, dtype=cl.float32).reinterpret_as_scalar(cl.uint64)
    (packed,) = cl._inline_ptx(
        "fma.rn.ftz.f32x2 %0, %1, %2, %3;", cl.uint64, lhs, rhs, add
    )
    result = cl.Vector(packed, dtype=cl.uint64).reinterpret_as_vector(cl.float32, 2)
    return result[0], result[1]


@cl.function
def _apply_gated_activation_pair(linear0, linear1, gate0, gate1):
    """Evaluate swap-AB SwiGLU with packed f32x2 operations."""
    linear_scaled0, linear_scaled1 = _ffma2(
        linear0, linear1, cl.float32(1.0), cl.float32(1.0),
        cl.float32(0.0), cl.float32(0.0),
    )
    gate_scaled0, gate_scaled1 = _fmul2(
        gate0, gate1, cl.float32(1.0), cl.float32(1.0)
    )
    neg0, neg1 = _fmul2(
        gate_scaled0, gate_scaled1,
        cl.float32(-1.4426950408889634), cl.float32(-1.4426950408889634),
    )
    exp0 = cl.exp2(neg0, flush_to_zero=True)
    exp1 = cl.exp2(neg1, flush_to_zero=True)
    denom0, denom1 = _fadd2(
        exp0, exp1, cl.float32(1.0), cl.float32(1.0)
    )
    sig0 = cl._nvvm.rcp_approx_ftz_f(denom0)
    sig1 = cl._nvvm.rcp_approx_ftz_f(denom1)
    gate_sig0, gate_sig1 = _fmul2(gate0, gate1, sig0, sig1)
    return _fmul2(linear_scaled0, linear_scaled1, gate_sig0, gate_sig1)


@dataclass(kw_only=True, eq=False)
class GmemCResource(ts.MemoryResource):
    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY)
    @staticmethod
    def init_store_state(stage_info):
        pass

    @ts.producer_work(work_attrs=ts.WorkAttr.AUXILIARY, outputs=2)
    @staticmethod
    def init_epilogue_tile_state(stage_info, cfg):
        return _load_tile_metadata(stage_info, cfg)

    @ts.producer_work
    @staticmethod
    def store_epilogue(
        stage_info,
        tile_expert_idx,
        tile_token_limit,
        t2r_rmem,
        t2r_rmem_1,
        t2r_output_call_idx,
        subtile_idx,
        cfg,
    ):
        tile_coord_m, tile_coord_n, _ = stage_info.work_tile.tile_idx
        warp_in_epi = stage_info.context.warp_index - cfg.epilogue_warp_idx
        row_in_tile = warp_in_epi * cl.int32(32) + cl.lane_index()
        row = tile_coord_m * cl.int32(cfg.tile_m) + row_in_tile
        epi_t2r_repx = cfg.epi_tile_n // 4
        tasks_inputs = stage_info.context.tasks_inputs
        # The validated grid contains complete 128-row tiles. As in the
        # source, ordinary stores include expert padding when TMA OOB is off.
        if cfg.act_kind == int(ActKind.SWIGLU):
            if cfg.is_swap_ab:
                # Source 16x256b register layout. Weights were first gated-
                # interleaved and then shuffled, so each register pair is an
                # adjacent (linear, gate) pair for one logical output row.
                lane_id = cl.lane_index()
                warp_in_epi4 = warp_in_epi % cl.int32(4)
                base_tmem_col = (lane_id & cl.int32(3)) * cl.int32(2)
                base_row_idx = (
                    warp_in_epi4 * cl.int32(16)
                    + (lane_id >> cl.int32(2)) * cl.int32(2)
                )
                output_m = tasks_inputs.problem_m // cl.int32(2)
                m_tile_base = tile_coord_m * cl.int32(cfg.tile_m)
                n_subtile_offset = subtile_idx * cl.int32(cfg.epi_tile_n)
                n_tile_base = (
                    tile_coord_n * cl.int32(cfg.tile_n) + n_subtile_offset
                )
                for group in cl.static_iter(range(cfg.epi_tile_n // 8)):
                    col_off = cl.int32(group * 8)
                    for col_sub in cl.static_iter(range(2)):
                        gate_idx = group * 4 + col_sub
                        up_idx = group * 4 + 2 + col_sub
                        tmem_col_even = base_tmem_col + col_off + cl.int32(col_sub)
                        gate0 = t2r_rmem[gate_idx]
                        up0 = t2r_rmem[up_idx]
                        gate1 = t2r_rmem_1[gate_idx]
                        up1 = t2r_rmem_1[up_idx]
                        result0, result1 = _apply_gated_activation_pair(
                            gate0, gate1, up0, up1
                        )
                        result0_out = tasks_inputs.gC.dtype(result0)
                        result1_out = tasks_inputs.gC.dtype(result1)
                        m_row0 = m_tile_base // cl.int32(2) + base_row_idx
                        n_col = n_tile_base + tmem_col_even
                        flat_idx0 = n_col * output_m + m_row0
                        flat_idx1 = flat_idx0 + cl.int32(1)
                        (tasks_inputs.gC.pointer() + flat_idx0).store(
                            result0_out, alignment=2
                        )
                        (tasks_inputs.gC.pointer() + flat_idx1).store(
                            result1_out, alignment=2
                        )
                return
            output_n = tasks_inputs.problem_n // cl.int32(2)
            col_base = (
                tile_coord_n * cl.int32(cfg.tile_n // 2)
                + subtile_idx * cl.int32(epi_t2r_repx // 2)
            )
            for pi in cl.static_iter(range(epi_t2r_repx // 2)):
                # Source names: first member is linear; SiLU acts on "up".
                gate = t2r_rmem[pi * 2]
                up = t2r_rmem[pi * 2 + 1]
                result = _apply_gated_activation(gate, up)
                result_out = tasks_inputs.gC.dtype(result)
                linear_idx = row * output_n + col_base + cl.int32(pi)
                (tasks_inputs.gC.pointer() + linear_idx).store(result_out, alignment=2)
        else:
            col_base = tile_coord_n * cl.int32(cfg.tile_n) + subtile_idx * epi_t2r_repx
            vec_out = t2r_rmem.astype(tasks_inputs.gC.dtype)
            linear_idx = row * tasks_inputs.problem_n + col_base
            (tasks_inputs.gC.pointer() + linear_idx).store(
                vec_out, alignment=2 * epi_t2r_repx
            )
