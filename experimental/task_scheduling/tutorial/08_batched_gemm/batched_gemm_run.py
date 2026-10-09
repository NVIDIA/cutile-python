# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Correctness runner for the CUDA Lang BatchedGemm port."""

import argparse
from dataclasses import dataclass

import torch

from .batched_gemm_config import ActKind, BatchMode, DType, TileScheduler, make_config
from .batched_gemm_kernel import build_batched_gemm_task_manager, gemm
from .batched_gemm_quant import kaiming_uniform_tensor as _kaiming_uniform_tensor


@dataclass(frozen=True)
class TokenLayout:
    """Generated-style MoE token layout for the batched token dimension."""

    tile_size: int
    cga_tile_size: int
    tokens_per_expert: list[int]
    padded_starts: list[int]
    expanded_to_expert: list[int]
    expanded_to_token: list[int]
    tile_idx: list[int]
    mn_limit: list[int]

    @property
    def total_padded_tokens(self) -> int:
        return self.padded_starts[-1]

    @property
    def num_token_tiles(self) -> int:
        return len(self.tile_idx)


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _make_token_layout(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    tile_size: int,
    cluster_dim_in_token: int,
) -> TokenLayout:
    """Build the MoE expanded/permuted token layout."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")

    cga_tile_size = tile_size * cluster_dim_in_token
    tokens_per_expert = [0] * num_experts
    assignments: list[tuple[int, int, int]] = []
    for token_idx in range(num_tokens):
        for k_idx in range(top_k):
            expert_idx = (token_idx * top_k + k_idx) % num_experts
            local_idx = tokens_per_expert[expert_idx]
            tokens_per_expert[expert_idx] += 1
            assignments.append((expert_idx, local_idx, token_idx))

    padded_starts = [0]
    for token_count in tokens_per_expert:
        padded_starts.append(padded_starts[-1] + _round_up(token_count, cga_tile_size))

    total_padded = padded_starts[-1]
    expanded_to_expert = [-1] * total_padded
    expanded_to_token = [0] * total_padded
    for expert_idx, local_idx, token_idx in assignments:
        expanded_idx = padded_starts[expert_idx] + local_idx
        expanded_to_expert[expanded_idx] = expert_idx
        expanded_to_token[expanded_idx] = token_idx

    num_token_tiles = total_padded // tile_size if total_padded > 0 else 1
    tile_idx = [0] * num_token_tiles
    mn_limit = [0] * num_token_tiles
    for expert_idx, token_count in enumerate(tokens_per_expert):
        padded_start = padded_starts[expert_idx]
        padded_end = padded_starts[expert_idx + 1]
        valid_end = padded_start + token_count
        for tile in range(padded_start // tile_size, padded_end // tile_size):
            tile_idx[tile] = expert_idx
            mn_limit[tile] = valid_end

    return TokenLayout(
        tile_size=tile_size,
        cga_tile_size=cga_tile_size,
        tokens_per_expert=tokens_per_expert,
        padded_starts=padded_starts,
        expanded_to_expert=expanded_to_expert,
        expanded_to_token=expanded_to_token,
        tile_idx=tile_idx,
        mn_limit=mn_limit,
    )


def _expand_bf16_activations(
    compact_activations: torch.Tensor,
    token_layout: TokenLayout,
) -> torch.Tensor:
    expanded = torch.zeros(
        (token_layout.total_padded_tokens, compact_activations.shape[1]),
        dtype=compact_activations.dtype,
        device=compact_activations.device,
    )
    for expanded_idx, expert_idx in enumerate(token_layout.expanded_to_expert):
        if expert_idx >= 0:
            token_idx = token_layout.expanded_to_token[expanded_idx]
            expanded[expanded_idx, :] = compact_activations[token_idx, :]
    return expanded


def reorder_rows_for_gated_act(tensor: torch.Tensor) -> torch.Tensor:
    """Interleave logical FC1 halves: [first-half, second-half] -> adjacent pairs."""
    if tensor.shape[-2] % 2:
        raise ValueError("SwiGLU weight rows must be even")
    half = tensor.shape[-2] // 2
    return torch.stack((tensor[..., :half, :], tensor[..., half:, :]), dim=-2).flatten(-3, -2)


def shuffle_matrix(tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """Shuffle swap-AB weight rows for the 16x256b epilogue."""
    block_size = 32 if epilogue_tile_m % 128 == 0 else 16
    if block_size > tensor.shape[-2]:
        return tensor.clone()
    group_size = block_size // 8
    out = torch.empty_like(tensor)
    for src_row in range(tensor.shape[-2]):
        block_start = src_row // block_size * block_size
        src_in_block = src_row % block_size
        dst_in_block = (src_in_block % group_size) * 8 + src_in_block // group_size
        out[..., block_start + dst_in_block, :] = tensor[..., src_row, :]
    return out


def unshuffle_matrix(tensor: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """Restore logical rows from the swap-AB physical weight layout."""
    block_size = 32 if epilogue_tile_m % 128 == 0 else 16
    if block_size > tensor.shape[-2]:
        return tensor.clone()
    group_size = block_size // 8
    out = torch.empty_like(tensor)
    for dst_row in range(tensor.shape[-2]):
        block_start = dst_row // block_size * block_size
        src_in_block = dst_row % block_size
        physical_in_block = (src_in_block % group_size) * 8 + src_in_block // group_size
        out[..., dst_row, :] = tensor[..., block_start + physical_in_block, :]
    return out


def prepare_tensors(
    *, num_experts=2, num_tokens=128, top_k=1, problem_n=64, problem_k=256, seed=42, cfg
):
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    token_layout = _make_token_layout(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=cfg.tile_n if cfg.is_swap_ab else cfg.tile_m,
        cluster_dim_in_token=1 if cfg.is_swap_ab else cfg.cluster_m,
    )
    torch.manual_seed(seed)
    # Keep source generation order: weights, compact activations, expansion.
    weights = _kaiming_uniform_tensor(
        (num_experts, problem_n, problem_k),
        fan_in=problem_k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    is_gated = cfg.act_kind == int(ActKind.SWIGLU)
    if is_gated:
        weights = reorder_rows_for_gated_act(weights)
    activation_compact_torch = _kaiming_uniform_tensor(
        (num_tokens, problem_k),
        fan_in=problem_k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    activations = _expand_bf16_activations(activation_compact_torch, token_layout)
    dtype_c = torch.bfloat16 if cfg.dtype_c == int(DType.BF16) else torch.float16
    if cfg.is_swap_ab:
        a = shuffle_matrix(weights, cfg.epi_tile_m)
        b = activations.unsqueeze(0)
        output_m = problem_n // 2 if is_gated else problem_n
        c_storage = torch.full(
            (output_m * token_layout.total_padded_tokens,),
            float("nan"), device="cuda", dtype=dtype_c,
        )
        c = torch.as_strided(
            c_storage,
            (output_m, token_layout.total_padded_tokens),
            (1, output_m),
        )
    else:
        a = activations.unsqueeze(0)
        b = weights
        c = torch.full(
            (token_layout.total_padded_tokens, problem_n // 2 if is_gated else problem_n),
            float("nan"), device="cuda", dtype=dtype_c,
        )
    tile_idx = torch.tensor(token_layout.tile_idx, device="cuda", dtype=torch.int32)
    mn_limit = torch.tensor(token_layout.mn_limit, device="cuda", dtype=torch.int32)
    return {"a": a, "b": b, "c": c, "tile_idx": tile_idx, "mn_limit": mn_limit, "cfg": cfg}


def run(tensors, stream=None):
    return gemm(**tensors, stream=stream)


def verify_output(tensors, *, atol=0.1, rtol=0.0):
    a, b, c, tile_idx, cfg = (
        tensors[key] for key in ("a", "b", "c", "tile_idx", "cfg")
    )
    expected = torch.empty_like(c)
    for tile, expert in enumerate(tile_idx.tolist()):
        if cfg.is_swap_ab:
            start = tile * cfg.tile_n
            logical_weights = unshuffle_matrix(a[expert], cfg.epi_tile_m)
            ref_gemm = (
                logical_weights.float()
                @ b[0, start:start + cfg.tile_n].float().T
            )
        else:
            start = tile * cfg.tile_m
            ref_gemm = (
                a[0, start:start + cfg.tile_m].float() @ b[expert].float().T
            )
        if cfg.act_kind == int(ActKind.SWIGLU):
            # Weights are physically interleaved; retain FP32 through activation.
            if cfg.is_swap_ab:
                gate, up = ref_gemm[0::2, :], ref_gemm[1::2, :]
            else:
                gate, up = ref_gemm[:, 0::2], ref_gemm[:, 1::2]
            ref_gemm = gate * up * torch.sigmoid(up)
        if cfg.is_swap_ab:
            expected[:, start:start + cfg.tile_n] = ref_gemm
        else:
            expected[start:start + cfg.tile_m] = ref_gemm
    torch.testing.assert_close(c, expected, atol=atol, rtol=rtol)


def reference_check(**kwargs):
    tensors = prepare_tensors(**kwargs)
    run(tensors)
    torch.cuda.synchronize()
    verify_output(tensors)
    return True


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--problem-n", type=int, default=64)
    parser.add_argument("--problem-k", type=int, default=256)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--epi-tile-n", type=int, default=8)
    parser.add_argument("--act-kind", choices=("none", "swiglu"), default="none")
    parser.add_argument("--swap-ab", action="store_true")
    parser.add_argument("--num-stages-a", type=int, default=3)
    parser.add_argument("--num-stages-b", type=int, default=3)
    parser.add_argument("--num-stages-tmem-acc", type=int, default=2)
    parser.add_argument(
        "--tile-scheduler", choices=("static", "persistent"), default="static"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        batch_mode=int(BatchMode.BATCH_N if args.swap_ab else BatchMode.BATCH_M),
        transpose_mma_output=int(args.swap_ab),
        tile_n=args.tile_n,
        mma_n=args.tile_n,
        tile_k=args.tile_k,
        mma_k=16,
        epi_tile_n=args.epi_tile_n,
        act_kind=int(ActKind.SWIGLU if args.act_kind == "swiglu" else ActKind.NONE),
        num_stages_a=args.num_stages_a,
        num_stages_b=args.num_stages_b,
        num_stages_tmem_acc=args.num_stages_tmem_acc,
        tile_scheduler=int(
            TileScheduler.PERSISTENT
            if args.tile_scheduler == "persistent"
            else TileScheduler.STATIC
        ),
    )
    if args.validate_only:
        token_layout = _make_token_layout(
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            tile_size=cfg.tile_n if cfg.is_swap_ab else cfg.tile_m,
            cluster_dim_in_token=1 if cfg.is_swap_ab else cfg.cluster_m,
        )
        build_batched_gemm_task_manager(
            cfg,
            (
                args.problem_n if cfg.is_swap_ab else token_layout.total_padded_tokens,
                token_layout.total_padded_tokens if cfg.is_swap_ab else args.problem_n,
                args.problem_k,
            ),
            verbose=True,
        )
    else:
        reference_check(
            cfg=cfg,
            num_experts=args.num_experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
            problem_n=args.problem_n,
            problem_k=args.problem_k,
            seed=args.seed,
        )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
