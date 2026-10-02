# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run, validate, benchmark, or dump PTX for the SM120 FMHA context kernel."""

import argparse
import math
import os
from pathlib import Path

import cuda.lang as cl
from cuda.lang.compilation import KernelSignature
import torch
import torch.nn.functional as F

try:
    from .fmha_kernel import build_fmha_task_manager, get_fmha_kernel, make_kv_tensor_maps
    from .fmha_resources import FmhaConfig, SUPPORTED_TILES
except ImportError:
    from fmha_kernel import build_fmha_task_manager, get_fmha_kernel, make_kv_tensor_maps
    from fmha_resources import FmhaConfig, SUPPORTED_TILES


LOG2_E = math.log2(math.e)


def _validate_shapes(q_shape, k_shape, cfg):
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise ValueError("Q and K shapes must be (batch, sequence, heads, head_dim)")
    if q_shape[0] != k_shape[0] or q_shape[2:] != k_shape[2:]:
        raise ValueError("Q and K must have matching batch, head, and head dimensions")
    if q_shape[3] != cfg.head_dim:
        raise ValueError("config head_dim does not match the input shape")
    if q_shape[1] <= 0 or k_shape[1] <= 0:
        raise ValueError("sequence lengths must be positive")


def make_config(q_shape, dtype, is_causal, q_tile, kv_tile):
    """Build a specialization and select the causal grid order."""
    batch_size, seqlen_q, num_heads_q, head_dim = q_shape
    use_causal_head_fast_grid = False
    if is_causal and torch.cuda.is_available():
        q_tiles = math.ceil(seqlen_q / q_tile)
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        use_causal_head_fast_grid = q_tiles * batch_size * num_heads_q > sm_count
    return FmhaConfig(
        head_dim=head_dim,
        dtype=dtype,
        is_causal=is_causal,
        q_tile=q_tile,
        kv_tile=kv_tile,
        use_causal_head_fast_grid=use_causal_head_fast_grid,
    )


def validate_schedule(
    q_shape=(2, 1024, 8, 128),
    k_shape=(2, 1024, 8, 128),
    dtype="fp16",
    is_causal=True,
    q_tile=SUPPORTED_TILES[0],
    kv_tile=SUPPORTED_TILES[0],
):
    """Build the task manager with concrete bounds and validate its schedule."""
    cfg = make_config(q_shape, dtype, is_causal, q_tile, kv_tile)
    _validate_shapes(q_shape, k_shape, cfg)
    task_manager = build_fmha_task_manager(cfg, verbose=True)
    print("Schedule validation: PASSED")
    return task_manager


def prepare_tensors(q_shape, k_shape, dtype="fp16"):
    cfg = FmhaConfig(head_dim=q_shape[3], dtype=dtype)
    _validate_shapes(q_shape, k_shape, cfg)
    torch.manual_seed(1111)
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    return {
        "q": torch.randn(q_shape, device="cuda:0", dtype=torch_dtype),
        "k": torch.randn(k_shape, device="cuda:0", dtype=torch_dtype),
        "v": torch.randn(k_shape, device="cuda:0", dtype=torch_dtype),
        "o": torch.empty(q_shape, device="cuda:0", dtype=torch_dtype),
    }


def _tma_view(tensor, cfg):
    batch, sequence, heads, head_dim = tensor.shape
    return torch.as_strided(
        tensor,
        size=(
            cfg.tma_swizzle_chunk_elems,
            sequence,
            cfg.tma_swizzle_chunks,
            heads,
            batch,
        ),
        stride=(
            1,
            heads * head_dim,
            cfg.tma_swizzle_chunk_elems,
            head_dim,
            sequence * heads * head_dim,
        ),
    )


def _prepare_launch(
    tensors,
    cfg: FmhaConfig,
    softmax_scale=1.0,
    stream=None,
    *,
    verbose=False,
):
    q, k, v, o = (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["o"],
    )
    _validate_shapes(tuple(q.shape), tuple(k.shape), cfg)
    if v.shape != k.shape or o.shape != q.shape:
        raise ValueError("V must match K and O must match Q")
    expected_dtype = torch.float16 if cfg.dtype == "fp16" else torch.bfloat16
    if any(tensor.dtype != expected_dtype for tensor in (q, k, v, o)):
        raise ValueError("tensor dtype does not match the FMHA configuration")

    kernel = get_fmha_kernel(cfg, verbose=verbose)
    batch_size, seqlen_q, num_heads_q, _ = q.shape
    physical_q_tiles = math.ceil(seqlen_q / cfg.q_tile)
    if cfg.use_causal_head_fast_grid:
        grid = (physical_q_tiles * num_heads_q, batch_size, 1)
    else:
        grid = (physical_q_tiles, batch_size, num_heads_q)
    k_view = _tma_view(k, cfg)
    v_view = _tma_view(v, cfg)
    k_map, v_map = make_kv_tensor_maps(k_view, v_view, cfg)
    return (
        torch.cuda.current_stream() if stream is None else stream,
        grid,
        (cfg.block_threads, 1, 1),
        kernel,
        (
            q.flatten(),
            k_map,
            v_map,
            o.flatten(),
            seqlen_q,
            k.shape[1],
            num_heads_q,
            softmax_scale * LOG2_E,
            cfg.head_dim,
            cfg.q_tile,
            cfg.kv_tile,
            cfg.num_compute_warps,
            cfg.tma_swizzle_chunk_elems,
            cfg.dtype == "bf16",
            cfg.is_causal,
            cfg.use_causal_head_fast_grid,
        ),
    )


def run(tensors, cfg: FmhaConfig, softmax_scale=1.0, stream=None, *, verbose=False):
    cl.launch(*_prepare_launch(tensors, cfg, softmax_scale, stream, verbose=verbose))


def verify_output(tensors, cfg, softmax_scale=1.0, tolerance=5.0e-2):
    reference = F.scaled_dot_product_attention(
        tensors["q"].permute(0, 2, 1, 3),
        tensors["k"].permute(0, 2, 1, 3),
        tensors["v"].permute(0, 2, 1, 3),
        scale=softmax_scale,
        is_causal=cfg.is_causal,
    ).permute(0, 2, 1, 3)
    maximum_error = (tensors["o"].float() - reference.float()).abs().max().item()
    torch.testing.assert_close(
        tensors["o"].float(), reference.float(), atol=tolerance, rtol=5.0e-2
    )
    return maximum_error


def benchmark(tensors, cfg, softmax_scale=1.0, warmups=20, iterations=200):
    if warmups < 0 or iterations <= 0:
        raise ValueError("warmups must be nonnegative and iterations must be positive")
    launch = _prepare_launch(tensors, cfg, softmax_scale)
    for _ in range(warmups):
        cl.launch(*launch)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        cl.launch(*launch)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def dump_ptx(tensors, cfg, output, softmax_scale=1.0):
    """Compile the exact launch specialization and save its PTX."""
    launches = []
    original_launch = cl.launch

    def capture_launch(
        stream,
        block_count,
        thread_count,
        kernel,
        kernel_args,
        /,
        **launch_options,
    ):
        launches.append((kernel, kernel_args))

    cl.launch = capture_launch
    try:
        run(tensors, cfg, softmax_scale)
    finally:
        cl.launch = original_launch
    if len(launches) != 1:
        raise RuntimeError(f"expected one captured launch, got {len(launches)}")
    kernel, kernel_args = launches[0]
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_LANG_MLIR2CUBIN_FLAGS"] = f"--dump-ptx={output}"
    signature = KernelSignature.from_kernel_args(kernel, kernel_args)
    result = cl.compile_simt(
        kernel,
        [signature],
        compiler_options=kernel._compiler_options,
    )
    if not result.cubin or not output.is_file():
        raise RuntimeError(f"compilation did not produce {output}")
    output.with_suffix(".cubin").write_bytes(result.cubin)
    return output


def parse_shape(value):
    result = tuple(int(item.strip()) for item in value.split(","))
    if len(result) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated dimensions")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q_shape", type=parse_shape, default=(2, 512, 8, 128))
    parser.add_argument("--k_shape", type=parse_shape, default=(2, 512, 8, 128))
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--q_tile", type=int, choices=SUPPORTED_TILES, default=128)
    parser.add_argument("--kv_tile", type=int, choices=SUPPORTED_TILES, default=128)
    parser.add_argument("--softmax_scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=5.0e-2)
    parser.add_argument("--warmup_iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dump_ptx", type=Path)
    args = parser.parse_args()

    cfg = make_config(
        args.q_shape,
        args.dtype,
        args.is_causal,
        args.q_tile,
        args.kv_tile,
    )
    _validate_shapes(args.q_shape, args.k_shape, cfg)
    if args.validate_only:
        validate_schedule(
            args.q_shape,
            args.k_shape,
            args.dtype,
            args.is_causal,
            args.q_tile,
            args.kv_tile,
        )
        return
    tensors = prepare_tensors(args.q_shape, args.k_shape, args.dtype)
    if args.dump_ptx is not None:
        output = dump_ptx(tensors, cfg, args.dump_ptx, args.softmax_scale)
        print(f"PTX: {output}")
        return
    run(tensors, cfg, args.softmax_scale, verbose=args.verbose)
    torch.cuda.synchronize()
    if not args.skip_ref_check:
        error = verify_output(tensors, cfg, args.softmax_scale, args.tolerance)
        print(f"max_error={error}")
    execution_us = benchmark(
        tensors,
        cfg,
        args.softmax_scale,
        args.warmup_iterations,
        args.iterations,
    )
    causal_factor = 0.5 if cfg.is_causal else 1.0
    batch_size, seqlen_q, num_heads, head_dim = args.q_shape
    flops = (
        4.0
        * batch_size
        * num_heads
        * seqlen_q
        * args.k_shape[1]
        * head_dim
        * causal_factor
    )
    tflops = flops / (execution_us * 1.0e-6) / 1.0e12
    print(f"exec_time={execution_us:.2f}us  tflops={tflops:.2f}")
    print("PASS")


if __name__ == "__main__":
    main()
