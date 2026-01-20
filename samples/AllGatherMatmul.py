# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating all-gather and matrix multiplication in a single kernel.

Run with:
    python AllGatherMatmul.py --correctness-check

Algorithm:
    Each rank has a local input tensor of size (M, K), and a weight tensor of size (K, N).
    We want to compute the output tensor of size (M * world_size, N), where each
    "slice" of size (M, N) is the result of the matrix multiplication of a peer input tensor
    and the weight tensor.
"""

import argparse
import random
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
import cuda.tile as ct


# cuTile kernel for gather then matmul
@ct.kernel
def gather_matmul_kernel(
    inp_list,
    w,
    out,
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    tile_k: ct.Constant[int],
):
    # Number of m tiles per peer
    peer_inp_size_m = inp_list[0].shape[0]
    num_tiles_m_per_peer = ct.cdiv(peer_inp_size_m, tile_m)
    num_tiles_k = ct.num_tiles(w, axis=0, shape=(tile_k, tile_n))

    # 0-dim maps to m_tile_idx, 1-dim maps to n_tile_idx, of out tensor
    m_tile_idx = ct.bid(0)
    n_tile_idx = ct.bid(1)

    # Which peer should this tile get input from?
    peer = m_tile_idx // num_tiles_m_per_peer
    # Select ct.Array from inp_list
    peer_inp = inp_list[peer]
    m_tile_idx_in_peer = m_tile_idx % num_tiles_m_per_peer

    # Initialize accumulator
    accumulator = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if peer_inp.dtype == ct.float32 else peer_inp.dtype

    for k in range(num_tiles_k):
        # Load remote input tile
        a = ct.load(
            peer_inp,
            index=(m_tile_idx_in_peer, k),
            shape=(tile_m, tile_k),
            padding_mode=zero_pad,
        ).astype(dtype)
        # Load weight tile
        b = ct.load(
            w,
            index=(k, n_tile_idx),
            shape=(tile_k, tile_n),
            padding_mode=zero_pad,
        ).astype(dtype)
        # Perform matrix multiplication
        accumulator = ct.mma(a, b, accumulator)

    # Cast result back to output dtype
    accumulator = ct.astype(accumulator, out.dtype)

    # Store result tile
    ct.store(out, index=(m_tile_idx, n_tile_idx), tile=accumulator)


# Host-side launcher for all-gather
def cutile_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
):
    handle = symm_mem.rendezvous(inp, group.group_name)
    world_size = handle.world_size
    inp_list = [
        handle.get_buffer(rank, inp.shape, inp.dtype, 0) for rank in range(world_size)
    ]

    # Allocate output tensor
    M = inp.shape[0]
    M_out = M * world_size
    N = w.shape[1]
    out = torch.empty(M_out, N, device=inp.device)

    assert inp.shape[1] == w.shape[0], "reduction dimension mismatch"
    K = inp.shape[1]
    tile_m = 128
    tile_n = 128
    tile_k = 128
    assert M % tile_m == 0
    assert N % tile_n == 0
    assert K % tile_k == 0

    # Map each output tile to a block
    grid = (ct.cdiv(M_out, tile_m), ct.cdiv(N, tile_n),)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gather_matmul_kernel,
        (inp_list, w, out, tile_m, tile_n, tile_k),
    )

    return out


# Reference gather then matmul implementation
def ref_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
):
    world_size = dist.get_world_size(group)
    ag_scratch = torch.empty((world_size * inp.shape[0], inp.shape[1]), device=inp.device)
    dist.all_gather_into_tensor(ag_scratch, inp, group=group)
    out = ag_scratch @ w
    return out


def test(rank: int, world_size: int, args: argparse.Namespace, port: int):
    print(f"Rank {rank} of {world_size} is initializing")
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    torch.manual_seed(rank + 52)

    bs = 256
    hid = 1024
    out_hid = 512
    ref_inp = torch.rand((bs, hid), device=device)
    inp = symm_mem.empty(bs, hid, device=device).copy_(ref_inp)
    w = torch.rand((hid, out_hid), device=device)

    # Make sure all ranks have initialized their inputs
    dist.barrier(group)

    out = cutile_gather_matmul(inp, w, group)

    if args.correctness_check:
        expected_out = ref_gather_matmul(ref_inp, w, group)
        torch.testing.assert_close(
            out,
            expected_out,
            atol=1e-3,
            rtol=1e-3,
            msg=f"Rank {rank} of {world_size}: Correctness check failed",
        )
        print(f"Rank {rank} of {world_size}: Correctness check passed")
    else:
        if rank == 0:
            print("Correctness check disabled")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    # IP port number for multi-process rendezvous
    port = random.randint(30000, 60000)
    world_size = torch.cuda.device_count()
    mp.spawn(test, args=(world_size, args, port), nprocs=world_size, join=True)
