# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import operator

import cuda.lang as cl
import pytest
import torch


__doc__ = """
Port of softmax kernels from Karpathy's llm.c.

https://github.com/karpathy/llm.c/blob/master/dev/cuda/softmax_forward.cu
"""  # noqa: E501


N = 8192
C = 50257


def warp_reduce(val: cl.float32, op) -> cl.float32:
    for offset in cl.static_iter([16, 8, 4, 2, 1]):
        shuffled = cl.shfl_down_sync(val, offset)
        val = op(val, shuffled)
    return val


@dataclass(frozen=True)
class SoftmaxForwardKernel1:
    block_size: int = 128

    @cl.kernel
    def kernel(out, inp, n: cl.Constant[int], c: cl.Constant[int]):
        row = cl.block_index(0) * cl.thread_count(0) + cl.thread_index(0)

        if row >= n:
            return

        base = row * c

        maxval = cl.float32(-float("inf"))
        for j in range(c):
            maxval = cl._libdevice.fmaxf(maxval, inp[base + j])

        sumval = cl.float64(0.0)
        for j in range(c):
            expval = cl._libdevice.expf(inp[base + j] - maxval)
            out[base + j] = expval
            sumval += cl.float64(expval)

        for j in range(c):
            out[base + j] /= cl.float32(sumval)

    def __call__(self, out, inp, n, c):
        cl.launch(
            torch.cuda.current_stream(),
            ((n + self.block_size - 1) // self.block_size,),
            (self.block_size,),
            self.kernel,
            (out, inp, n, c),
        )


@dataclass(frozen=True)
class SoftmaxForwardKernel7:
    block_size: int = 256
    unroll_factor: int = 8

    def __post_init__(self):
        if self.block_size % 32 != 0:
            raise ValueError("block_size must be a multiple of warp size")

    @property
    def warps_per_block(self):
        return self.block_size // 32

    @cl.kernel
    def kernel(
        out,
        inp,
        n: cl.Constant[int],
        c: cl.Constant[int],
        block_size: cl.Constant[int],
        unroll_factor: cl.Constant[int],
        warps_per_block: cl.Constant[int],
    ):
        row = cl.block_index(0)
        tid = cl.thread_index(0)

        if row >= n:
            return

        warp_id = tid // cl.lane_count()
        lane_id = tid % cl.lane_count()
        row_base = row * c

        maxvals = cl.shared_array(shape=(warps_per_block,), dtype=cl.float32)
        sumvals = cl.shared_array(shape=(warps_per_block,), dtype=cl.float32)

        maxval = cl.float32(-float("inf"))
        for i in range(0, c, block_size * unroll_factor):
            for u in cl.static_iter(range(unroll_factor)):
                col = i + u * block_size + tid
                if col < c:
                    maxval = cl._libdevice.fmaxf(maxval, inp[row_base + col])
        maxval = warp_reduce(maxval, cl._libdevice.fmaxf)

        if lane_id == 0:
            maxvals[warp_id] = maxval
        cl.barrier_sync_block()

        if tid == 0:
            block_max = maxvals[0]
            for i in cl.static_iter(range(1, warps_per_block)):
                block_max = cl._libdevice.fmaxf(block_max, maxvals[i])
            maxvals[0] = block_max
        cl.barrier_sync_block()

        offset = maxvals[0]

        sumval = cl.float32(0.0)
        for i in range(0, c, block_size * unroll_factor):
            for u in cl.static_iter(range(unroll_factor)):
                col = i + u * block_size + tid
                if col < c:
                    output = cl._libdevice.expf(inp[row_base + col] - offset)
                    out[row_base + col] = output
                    sumval = sumval + output
        sumval = warp_reduce(sumval, operator.add)

        if lane_id == 0:
            sumvals[warp_id] = sumval
        cl.barrier_sync_block()

        if tid == 0:
            block_sum = sumvals[0]
            for i in cl.static_iter(range(1, warps_per_block)):
                block_sum = block_sum + sumvals[i]
            sumvals[0] = block_sum
        cl.barrier_sync_block()

        denom = sumvals[0]

        for i in range(0, c, block_size * unroll_factor):
            for u in cl.static_iter(range(unroll_factor)):
                col = i + u * block_size + tid
                if col < c:
                    out[row_base + col] = out[row_base + col] / denom

    def __call__(self, out, inp, n, c):
        cl.launch(
            torch.cuda.current_stream(),
            (n,),
            (self.block_size,),
            self.kernel,
            (
                out,
                inp,
                n,
                c,
                self.block_size,
                self.unroll_factor,
                self.warps_per_block,
            ),
        )


@pytest.mark.parametrize(
    "driver",
    (
        SoftmaxForwardKernel1(),
        SoftmaxForwardKernel7(),
    ),
    ids=lambda x: str(x)
)
def test_softmax_forward(driver):
    generator = torch.Generator(device="cpu").manual_seed(42)
    inp_cpu = torch.randn(
        (N, C),
        generator=generator,
        dtype=torch.float32,
    )

    inp = inp_cpu.reshape(N * C).contiguous().cuda()
    out = torch.empty_like(inp)

    driver(out, inp, N, C)
    torch.cuda.synchronize()

    actual = out.reshape(N, C).cpu()
    expected = torch.nn.functional.softmax(inp_cpu, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
