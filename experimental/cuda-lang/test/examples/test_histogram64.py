# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch


__doc__ = """
cuda.lang port of the `histogram64 <histogram64>`__ CUDA sample.

The goal is to reproduce the original program faithfully, even if it is
not the most Pythonic. The kernel structure and device functions stay
very close to the original to make the kernel easy to verify.

.. _histogram64: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/histogram/histogram64.cu
"""  # noqa: E501


HISTOGRAM64_BIN_COUNT = 64
SHARED_MEMORY_BANKS = 16
HISTOGRAM64_THREADBLOCK_SIZE = 4 * SHARED_MEMORY_BANKS
MERGE_THREADBLOCK_SIZE = 256
VECTOR_COUNT = 1024


@cl.function
def add_byte(s_ThreadBase, data):
    s_ThreadBase[cl.int32(data) * HISTOGRAM64_THREADBLOCK_SIZE] += cl.uint8(1)


@cl.function
def add_word(s_ThreadBase, data):
    add_byte(s_ThreadBase, (data >> 2) & 0x3F)
    add_byte(s_ThreadBase, (data >> 10) & 0x3F)
    add_byte(s_ThreadBase, (data >> 18) & 0x3F)
    add_byte(s_ThreadBase, (data >> 26) & 0x3F)


@cl.kernel
def histogram64_kernel(d_PartialHistograms, d_Data, dataCount):
    tx = cl.thread_index(0)
    bx = cl.block_index(0)
    bdx = cl.thread_count(0)
    gdx = cl.block_count(0)

    # Encode thread index to avoid bank conflicts in s_Hist[] access.
    threadPos = (
        ((tx & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0)
        | ((tx & (SHARED_MEMORY_BANKS - 1)) << 2)
        | ((tx & (SHARED_MEMORY_BANKS * 3)) >> 4)
    )

    s_Hist = cl.shared_array(
        shape=(HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT,),
        dtype=cl.uint8,
    )
    s_ThreadBase = cl.reinterpret_pointer_as_array(
        s_Hist.get_element_pointer((threadPos,)),
        cl.uint8,
        1,
    )

    for i in cl.static_iter(range(HISTOGRAM64_BIN_COUNT)):
        s_Hist[tx + i * HISTOGRAM64_THREADBLOCK_SIZE] = cl.uint8(0)

    cl.barrier_sync_block()

    for pos in range(bx * bdx + tx, dataCount, bdx * gdx):
        base = 4 * pos
        add_word(s_ThreadBase, d_Data[base + 0])
        add_word(s_ThreadBase, d_Data[base + 1])
        add_word(s_ThreadBase, d_Data[base + 2])
        add_word(s_ThreadBase, d_Data[base + 3])

    cl.barrier_sync_block()

    if tx < HISTOGRAM64_BIN_COUNT:
        s_HistBase = cl.reinterpret_pointer_as_array(
            s_Hist.get_element_pointer((tx * HISTOGRAM64_THREADBLOCK_SIZE,)),
            cl.uint8,
            1,
        )
        sum = cl.uint32(0)
        pos = 4 * (tx & (SHARED_MEMORY_BANKS - 1))

        for _ in cl.static_iter(range(HISTOGRAM64_THREADBLOCK_SIZE // 4)):
            sum = (
                sum
                + cl.uint32(s_HistBase[pos + 0])
                + cl.uint32(s_HistBase[pos + 1])
                + cl.uint32(s_HistBase[pos + 2])
                + cl.uint32(s_HistBase[pos + 3])
            )
            pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1)

        d_PartialHistograms[bx * HISTOGRAM64_BIN_COUNT + tx] = sum


@cl.kernel
def merge_histogram64_kernel(d_Histogram, d_PartialHistograms, histogramCount):
    tx = cl.thread_index(0)
    bx = cl.block_index(0)

    data = cl.shared_array(shape=(MERGE_THREADBLOCK_SIZE,), dtype=cl.uint32)
    sum = cl.uint32(0)

    for i in range(tx, histogramCount, MERGE_THREADBLOCK_SIZE):
        sum = sum + d_PartialHistograms[bx + i * HISTOGRAM64_BIN_COUNT]

    data[tx] = sum

    for stride in cl.static_iter([128, 64, 32, 16, 8, 4, 2, 1]):
        cl.barrier_sync_block()
        if tx < stride:
            data[tx] = data[tx] + data[tx + stride]

    if tx == 0:
        d_Histogram[bx] = data[0]


def i_div_up(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)


def i_snap_down(a, b):
    return a - a % b


def histogram64_cpu(words):
    histogram = torch.zeros(HISTOGRAM64_BIN_COUNT, dtype=torch.int64)
    for word in words.tolist():
        histogram[(word >> 2) & 0x3F] += 1
        histogram[(word >> 10) & 0x3F] += 1
        histogram[(word >> 18) & 0x3F] += 1
        histogram[(word >> 26) & 0x3F] += 1
    return histogram


def test_histogram64():
    generator = torch.Generator().manual_seed(1234)
    d_Data = (
        torch
        .randint(
            0,
            2**32,
            (VECTOR_COUNT * 4,),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )
        .to(torch.uint32)
        .cuda()
    )

    byteCount = d_Data.numel() * 4
    histogramCount = i_div_up(
        byteCount, HISTOGRAM64_THREADBLOCK_SIZE * i_snap_down(255, 16)
    )

    d_PartialHistograms = torch.zeros(
        histogramCount * HISTOGRAM64_BIN_COUNT,
        dtype=torch.uint32,
        device="cuda",
    )
    d_Histogram = torch.zeros(HISTOGRAM64_BIN_COUNT, dtype=torch.uint32, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        (histogramCount,),
        (HISTOGRAM64_THREADBLOCK_SIZE,),
        histogram64_kernel,
        (d_PartialHistograms, d_Data, VECTOR_COUNT),
    )
    cl.launch(
        torch.cuda.current_stream(),
        (HISTOGRAM64_BIN_COUNT,),
        (MERGE_THREADBLOCK_SIZE,),
        merge_histogram64_kernel,
        (d_Histogram, d_PartialHistograms, histogramCount),
    )
    torch.cuda.synchronize()

    expected = histogram64_cpu(d_Data.cpu().to(torch.int64))
    assert torch.equal(d_Histogram.cpu().to(torch.int64), expected)
