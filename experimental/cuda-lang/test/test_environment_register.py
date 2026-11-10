# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import pytest
import torch

from cuda.lang._exception import InvalidValueError, TypeCheckingError
from cuda.lang.compilation import KernelSignature
from .util import make_symbolic_scalar


@pytest.mark.parametrize("register", (-1, 32))
def test_environment_register_out_of_range(register):
    @cl.kernel
    def kernel():
        cl.environment_register(register)

    with pytest.raises(InvalidValueError, match="Environment register index"):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_environment_register_requires_constant():
    @cl.kernel
    def kernel(register):
        cl.environment_register(register)

    signature = KernelSignature([make_symbolic_scalar(cl.int32)])
    with pytest.raises(TypeCheckingError, match="Expected an integer constant"):
        cl.compile_simt(kernel, [signature])


def test_environment_register_requires_integer():
    @cl.kernel
    def kernel():
        cl.environment_register(1.5)

    with pytest.raises(TypeCheckingError, match="Expected an integer constant"):
        cl.compile_simt(kernel, [KernelSignature([])])


@pytest.mark.parametrize("cooperative", (False, True))
def test_driver_grid_workspace(cooperative):
    @cl.kernel
    def kernel(out):
        header = cl._driver_grid_workspace()
        cl.static_assert(cl.dtype_of(header.valid) == cl.bool_)
        cl.static_assert(cl.dtype_of(header.workspace_size) == cl.uint32)
        cl.static_assert(
            cl.dtype_of(header.barrier) == cl.pointer_dtype(cl.uint32, cl.MemorySpace.GLOBAL)
        )
        out[0], out[1] = header.valid, header.workspace_size
        if header.valid:
            out[2] = header.barrier.load()
        else:
            out[2] = cl.bitcast(header.barrier, cl.uint64) != 0
        high = cl.uint64(cl.environment_register(1))
        low = cl.uint64(cl.environment_register(2))
        address = (high << 32) | low
        out[3] = address != 0
        if address != 0:
            workspace = cl.bitcast(address, cl.pointer_dtype(cl.uint32, cl.MemorySpace.GLOBAL))
            out[4], out[5] = workspace[0], workspace[1]

    out = torch.zeros(6, dtype=torch.uint32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,), cooperative=cooperative)
    valid, size, barrier, pointer_valid, raw_size, raw_barrier = out.cpu().tolist()
    if cooperative:
        # An ordinary launch may keep a nonnull workspace after a cooperative launch.
        assert valid
    assert (size, barrier) == (raw_size, raw_barrier)
