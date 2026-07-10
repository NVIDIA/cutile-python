# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
from .util import require_blackwell_or_newer


@pytest.mark.parametrize("src_dtype", [torch.float16, torch.float32])
def test_float_intrinsic(src_dtype):
    @cl.kernel
    def kern(x, y):
        res = cl._nvvm.add_rz_f(x[0], x[1])
        cl.static_assert(cl.dtype_of(res) == cl.float32)
        y[()] = res

    x = torch.tensor([3.0, 5.0], dtype=src_dtype, device="cuda")
    y = torch.zeros((), dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y))
    assert y.item() == 8.0


def test_float_intrinsic_invalid_implicit_cast():
    @cl.kernel
    def kern(x, y):
        res = cl._nvvm.add_rz_f(x[0], x[1])
        cl.static_assert(res.dtype == cl.float32)
        y[()] = res

    x = torch.tensor([3.0, 5.0], dtype=torch.float64, device="cuda")
    y = torch.zeros((), dtype=torch.float32, device="cuda")
    with pytest.raises(TypeCheckingError, match="cannot implicitly cast float64 to float32"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y))


@pytest.mark.parametrize("src_dtype",
                         [torch.int8, torch.int16, torch.int32,
                          torch.uint8, torch.uint16, torch.uint32])
def test_integer_arg_intrinsic(src_dtype):
    @cl.kernel
    def kern(x, y):
        res = cl._nvvm.i2f_rn(x[()])
        y[()] = res

    x = torch.tensor(17, dtype=src_dtype, device="cuda")
    y = torch.zeros((), dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, y))
    assert y.item() == 17.0


def test_any_pointer_arg_intrinsic():
    @cl.kernel
    def kern(x, y):
        smem = cl.shared_array(64, dtype=cl.int16, alignment=16)
        smem[cl.thread_index(0)] = x[cl.thread_index(0)]
        smem[cl.thread_index(0) + 32] = x[cl.thread_index(0) + 32]
        r = cl._nvvm.ldmatrix_sync_aligned_m8n8_x1_b16(
            smem.get_element_pointer(cl.thread_index(0) * 8))
        y[cl.thread_index(0)] = r

    x = torch.arange(64, dtype=torch.int16, device="cuda")
    y = torch.zeros(32, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x, y))

    for i, val in enumerate(y.tolist()):
        assert val == ((2*i + 1) << 16) + 2*i


@require_blackwell_or_newer()
def test_smem_pointer_arg_intrinsic():
    @cl.kernel
    def kern(y):
        smem = cl.shared_array(1, dtype=cl.int32)
        p = cl._nvvm.mapa_shared_cluster(smem.get_base_pointer(), 0)
        a = cl.reinterpret_pointer_as_array(p, cl.int32, (1,))
        a[0] = 13
        y[0] = smem[0]

    y = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (y,))
    assert y.tolist() == [13]


@require_blackwell_or_newer()
def test_generic_pointer_arg_intrinsic():
    @cl.kernel
    def kern(y):
        smem = cl.shared_array(1, dtype=cl.int32)
        p = cl._nvvm.mapa(smem.get_base_pointer(), 0)
        a = cl.reinterpret_pointer_as_array(p, cl.int32, (1,))
        a[0] = 13
        y[0] = smem[0]

    y = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (y,))
    assert y.tolist() == [13]
