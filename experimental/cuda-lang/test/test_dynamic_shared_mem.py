# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest

import cuda.lang as cl
import torch

from cuda.lang.compilation import KernelSignature, ScalarConstraint
from cuda.lang._exception import TypeCheckingError
from cuda.tile._cext import _spy_on_cuLaunchKernel_begin, _spy_on_cuLaunchKernel_end


class LaunchSpy:
    def __init__(self):
        self.dynamic_smem_size = []

    def get_dynamic_smem_size(self) -> int:
        assert len(self.dynamic_smem_size) == 1
        return self.dynamic_smem_size[0]

    def __call__(self, f, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                 smem_bytes, stream):
        self.dynamic_smem_size.append(smem_bytes)


@contextmanager
def spy_on_kernel_launch():
    spy = LaunchSpy()
    _spy_on_cuLaunchKernel_begin(spy)
    try:
        yield spy
    finally:
        _spy_on_cuLaunchKernel_end()


def test_single_1d_array():
    @cl.kernel
    def kern(x, n):
        smem = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)

        i = cl.thread_index(0)
        smem[i] = x[i]
        x[i] = smem[i] + 12

    x = torch.ones((32,), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x, 33,))

    assert x[0] == 13
    assert spy.get_dynamic_smem_size() == 33 * 4


def test_single_1d_array_with_constant_shape():
    @cl.kernel
    def kern(x):
        smem = cl.shared_array(shape=(33,), dtype=cl.int32, dynamic=True)

        i = cl.thread_index(0)
        smem[i] = x[i]
        x[i] = smem[i] + 12

    x = torch.ones((32,), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x,))

    assert x[0] == 13
    assert spy.get_dynamic_smem_size() == 33 * 4


def test_dynamic_shared_alignment_size_program():
    @cl.kernel
    def kern(n):
        desc = cl.shared_array(shape=(16,), dtype=cl.uint8, dynamic=True, alignment=128)
        values = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)
        values[0] = cl.int32(desc[0])

    result = cl.compile_simt(
        kern,
        [KernelSignature((ScalarConstraint(dtype=cl.int32),))],
        gpu_name="sm_80",
        arch="compute_80",
        keep_mlir=True,
    )

    assert result.dyn_smem_size_program is not None
    assert result.dyn_smem_size_program.opcodes == [
        "Const",
        "Const",
        "KernelArgI32",
        "Mul",
        "Add",
    ]
    assert result.dyn_smem_size_program.op_attrs == [128, 4, 0]
    assert "alignment = 1024 : i64" in result.mlir


def test_dynamic_shared_alignment_cannot_exceed_initial_alignment():
    @cl.kernel
    def kern():
        smem = cl.shared_array(
            shape=(1,), dtype=cl.uint8, dynamic=True, alignment=2048
        )
        smem[0] = cl.uint8(0)

    with pytest.raises(TypeCheckingError, match="cannot exceed 1024"):
        cl.compile_simt(
            kern,
            [KernelSignature(())],
            gpu_name="sm_80",
            arch="compute_80",
        )


def test_dynamic_shared_alignment_runtime_round_up_size_program():
    @cl.kernel
    def kern(n):
        smem = cl.shared_array(
            shape=(n,), dtype=cl.uint8, dynamic=True, alignment=128
        )
        values = cl.shared_array(shape=(1,), dtype=cl.int32, dynamic=True)
        values[0] = cl.int32(smem[0])

    result = cl.compile_simt(
        kern,
        [KernelSignature((ScalarConstraint(dtype=cl.int32),))],
        gpu_name="sm_80",
        arch="compute_80",
    )

    assert result.dyn_smem_size_program is not None
    assert result.dyn_smem_size_program.opcodes == [
        "KernelArgI32",
        "RoundUpToPow2",
        "Const",
        "Add",
    ]
    assert result.dyn_smem_size_program.op_attrs == [0, 128, 4]


def test_dynamic_shared_alignment_pads_final_allocation_to_current_alignment():
    @cl.kernel
    def kern(n):
        header = cl.shared_array(
            shape=(1,), dtype=cl.uint8, dynamic=True, alignment=128
        )
        values = cl.shared_array(
            shape=(n,), dtype=cl.uint8, dynamic=True, alignment=4
        )
        values[0] = header[0]

    result = cl.compile_simt(
        kern,
        [KernelSignature((ScalarConstraint(dtype=cl.int32),))],
        gpu_name="sm_80",
        arch="compute_80",
    )

    assert result.dyn_smem_size_program is not None
    assert result.dyn_smem_size_program.opcodes == [
        "Const",
        "KernelArgI32",
        "RoundUpToPow2",
        "Add",
    ]
    assert result.dyn_smem_size_program.op_attrs == [128, 0, 4]


def test_dynamic_shared_alignment_runtime_round_up_launch():
    @cl.kernel
    def kern(x, n):
        smem = cl.shared_array(
            shape=(n,), dtype=cl.uint8, dynamic=True, alignment=128
        )
        values = cl.shared_array(shape=(1,), dtype=cl.int32, dynamic=True)
        smem[0] = cl.uint8(0)
        values[0] = 42
        x[0] = values[0] + cl.int32(smem[0])

    x = torch.zeros((1,), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x, 33,))

    assert x[0] == 42
    assert spy.get_dynamic_smem_size() == 128 + 4


def test_dynamic_1d_array_and_static_1d_array():
    @cl.kernel
    def kern(x, n):
        smem = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)
        smem_static = cl.shared_array(shape=(32,), dtype=cl.int32)

        i = cl.thread_index(0)
        smem[i] = x[i]
        smem_static[i] = smem[i] + 1
        x[i] = smem_static[i] + 12

    x = torch.ones((32,), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x, 33,))

    assert x[0] == 14
    assert spy.get_dynamic_smem_size() == 33 * 4


def test_two_1d_arrays_in_order():
    @cl.kernel
    def kern(x, y, n, m):
        smem1 = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)
        smem2 = cl.shared_array(shape=(m,), dtype=cl.int8, dynamic=True)

        i = cl.thread_index(0)

        smem1[i] = x[i]
        x[i] = smem1[i] + 12

        smem2[i] = y[i]
        y[i] = smem2[i] + 27

    x = torch.ones((32,), dtype=torch.int32, device="cuda")
    y = torch.ones((32,), dtype=torch.int8, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x, y, 33, 37))

    assert x[0] == 13
    assert y[0] == 28
    assert spy.get_dynamic_smem_size() == 33 * 4 + 37


def test_two_1d_arrays_out_of_order():
    @cl.kernel
    def kern(x, y, n, m):
        smem2 = cl.shared_array(shape=(m,), dtype=cl.int8, dynamic=True)
        smem1 = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)

        i = cl.thread_index(0)

        smem1[i] = x[i]
        smem2[i] = y[i]
        x[i] = smem1[i] + 12
        y[i] = smem2[i] + 27

    x = torch.ones((32,), dtype=torch.int32, device="cuda")
    y = torch.ones((32,), dtype=torch.int8, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (x, y, 33, 37))

    assert x[0] == 13
    assert y[0] == 28
    assert spy.get_dynamic_smem_size() == 33 * 4 + 37


def test_single_2d_array():
    @cl.kernel
    def kern(x, n, m):
        smem = cl.shared_array(shape=(n, m), dtype=cl.int32, dynamic=True)

        i, j = cl.thread_index(0), cl.thread_index(1)
        smem[i, j] = x[i, j]
        x[i, j] = smem[i, j] + 10 * i + 1000 * j

    x = torch.ones((16, 8), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (16, 8), kern, (x, 17, 11))

    assert spy.get_dynamic_smem_size() == 17 * 11 * 4

    x_np = x.cpu().numpy()
    for i in range(16):
        for j in range(8):
            assert x_np[i, j] == 1 + 10*i + 1000*j


def test_two_2d_arrays_out_of_order():
    @cl.kernel
    def kern(x, y, n, m):
        smem2 = cl.shared_array(shape=(m, n), dtype=cl.int16, dynamic=True)
        smem1 = cl.shared_array(shape=(n, m), dtype=cl.int32, dynamic=True)

        i, j = cl.thread_index(0), cl.thread_index(1)

        smem1[i, j] = x[i, j]
        x[i, j] = smem1[i, j] + 10 * i + 1000 * j

        smem2[j, i] = y[j, i]
        y[j, i] = cl.int16(smem2[j, i] + 10 * i + 1000 * j)

    x = torch.ones((16, 8), dtype=torch.int32, device="cuda")
    y = torch.ones((8, 16), dtype=torch.int16, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1, 1), (16, 8), kern, (x, y, 17, 11))

    assert spy.get_dynamic_smem_size() == 17 * 11 * 4 + 11 * 17 * 2

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    for i in range(16):
        for j in range(8):
            assert x_np[i, j] == 1 + 10*i + 1000*j
            assert y_np[j, i] == 1 + 10*i + 1000*j


def test_single_2d_array_static_second_dim():
    @cl.kernel
    def kern(x, n):
        smem = cl.shared_array(shape=(n, 11), dtype=cl.int32, dynamic=True)
        cl.static_assert(smem.shape[1] == 11)
        cl.static_assert(smem.strides[0] == 11)
        cl.static_assert(smem.strides[1] == 1)

        i, j = cl.thread_index(0), cl.thread_index(1)
        smem[i, j] = x[i, j]
        x[i, j] = smem[i, j] + 10 * i + 1000 * j

    x = torch.ones((16, 8), dtype=torch.int32, device="cuda")
    with spy_on_kernel_launch() as spy:
        cl.launch(torch.cuda.current_stream(), (1,), (16, 8), kern, (x, 17))

    assert spy.get_dynamic_smem_size() == 17 * 11 * 4

    x_np = x.cpu().numpy()
    for i in range(16):
        for j in range(8):
            assert x_np[i, j] == 1 + 10*i + 1000*j


def test_dynamic_kwarg_is_required():
    @cl.kernel
    def kern(n):
        cl.shared_array(shape=(n,), dtype=cl.int32)

    with pytest.raises(TypeCheckingError,
                       match="Shape must be constant when `dynamic` is False"):
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (33,))


def test_arbitrary_expression_is_not_allowed():
    @cl.kernel
    def kern(n):
        cl.shared_array(shape=(n + cl.thread_count(0),), dtype=cl.int32, dynamic=True)

    with pytest.raises(TypeCheckingError,
                       match="Size of shared array must be either"
                             " a constant or a kernel parameter"):
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kern, (33,))
