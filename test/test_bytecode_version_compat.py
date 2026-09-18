# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO

import pytest
import torch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import get_sm_arch
from cuda.tile._exception import TileUnsupportedFeatureError
from cuda.tile._ir2bytecode import _resolve_num_worker_warps
from cuda.tile._numeric_semantics import RoundingMode
from cuda.tile.compilation import CallingConvention, KernelSignature


def compile_with_version(kernel, args, version: str):
    cconv = CallingConvention.cutile_python_v1()
    sig = KernelSignature.from_kernel_args(kernel, args, cconv)
    ct.compilation.export_kernel(kernel, [sig], output_file=BytesIO(),
                                 gpu_code=get_sm_arch(), output_format="cubin",
                                 bytecode_version=version)


def tensor(dtype=torch.float32):
    return torch.zeros(64, dtype=dtype, device='cuda:0')


def test_atan2_requires_13_2():
    @ct.kernel
    def kernel(x, y, z):
        tx = ct.load(x, 0, shape=64)
        ty = ct.load(y, 0, shape=64)
        ct.store(z, 0, tile=ct.atan2(tx, ty))

    with pytest.raises(TileUnsupportedFeatureError, match=r"atan2 requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor(), tensor()), "13.1")


def test_tanh_rounding_mode_requires_13_2():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx, rounding_mode=RoundingMode.APPROX))

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"tanh rounding_mode=approx requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor()), "13.1")


def test_tanh_without_rounding_mode_works_on_13_1():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx))

    # Should not raise version error
    compile_with_version(kernel, (tensor(), tensor()), "13.1")


def test_exp_rounding_mode_requires_13_3():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.exp(tx, rounding_mode=RoundingMode.APPROX))

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"exp rounding_mode=approx requires tileiras 13\.3"):
        compile_with_version(kernel, (tensor(), tensor()), "13.2")


def test_exp_without_rounding_mode_works_on_13_1():
    @ct.kernel
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.exp(tx))

    # Should not raise version error
    compile_with_version(kernel, (tensor(), tensor()), "13.1")


def test_num_worker_warps_warns_below_13_3():
    @ct.kernel(num_worker_warps=8)
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=tx)

    match = r"num_worker_warps is ignored: requires tileiras 13\.3, but current version is 13\.1"
    with pytest.warns(UserWarning, match=match):
        compile_with_version(kernel, (tensor(), tensor()), "13.1")


@pytest.mark.parametrize("value", [1, 2, 16, 32])
def test_relaxed_num_worker_warps_requires_13_5(value):
    match = rf"num_worker_warps={value} requires tileiras 13\.5 or later"
    with pytest.raises(TileUnsupportedFeatureError, match=match):
        _resolve_num_worker_warps(value, BytecodeVersion.V_13_4)


@pytest.mark.parametrize("value", [1, 2, 4, 8, 16, 32])
def test_num_worker_warps_supported_values_on_13_5(value):
    assert _resolve_num_worker_warps(value, BytecodeVersion.V_13_5) == value


@pytest.mark.parametrize("value", [4, 8])
def test_num_worker_warps_supported_before_13_5(value):
    assert _resolve_num_worker_warps(value, BytecodeVersion.V_13_4) == value
