# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
# Move cutile types to the top level?
import cuda.tile as ct

from pathlib import Path
from math import ceil
from util import assert_equal, jit_kernel, require_rubin_cc107
from conftest import float_dtypes, int_dtypes, bool_dtypes, dtype_id, get_cupy_or_skip, \
                     requires_tileiras
from cuda.tile._exception import TileTypeError
from dataclasses import dataclass
from cuda.tile._bytecode.version import BytecodeVersion


@dataclass
class DTypeStr:
    numpy: str
    cutile: str


torch_to_dtype_str = {
    torch.float64: DTypeStr("np.float64", "ct.float64"),
    torch.float32: DTypeStr("np.float32", "ct.float32"),
    torch.float16: DTypeStr("np.float16", "ct.float16"),
    torch.bfloat16: DTypeStr(None, "ct.bfloat16"),
    torch.int64: DTypeStr("np.int64", "ct.int64"),
    torch.int32: DTypeStr("np.int32", "ct.int32"),
    torch.bool: DTypeStr("np.bool_", "ct.bool_"),
    torch.int16: DTypeStr("np.int16", "ct.int16"),
    torch.int8: DTypeStr("np.int8", "ct.int8"),
    # Add other dtypes as needed
}

value_call_kernel_template = """
def {name}(x, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.full((TILE,), {value_call}({value}), {dtype})
    ct.store(x, index=(bidx,), tile=tx)"""


def value_call_full_kernel(name: str, value_call: str, value: str, dtype: str,
                           tmp_path: Path, globals: dict):
    source = value_call_kernel_template.format(name=name,
                                               value_call=value_call,
                                               value=value,
                                               dtype=dtype)
    return jit_kernel(name, source, tmp_path, globals)


@pytest.mark.parametrize("dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("value", [1, 1.5, None])
@pytest.mark.parametrize("use_cupy", [True, False])
def test_full_np_value_call(dtype, value, use_cupy, tmp_path: Path):
    if dtype == torch.bfloat16:
        pytest.skip("bfloat16 is not supported in NumPy")
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    dtype_str = torch_to_dtype_str[dtype].numpy
    if use_cupy:
        dtype_str = dtype_str.replace("np.", "cp.")
        globals = {"cp": get_cupy_or_skip()}
    else:
        globals = {"np": np}
    value_str = str(value) if value is not None else ""
    kernel = value_call_full_kernel("create_full_value_call",
                                    dtype_str, value_str, dtype_str, tmp_path, globals)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    torch_value = value if value is not None else 0
    assert_equal(x, torch.full(shape, torch_value, dtype=dtype, device=x.device))


@pytest.mark.parametrize("dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("value", [1, 1.5, None])
def test_full_cutile_value_call(dtype, value, tmp_path: Path):
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    cutile_dtype_str = torch_to_dtype_str[dtype].cutile
    value_str = str(value) if value is not None else ""
    kernel = value_call_full_kernel("create_full_value_call",
                                    cutile_dtype_str, value_str, cutile_dtype_str,
                                    tmp_path, globals={"ct": ct})
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    torch_value = value if value is not None else 0
    assert_equal(x, torch.full(shape, torch_value, dtype=dtype, device=x.device))


@pytest.mark.parametrize("invalid_value", ["(1, 2, 3)", "'string'"])
def test_full_value_invalid_call(invalid_value, tmp_path: Path):
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    dtype = torch.float32
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    np_dtype_str = "np.float32"
    kernel = value_call_full_kernel("create_full_value_call",
                                    np_dtype_str, invalid_value, np_dtype_str,
                                    tmp_path, globals={"np": np})
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))


def test_full_value_invalid_torch_call(tmp_path: Path):
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    dtype = torch.float32
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    kernel = value_call_full_kernel("create_full_value_call",
                                    str(dtype), "1.0", str(dtype),
                                    tmp_path, globals={"torch": torch})
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))


create_full_kernel_template = """
def {name}(x, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.full((TILE,), {value}, {dtype})
    ct.store(x, index=(bidx,), tile=tx)"""


def create_full_kernel(name: str, value: str, dtype: str, tmp_path: Path, globals: dict):
    source = create_full_kernel_template.format(name=name,
                                                value=value,
                                                dtype=dtype)
    return jit_kernel(name, source, tmp_path, globals)


@pytest.mark.parametrize("value_dtype", [
    ("1.0", torch.float64),
    ("1.0", torch.float32),
    ("np.inf", torch.float32),
    ("float('inf')", torch.float32),
    ("float('-inf')", torch.float32),
    ("1.0", torch.float16),
    ("1", torch.int64),
    ("1", torch.int32),
    ("True", torch.bool)])
@pytest.mark.parametrize("use_cupy", [True, False])
def test_full_np_dtype(value_dtype, use_cupy: bool, tmp_path: Path):
    value_str, dtype = value_dtype
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    dtype_str = torch_to_dtype_str[dtype].numpy
    if use_cupy:
        dtype_str = dtype_str.replace("np.", "cp.")
        if value_str == "np.inf":
            value_str = "cp.inf"
        globals = {"cp": get_cupy_or_skip()}
    else:
        globals = {"np": np}
    kernel = create_full_kernel("create_full_np_dtype", value_str, dtype_str,
                                tmp_path, globals)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    if "-inf" in value_str:
        torch_value = -np.inf
    elif "inf" in value_str:
        torch_value = np.inf
    else:
        torch_value = 1
    assert_equal(x, torch.full(shape, torch_value, dtype=dtype, device=x.device))


@pytest.mark.parametrize("value_dtype", [
    ("1.0", torch.float64),
    ("1.0", torch.float32),
    ("1.0", torch.float16),
    ("1.0", torch.bfloat16),
    ("1", torch.int64),
    ("1", torch.int32),
    ("True", torch.bool)])
def test_full_torch_dtype(value_dtype, tmp_path: Path):
    value_str, dtype = value_dtype
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    kernel = create_full_kernel("create_full_torch_dtype", value_str, str(dtype),
                                tmp_path, globals={"torch": torch})
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    assert_equal(x, torch.full(shape, 1, dtype=dtype, device=x.device))


@pytest.mark.parametrize("value_dtype", [
    ("1.0", torch.float64),
    ("1.0", torch.float32),
    ("1.0", torch.float16),
    ("1.0", torch.bfloat16),
    ("1", torch.int64),
    ("1", torch.int32),
    ("True", torch.bool)])
def test_full_cutile_dtype(value_dtype, tmp_path: Path):
    value_str, dtype = value_dtype
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    cutile_dtype_str = torch_to_dtype_str[dtype].cutile
    kernel = create_full_kernel("create_full_cutile_dtype", value_str, cutile_dtype_str,
                                tmp_path, globals={"ct": ct})
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    assert_equal(x, torch.full(shape, 1, dtype=dtype, device=x.device))


@ct.kernel
def full_f8e5m3fnu_kernel(x, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.full((TILE,), 1.5, ct.float8_e5m3fnu)
    tx = ct.astype(tx, ct.float32)
    ct.store(x, index=(bidx,), tile=tx)


@require_rubin_cc107()
@requires_tileiras(BytecodeVersion.V_13_4)
def test_full_f8e5m3fnu():
    shape = (256,)
    tile = (128,)
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    x = torch.zeros(shape, dtype=torch.float32, device="cuda")

    ct.launch(torch.cuda.current_stream(), grid, full_f8e5m3fnu_kernel, (x, tile[0]),)
    assert_equal(x, torch.full(shape, 1.5, dtype=torch.float32, device="cuda"))


create_ones_zeros_kernel_template = """
def {name}(x, TILE: ct.Constant[int]):
    bidx = ct.bid(0)
    tx = ct.{value}((TILE,), {dtype})
    ct.store(x, index=(bidx,), tile=tx)"""


def create_ones_zeros_kernel(name: str, value: str, dtype: str, tmp_path: Path,
                             globals: dict | None = None):
    source = create_ones_zeros_kernel_template.format(name=name,
                                                      value=value,
                                                      dtype=dtype)
    return jit_kernel(name, source, tmp_path, globals)


@pytest.mark.parametrize("dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_ones(dtype, tmp_path: Path):
    shape = (256,)
    tile = (128,)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    cutile_dtype_str = torch_to_dtype_str[dtype].cutile
    kernel = create_ones_zeros_kernel("create_ones_cutile_dtype", "ones", cutile_dtype_str,
                                      tmp_path)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    assert_equal(x, torch.ones(shape, dtype=dtype, device=x.device))


@pytest.mark.parametrize("dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_zeros(dtype, tmp_path: Path):
    shape = (256,)
    tile = (128,)
    x = torch.zeros(shape, dtype=dtype, device='cuda')
    grid = (ceil(shape[0] / tile[0]), 1, 1)
    cutile_dtype_str = torch_to_dtype_str[dtype].cutile
    kernel = create_ones_zeros_kernel("create_zeros_cutile_dtype", "zeros", cutile_dtype_str,
                                      tmp_path)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile[0]))
    assert_equal(x, torch.zeros(shape, dtype=dtype, device=x.device))


@ct.kernel
def full_scalar_shape(x):
    tx = ct.full(2, fill_value=0.0, dtype=ct.float16)
    ct.store(x, 0, tx)


def test_scalar_shape():
    x = torch.zeros((2,), dtype=torch.float16, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,),
              full_scalar_shape, (x,))
