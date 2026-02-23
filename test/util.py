# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
from contextlib import contextmanager

import pytest
import torch
import numpy as np
from typing import Union, Optional
from math import ceil
import cuda.tile as ct
import tempfile

from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._ir.typing_support import to_dtype

from cuda.tile import _datatype as datatype

from cuda.tile._exception import TileTypeError
from cuda.tile._compile import compile_tile
from cuda.tile._cext import get_compute_capability


TensorLike = torch.Tensor
Scalar = Union[int, float]


def get_bytecode(kernel, kernel_args) -> bytearray:
    pyfunc = kernel._pyfunc if isinstance(kernel, ct.kernel) else kernel
    return compile_tile(pyfunc, kernel_args, CompilerOptions()).bytecode


def jit_kernel(name: str, source: str, tmp_path, globals: dict = None):
    fname = tmp_path / f"{name}.py"
    with open(fname, 'w') as f:
        f.write(source)
    code = compile(source, fname, 'exec')
    exec_globals = {"ct": ct}
    if globals is not None:
        exec_globals.update(globals)
    exec(code, exec_globals)
    kernel = ct.kernel(exec_globals[name])
    return kernel


def launch_binary(kernel, x, y, z, tile: int):
    assert z.ndim >= 1 and z.ndim <= 3
    grid = tuple(map(lambda d: ceil(d / tile), z.shape))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z, tile))


def launch_unary(kernel, x, y, tile: int):
    assert y.ndim >= 1 and y.ndim <= 3
    grid = tuple(map(lambda d: ceil(d / tile), y.shape))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile))


def assert_close(actual: TensorLike, ref: Union[TensorLike, Scalar],
                 rtol: Optional[float] = None, atol: Optional[float] = None):
    if hasattr(ref, 'dtype'):
        assert actual.dtype == ref.dtype
    else:
        ref = torch.full_like(actual, ref)
    torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol, equal_nan=True)


def assert_equal(actual: TensorLike, ref: Union[TensorLike, Scalar]):
    assert_close(actual, ref, rtol=0, atol=0)


def get_ptr_16_byte_divisible_view(A: TensorLike):
    assert A.ndim == 1 and A.shape[0] > 16
    remainder = A.data_ptr() % 16
    if remainder == 0:
        return A
    return A[remainder:]


def get_ptr_16_byte_non_divisible_view(A: TensorLike):
    assert A.ndim == 1 and A.shape[0] > 16
    remainder = A.data_ptr() % 16
    if remainder != 0:
        return A
    return A[1:]


def torch_to_tf32(x: torch.Tensor):
    assert torch.is_floating_point(x)
    x_f32 = x.to(torch.float32)
    assert torch.all(torch.isfinite(x_f32))
    # fp32: 9 bits sign+expo + 23 bits mantissa
    # tf32: 9 bits sign+expo + 10 bits mantissa + 13bits zeros
    x_bits = x_f32.view(torch.int32).cpu().numpy()
    # LSB, Guard, Round, Sticky
    lsb = ((x_bits >> 13) & 1)
    guard = ((x_bits >> 12) & 1)
    round = ((x_bits >> 11) & 1)
    sticky = (x_bits & ((1 << 11) - 1)) != 0
    round_down = (guard == 0)
    round_down |= ((guard == 1) & (round == 0) & (sticky == 0) & (lsb == 0))
    mask = ~((1 << 13) - 1)
    x_down = (x_bits & mask).view(np.uint32)
    # since we checked the fp32 value is finite,
    # it is safe to add one bit mantissa wihtout overflow check
    x_up = x_down + (1 << 13)
    x_tf32_bits = np.where(round_down, x_down, x_up)
    return torch.tensor(x_tf32_bits,
                        dtype=torch.uint32,
                        device=x.device).view(torch.float32).view(x.shape).to(x.dtype)


@contextmanager
def raises_if(cond, exc_ty, match):
    if cond:
        with pytest.raises(exc_ty, match=match):
            yield
    else:
        yield


def raises_autocast_error(launch, from_ty, to_ty) -> bool:
    from_ty = to_dtype(from_ty)
    to_ty = to_dtype(to_ty)
    if not datatype.can_autocast_dtypes(from_ty, to_ty):
        msg = re.escape(
            f"Autocast from value of type {from_ty} to {to_ty} is not allowed. "
            f"Please perform explicit cast using `astype`."
        )
        with pytest.raises(TileTypeError, match=msg):
            launch()
        return True
    else:
        return False


def estimate_bench_iter(f, tuple_of_args):
    warmup_iter_guess = 5
    min_round_time_ms = 100
    rounds = 5
    warmup_rounds = 1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(warmup_iter_guess):
        f(*tuple_of_args)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / warmup_iter_guess

    main_iter = ceil(min_round_time_ms / elapsed)

    return warmup_rounds, main_iter, rounds


def _find_filecheck_bin() -> Optional[str]:
    filecheck_path = shutil.which("FileCheck")
    if filecheck_path:
        return filecheck_path
    raise FileNotFoundError("'FileCheck' not found")


def filecheck(bytecode_buf: bytearray, check_directive: str) -> None:
    mod = pytest.importorskip("cuda.tile_internal._internal_cext")
    mlir_text = mod.bytecode_to_mlir_text(bytecode_buf)

    filecheck_bin = _find_filecheck_bin()
    with (
        tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as check_file,
        tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as input_file
    ):
        check_file.write(check_directive)
        check_file.flush()
        input_file.write(mlir_text)
        input_file.flush()
        result = subprocess.run(
            [filecheck_bin, "--dump-input=always",
             "--input-file", input_file.name, check_file.name],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"FileCheck failed:\n{result.stderr}"


def get_int_dtype_of_same_size(t: torch.dtype) -> torch.dtype:
    match t:
        case torch.bool: return torch.bool
        case torch.float32: return torch.int32
        case torch.float64: return torch.int64
        case torch.int32: return torch.int32
        case torch.int64: return torch.int64
        case torch.uint32: return torch.int32
        case torch.uint64: return torch.int64
        case torch.int16: return torch.int16
        case torch.int8: return torch.int8
        case _: raise NotImplementedError()


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


@contextmanager
def torch_use_tf32_matmul():
    origin = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    yield
    torch.backends.cuda.matmul.fp32_precision = origin


def is_ampere_or_ada():
    major, _minor = get_compute_capability()
    return major == 8
