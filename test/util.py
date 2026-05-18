# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from io import BytesIO
import sys
import pytest
import torch
import numpy as np
from typing import Union, Optional
from math import ceil
import struct
from torch.testing import make_tensor
import cuda.tile as ct
import tempfile

from cuda.tile._compile import get_sm_arch
from cuda.tile._ir.typing_support import to_dtype

from cuda.tile import _datatype as datatype

from cuda.tile._exception import TileTypeError
from cuda.tile._cext import get_compute_capability
from cuda.tile.compilation import CallingConvention, KernelSignature

TensorLike = torch.Tensor
Scalar = Union[int, float]


def get_bytecode(kernel, kernel_args, sm_arch_func=get_sm_arch) -> bytes:
    if not isinstance(kernel, ct.kernel):
        kernel = ct.kernel(kernel)

    cconv = CallingConvention.cutile_python_v1()
    sig = KernelSignature.from_kernel_args(kernel, kernel_args, cconv)
    io = BytesIO()
    ct.compilation.export_kernel(kernel, [sig], io,
                                 gpu_code=sm_arch_func(), output_format="tileir_bytecode")
    return io.getvalue()


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


def benchmark_cudagraph_runner(f, args, kwargs):
    # For patching BenchmarkFixture._make_runner
    def runner(loops_range, **unused) -> float:
        # run the regular function a few times to ensure kernel and memory states are stable
        # before graph capture
        for _ in range(3):
            f(*args, **kwargs)

        # cuda graph capture must happen on non-default stream
        if torch.cuda.current_stream() == torch.cuda.default_stream():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            ev_start = torch.cuda.Event(enable_timing=True, external=True)
            ev_end = torch.cuda.Event(enable_timing=True, external=True)
            l2_size = torch.cuda.get_device_properties(0).L2_cache_size
            cache_flush_tensor = torch.empty(l2_size, dtype=torch.uint8, device="cuda")

            with torch.cuda.graph(g):
                cache_flush_tensor.zero_()
                ev_start.record()
                f(*args, **kwargs)
                ev_end.record()

            torch.cuda.synchronize()
            assert loops_range is not None
            ret = 0
            for _ in loops_range:
                g.replay()
                ev_end.synchronize()
                ret += ev_start.elapsed_time(ev_end)
            return ret / 1000  # secs
    return runner


def benchmark_eager_runner(f, args, kwargs):
    def runner(loops_range, **unused) -> float:
        assert loops_range is not None
        torch.cuda.synchronize()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        for _ in loops_range:
            f(*args, **kwargs)
        ev_end.record()
        ev_end.synchronize()
        return ev_start.elapsed_time(ev_end) / 1000
    return runner


def estimate_bench_iter(f, tuple_of_args, cudagraph=False):
    warmup_iter_guess = 5
    min_round_time_ms = 100
    rounds = 5
    warmup_rounds = 1
    runner = (benchmark_cudagraph_runner(f, tuple_of_args, {}) if cudagraph else
              benchmark_eager_runner(f, tuple_of_args, {}))
    time_per_iter = runner(range(warmup_iter_guess)) / warmup_iter_guess
    main_iter = max(min(ceil(min_round_time_ms / (time_per_iter * 1000)), 200), warmup_iter_guess)
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
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = origin


def is_ampere_or_ada():
    return get_compute_capability()[0] == 8


def is_hopper_or_newer():
    return get_compute_capability()[0] >= 9


def require_hopper_or_newer():
    return pytest.mark.skipif(not is_hopper_or_newer(),
                              reason="feature requires Hopper or newer")


def is_blackwell_or_newer():
    return get_compute_capability()[0] >= 10


def require_blackwell_or_newer():
    return pytest.mark.skipif(not is_blackwell_or_newer(),
                              reason="feature requires Blackwell or newer")


def make_test_tensor(shape, dtype, device):
    if dtype == torch.float8_e8m0fnu:
        return make_tensor(shape, dtype=torch.uint8, device=device).view(dtype)
    else:
        return make_tensor(shape, dtype=dtype, device=device)


class AtomicOp(Enum):
    XCHG = 0
    ADD = 1
    MAX = 2
    MIN = 3
    AND = 4
    OR = 5
    XOR = 6

    def is_bitwise(self):
        return self in {AtomicOp.AND, AtomicOp.OR, AtomicOp.XOR}


int_32_64_dtypes = [torch.uint32, torch.uint64, torch.int32, torch.int64]
float_32_64_dtypes = [torch.float32, torch.float64]
int_float_32_64_dtypes = int_32_64_dtypes + float_32_64_dtypes


def ref_atomic_arith(x, y, operation):
    if x.dtype in [torch.uint32, torch.uint64]:
        # Cast to float64 because torch cuda maximum, minimum do not support uint32/64
        ref_x = operation(x.to(torch.float64), y.to(torch.float64))
        ref_x = ref_x.to(x.dtype)
    else:
        ref_x = operation(x, y.to(x.dtype))
    ref_z = x.clone()
    return ref_x, ref_z


def ref_atomic_bitwise(x, y, operation):
    int_dtype = get_int_dtype_of_same_size(x.dtype)
    ref_x = operation(x.view(int_dtype), y.view(int_dtype)).view(x.dtype)
    ref_z = x.clone()
    return ref_x, ref_z


class FdCaptureRunner:
    def __init__(self, script_path, *args):
        self._proc = subprocess.Popen(
            [sys.executable, "-u", script_path, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def close(self):
        if not self.is_alive():
            return
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait()
        self._proc = None

    def run_cmd(self, *args: str) -> tuple[list[str], list[str]]:
        if not self.is_alive():
            raise RuntimeError("FdCaptureRunner is not running")
        buf = struct.pack("!I", len(args))
        for arg in args:
            encoded = arg.encode('utf-8')
            buf += struct.pack("!I", len(encoded)) + encoded
        self._proc.stdin.write(buf)
        self._proc.stdin.flush()
        return self._read_response(self._proc.stdout), self._read_response(self._proc.stderr)

    @staticmethod
    def get_cmd_args(stdin) -> list[str] | None:
        count_bytes = stdin.read(4)
        if not count_bytes:
            return None
        count = struct.unpack("!I", count_bytes)[0]
        args = []
        for _ in range(count):
            length = struct.unpack("!I", stdin.read(4))[0]
            args.append(stdin.read(length).decode('utf-8'))
        return args

    _BEGIN_MARKER = "# begin-snippet"
    _END_MARKER = "# end-snippet"

    @staticmethod
    def write_begin_marker() -> None:
        os.write(1, f"\n{FdCaptureRunner._BEGIN_MARKER}\n".encode('utf-8'))
        os.write(2, f"\n{FdCaptureRunner._BEGIN_MARKER}\n".encode('utf-8'))

    @staticmethod
    def write_end_marker() -> None:
        os.write(1, f"\n{FdCaptureRunner._END_MARKER}\n".encode('utf-8'))
        os.write(2, f"\n{FdCaptureRunner._END_MARKER}\n".encode('utf-8'))

    def _read_response(self, stream) -> list[str]:
        captured_output = []
        capturing = False
        while True:
            line = stream.readline()
            if not line:
                break
            decoded = line.decode('utf-8').rstrip('\r\n')
            if decoded == FdCaptureRunner._BEGIN_MARKER:
                capturing = True
            elif decoded == FdCaptureRunner._END_MARKER:
                break
            elif capturing and decoded:
                captured_output.append(decoded)
        return captured_output
