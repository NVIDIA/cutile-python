# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import cuda_timer
import subprocess
import sys
import math
import tempfile
from functools import cache

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import _get_max_supported_bytecode_version


def pytest_addoption(parser):
    parser.addoption(
        "--error-on-import-skip",
        action="store_true",
        default=False,
        help="Treat import-related skips as errors",
    )


def pytest_configure(config):
    if config.getoption("error_on_import_skip", default=False):
        _original = pytest.importorskip

        def strict_importorskip(modname, *args, **kwargs):
            try:
                return _original(modname, *args, **kwargs)
            except pytest.skip.Exception as e:
                pytest.fail(f"Required import skipped: {e}")

        pytest.importorskip = strict_importorskip


@cache
def get_tileiras_version():
    return _get_max_supported_bytecode_version(tempfile.gettempdir())


def requires_tileiras(version: BytecodeVersion):
    """Skip test if tileiras version is lower than required."""
    current = get_tileiras_version()
    return pytest.mark.skipif(
        current < version,
        reason=f"Requires tileiras {version.major()}.{version.minor()}, "
               f"found {current.major()}.{current.minor()}"
    )


def dtype_id(dtype):
    match(dtype):
        case torch.float8_e4m3fn: return "f8e4m3fn"
        case torch.float8_e5m2: return "f8e5m2"
        case torch.float16: return "f16"
        case torch.bfloat16: return "bf16"
        case torch.float32: return "f32"
        case torch.float64: return "f64"
        case torch.int32: return "i32"
        case torch.int64: return "i64"
        case torch.bool: return "bool"
        case torch.complex32: return "c32"
        case torch.complex64: return "c64"
        case torch.complex128: return "c128"
        case torch.uint32: return "u32"
        case torch.uint64: return "u64"
        case torch.int16: return "i16"
        case torch.int8: return "i8"


def _size_suffix(_size):
    suffix = 1024 ** 4
    suffix_map = {
        1024 ** 4: "T",
        1024 ** 3: "G",
        1024 ** 2: "M",
        1024: "K",
        1: "",
    }
    while suffix > 0:
        if _size % suffix == 0:
            return f"{_size // suffix}{suffix_map[suffix]}"
        suffix //= 1024


def shape_id(shape):
    shape_tokens = [_size_suffix(x) for x in shape]
    return '-'.join(str(x) for x in shape_tokens)


def shape_size_id(shape):
    overall_size = math.prod(shape)
    shape_size_tokens = [_size_suffix(overall_size)]
    shape_size_tokens.extend([
        "x".join(_size_suffix(x) for x in shape)
    ])
    return '-'.join(str(x) for x in shape_size_tokens)


# TODO: add float64.
float_dtypes = [torch.float16, torch.bfloat16, torch.float32]
int_dtypes = [torch.int32, torch.int64, torch.int16, torch.int8]
bool_dtypes = [torch.bool]
uint_dtypes = [torch.uint32, torch.uint64]
arithmetic_dtypes = int_dtypes + uint_dtypes + float_dtypes + bool_dtypes


@pytest.fixture(params=float_dtypes, ids=dtype_id)
def float_dtype(request):
    return request.param


@pytest.fixture(params=int_dtypes, ids=dtype_id)
def int_dtype(request):
    return request.param


@pytest.fixture(params=bool_dtypes, ids=dtype_id)
def bool_dtype(request):
    return request.param


@pytest.fixture(params=uint_dtypes, ids=dtype_id)
def uint_dtype(request):
    return request.param


# ----- For pytest benchmark
@pytest.fixture
def benchmark(benchmark):
    # Patch benchmark fixture to use cuda timer
    benchmark._timer = cuda_timer.time
    return benchmark


@pytest.fixture(params=["cutile", "cutile_autotune", "torch"])
def backend(request):
    """A fixture to automatically find the corresponding cutile/torch implementation of
    the benchmark target.

    Examples:

        If the request function is named "bench_matmul", we will look for `torch_matmul`
        and `cutile_matmul` as different backend implementation to `matmul`.
    """
    func_name = request.function.__name__
    if not func_name.startswith("bench_"):
        raise RuntimeError(f"Benchmark function must starts with \"bench_\", got {func_name}")
    base_name = func_name[len("bench_"):]
    backend_name = f'{request.param}_{base_name}'
    if request.param == "cutile_autotune" and not hasattr(request.module, backend_name):
        pytest.skip(f"Backend '{backend_name}' not implemented in {request.module.__name__}")
    return getattr(request.module, backend_name)


def pytest_benchmark_update_machine_info(config, machine_info):
    # TODO: PyTorch version, Driver version, and SM version
    pass


def pytest_benchmark_update_json(config, benchmarks, output_json):
    """
    Automatically add throughput (TF/s) and bandwidth (GB/s)
    to the extra_info field in the pytest-benchmark JSON output,
    if 'flops' and 'bytes_rw' are present.
    """

    for bench in output_json["benchmarks"]:
        extra = bench.get("extra_info", {})
        mean_time = bench["stats"]["mean"]

        # Bandwidth: bytes_rw / mean_time -> GB/s
        if "bytes_rw" in extra and mean_time > 0:
            gb_s = float(extra["bytes_rw"]) / mean_time / 1e9
            extra["bandwidth_GBps"] = gb_s

        else:
            extra["bandwidth_GBps"] = None

        # Throughput: flop_count / mean_time -> TF/s
        if "flop_count" in extra and mean_time > 0:
            tf_s = float(extra["flop_count"]) / mean_time / 1e12
            extra["throughput_TFps"] = tf_s

        else:
            extra["throughput_TFps"] = None

        bench["extra_info"] = extra


@pytest.fixture(scope="session")
def numba_cuda():
    smoke_test = """
import numpy
from numba import cuda
cuda.to_device(numpy.ones(10))
"""
    result = subprocess.run([sys.executable, "-c", smoke_test],
                            capture_output=True)
    if result.returncode != 0:
        pytest.xfail(f"Numba smoke test failed {result.returncode}. Skip.")
    import numba
    return numba.cuda
