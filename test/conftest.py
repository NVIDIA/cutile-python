# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import subprocess
import sys
import math
import tempfile
from functools import cache, partial

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import (
        _get_max_supported_bytecode_version,
        _SUPPORTED_VERSIONS,
        _find_compiler_bin)
from cuda.tile._cext import dev_features_enabled
from util import (require_blackwell_or_newer, require_hopper_or_newer,
                  benchmark_cudagraph_runner, benchmark_eager_runner)


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


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    print("Tile compiler path:", _find_compiler_bin().path)
    print("Dev features enabled:", dev_features_enabled())
    print("Bytecode version:", get_tileiras_version().as_string())


@cache
def get_tileiras_version():
    return _get_max_supported_bytecode_version(tempfile.gettempdir(),
                                               allow_dev=dev_features_enabled())


def requires_tileiras(version: BytecodeVersion):
    """Skip test if tileiras version is lower than required."""

    def vstr(v):
        return f"{v.major()}.{v.minor()}"

    if version not in _SUPPORTED_VERSIONS and not dev_features_enabled():
        return pytest.mark.skip(
            reason=f"Requires dev features enabled for version {vstr(version)}"
        )

    current = get_tileiras_version()
    return pytest.mark.skipif(
        current < version,
        reason=f"Requires tileiras {vstr(version)}, found {vstr(current)}"
    )


def dtype_id(dtype):
    match(dtype):
        case torch.float8_e4m3fn: return "f8e4m3fn"
        case torch.float8_e5m2: return "f8e5m2"
        case torch.float8_e8m0fnu: return "f8e8m0fnu"
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
uint_dtypes = [torch.uint8, torch.uint32, torch.uint64]
arithmetic_dtypes = int_dtypes + uint_dtypes + float_dtypes + bool_dtypes
float8_dtypes = [
    pytest.param(torch.float8_e5m2, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e4m3fn, marks=require_hopper_or_newer()),
    pytest.param(torch.float8_e8m0fnu, marks=(require_blackwell_or_newer(),
                                              requires_tileiras(BytecodeVersion.V_13_2))),
]


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


def patch_benchmark_fixture(benchmark):
    """Patch BenchmarkFixture to use custom runner: eager or cudagraph.
    Extends the `pedantic` method to take additional `cudagraph` argument.
    """

    benchmark._make_runner = benchmark_eager_runner

    def pedantic(original, *args, **kwargs):
        if 'cudagraph' in kwargs:
            cudagraph = kwargs.pop('cudagraph')
            if cudagraph:
                benchmark._make_runner = benchmark_cudagraph_runner
        return original(*args, **kwargs)

    benchmark.pedantic = partial(pedantic, benchmark.pedantic)


# ----- For pytest benchmark
@pytest.fixture
def benchmark(benchmark):
    patch_benchmark_fixture(benchmark)
    return benchmark


@pytest.fixture(params=["cutile", "torch"])
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
    return getattr(request.module, backend_name)


def pytest_benchmark_update_machine_info(config, machine_info):
    fields = ['name', 'compute_cap', 'driver_version',
              'memory.total', 'clocks.max.sm', 'clocks.max.mem', 'persistence_mode', 'power.limit']
    query_gpu = f"--query-gpu={','.join(fields)}"
    command = ["nvidia-smi", "-i", "0", query_gpu, "--format=csv,noheader"]
    result = subprocess.check_output(command)
    machine_info["gpu_info"] = dict(zip(fields, result.decode().strip().split(","), strict=True))


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


def get_cupy_or_skip():
    try:
        import cupy as cupy
    except ImportError:
        pytest.skip("Cupy not installed. Skip test.")
    return cupy


@pytest.fixture(scope="session")
def cupy():
    return get_cupy_or_skip()
