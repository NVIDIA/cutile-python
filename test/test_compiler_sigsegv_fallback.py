# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._cext import TileContext
from cuda.tile._compile import compile_tile
from cuda.tile._context import TileContextConfig
from cuda.tile._exception import TileCompilerExecutionError
from cuda.tile.compilation import ArrayConstraint, CallingConvention, KernelSignature


def test_sigsegv_with_occupancy_falls_back_to_auto(monkeypatch, tmp_path):
    @ct.kernel(occupancy=2)
    def kernel(x, y):
        t = ct.load(x, (0,), shape=(32,))
        ct.store(y, (0,), tile=t)

    constraint = ArrayConstraint(
        ct.float32,
        1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
        stride_constant=(1,),
        base_addr_divisible_by=16,
    )
    sig = KernelSignature(
        parameters=[constraint, constraint],
        calling_convention=CallingConvention.cutile_python_v1(),
    )

    seen = []

    def fake_compile_cubin(fname_bytecode, compiler_options, sm_arch, timeout_sec,
                           remarks_output_file=None):
        seen.append(compiler_options.occupancy)
        if compiler_options.occupancy is not None:
            raise TileCompilerExecutionError(-11, "", "--gpu-name sm_120 -O2 --lineinfo",
                                             "13.3")
        cubin_path = Path(fname_bytecode).with_suffix(".cubin")
        cubin_path.write_bytes(b"FAKE_CUBIN")
        return cubin_path

    monkeypatch.setattr("cuda.tile._compile.compile_cubin", fake_compile_cubin)
    context = TileContext(config=TileContextConfig(
        temp_dir=str(tmp_path),
        compiler_timeout_sec=None,
        enable_crash_dump=False,
        cache_dir=None,
        cache_size_limit=0,
    ))

    result = compile_tile(
        kernel._annotated_function,
        [sig],
        sm_arch="sm_120",
        compiler_options=kernel._compiler_options,
        context=context,
        bytecode_version=BytecodeVersion.V_13_3,
        return_cubin=True,
    )

    assert result.cubin == b"FAKE_CUBIN"
    assert seen == [2, None]
