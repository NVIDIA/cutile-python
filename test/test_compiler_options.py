# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import cuda.tile as ct
import pytest
import torch

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compiler_options import CompilerOptions
from util import filecheck, get_bytecode


def test_invalid_target_name():
    err = r"Invalid GPU architecture name: sm100, expected sm_<major><minor>"
    with pytest.raises(ValueError, match=err):
        ct.ByTarget(sm100=4)


def _dummy():
    pass


@pytest.mark.parametrize("value", [None, 1, 2, 4, 8, 16, 32])
def test_num_worker_warps_accepts_valid(value):
    ct.kernel(_dummy, num_worker_warps=value)


@pytest.mark.parametrize("value", [0, 3, 7, 10, 64])
def test_num_worker_warps_rejects_invalid(value):
    with pytest.raises(ValueError,
                       match=r"num_worker_warps should be a power of 2 in \[1, 32\]"):
        ct.kernel(_dummy, num_worker_warps=value)


def test_num_worker_warps_accepts_by_target():
    ct.kernel(_dummy, num_worker_warps=ct.ByTarget(sm_100=8, default=4))


@pytest.mark.parametrize(
    "num_ctas, expected_sm_100, expected_default",
    [
        (8, None, 8),
        (ct.ByTarget(sm_100=8, default=2), 8, 2),
        (ct.ByTarget(sm_100=8), 8, None),
        (ct.ByTarget(default=2), None, 2),
    ],
)
def test_hints_by_target_single_field(num_ctas, expected_sm_100, expected_default):
    result = CompilerOptions(num_ctas=num_ctas).hints_by_target()
    assert result.get("sm_100", {}).get("num_ctas") == expected_sm_100
    assert result.get("default", {}).get("num_ctas") == expected_default


def test_hints_by_target_multiple_fields():
    result = CompilerOptions(
        num_ctas=ct.ByTarget(sm_100=8, default=2),
        occupancy=4,
        num_worker_warps=ct.ByTarget(sm_120=8),
        opt_level=ct.ByTarget(sm_100=0)
    ).hints_by_target()
    expected = {
        "sm_100": {"num_ctas": 8, "opt_level": 0},
        "sm_120": {"num_worker_warps": 8},
        "default": {"num_ctas": 2, "occupancy": 4, "num_worker_warps": None, "opt_level": 3}
    }
    assert result == expected


@pytest.mark.parametrize(
    "opt_level, target, expected",
    [
        (1, "sm_100", 1),
        (ct.ByTarget(sm_100=2, default=1), "sm_100", 2),
        (ct.ByTarget(sm_100=2, default=1), "sm_90", 1),
        (ct.ByTarget(default=1), "sm_90", 1),
    ],
)
def test_opt_level_for_target(opt_level, target, expected):
    assert CompilerOptions(opt_level=opt_level).opt_level_for_target(target) == expected


def test_opt_level_for_target_default():
    assert CompilerOptions().opt_level_for_target("sm_100") == 3


def _tensor():
    return torch.zeros(64, dtype=torch.float32, device='cuda:0')


def _kernel_body(x, y):
    tx = ct.load(x, 0, shape=64)
    ct.store(y, 0, tile=tx)


def _force_bytecode_version(monkeypatch, version: BytecodeVersion):
    monkeypatch.setattr(
        "cuda.tile._compile._get_max_supported_bytecode_version",
        lambda *args, **kwargs: version,
    )


@pytest.mark.use_mlir
@pytest.mark.parametrize(
    "num_ctas, expected_sm_100, expected_default",
    [
        (8, None, 8),
        (ct.ByTarget(sm_100=8, default=2), 8, 2),
        (ct.ByTarget(sm_100=8), 8, None),
        (ct.ByTarget(default=2), None, 2),
    ],
)
@pytest.mark.parametrize("forced_version", [BytecodeVersion.V_13_2, BytecodeVersion.V_13_3])
def test_hints_single_field_emission(num_ctas, expected_sm_100, expected_default,
                                     forced_version, monkeypatch):
    _force_bytecode_version(monkeypatch, forced_version)
    kernel = ct.kernel(_kernel_body, num_ctas=num_ctas)
    bytecode = get_bytecode(kernel, (_tensor(), _tensor()), lambda: "sm_100")

    parts = []
    if forced_version >= BytecodeVersion.V_13_3:
        if expected_sm_100 is not None:
            parts += ["// CHECK-DAG: sm_100 = {{.*}}" + f"num_cta_in_cga = {expected_sm_100}"]
        if expected_default is not None:
            parts += ["// CHECK-DAG: default = {{.*}}" + f"num_cta_in_cga = {expected_default}"]
    else:
        resolved = expected_sm_100 if expected_sm_100 is not None else expected_default
        parts += [
            "// CHECK-NOT: default = ",
            "// CHECK: sm_100 = {{.*}}" + f"num_cta_in_cga = {resolved}",
            "// CHECK-NOT: default = ",
        ]
    filecheck(bytecode, "\n".join(parts))


@pytest.mark.use_mlir
@pytest.mark.parametrize("forced_version", [BytecodeVersion.V_13_2, BytecodeVersion.V_13_3])
def test_hints_multiple_fields_emission(forced_version, monkeypatch):
    _force_bytecode_version(monkeypatch, forced_version)
    kernel = ct.kernel(
        _kernel_body,
        num_ctas=ct.ByTarget(sm_100=8, default=2),
        occupancy=4,
    )
    bytecode = get_bytecode(kernel, (_tensor(), _tensor()), lambda: "sm_100")
    if forced_version >= BytecodeVersion.V_13_3:
        parts = [
            "// CHECK-DAG: sm_100 = {{.*}}num_cta_in_cga = 8",
            "// CHECK-DAG: default = {{.*}}num_cta_in_cga = 2{{.*}}occupancy = 4",
        ]
    else:
        parts = [
            "// CHECK-NOT: default = ",
            "// CHECK: sm_100 = {{.*}}num_cta_in_cga = 8{{.*}}occupancy = 4",
            "// CHECK-NOT: default = ",
        ]
    filecheck(bytecode, "\n".join(parts))
