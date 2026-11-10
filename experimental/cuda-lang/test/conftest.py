# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import wraps

import pytest
import torch

cl = pytest.importorskip("cuda.lang", reason="Skipping cuda-lang test: module not found")


def pytest_addoption(parser):
    parser.addoption(
        "--cuda-lang-backend",
        choices=("mlir2cubin", "directnvvm"),
        default=None,
        help="Override the cuda.lang compilation backend for this test run",
    )


def pytest_configure(config):
    backend = config.getoption("--cuda-lang-backend")
    if backend is None:
        return

    import cuda.lang._compile as compile_module

    original_compile_simt = compile_module.compile_simt

    @wraps(original_compile_simt)
    def compile_simt(*args, **kwargs):
        kwargs["direct_nvvm"] = backend == "directnvvm"
        return original_compile_simt(*args, **kwargs)

    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setattr(compile_module, "compile_simt", compile_simt)
    patch.setattr(cl, "compile_simt", compile_simt)


@pytest.fixture(autouse=True)
def reset_torch_device():
    torch.cuda.set_device(0)
