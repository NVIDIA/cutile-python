# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch

import cuda.lang as cl
from cuda.lang.compilation import CallingConvention, KernelSignature, TensorMapConstraint
from cuda.tile._cext import cconv_v3_enabled

from test.util import compile_kernel, require_hopper_or_newer

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}


@pytest.mark.parametrize(
    "memory_space",
    (
        cl.MemorySpace.GENERIC,
        cl.MemorySpace.GLOBAL,
        cl.MemorySpace.LOCAL,
    ),
)
@pytest.mark.parametrize("level", tuple(cl.PrefetchLevel))
@pytest.mark.parametrize(
    "eviction_priority",
    (None, *tuple(cl.CachePolicy)),
)
def test_prefetch(memory_space, level, eviction_priority):
    def kernel():
        address = cl.address_space_cast(
            cl.shared_array(1, cl.int8).pointer(),
            memory_space,
        )
        cl.prefetch(
            address,
            level=level,
            eviction_priority=eviction_priority,
        )

    if eviction_priority is not None and level == cl.PrefetchLevel.L1:
        raises = pytest.raises(
            Exception,
            match="Prefetch eviction priority is supported only for L2",
        )
        compile_kernel(kernel, raises=raises)
        return

    if eviction_priority in (
        cl.CachePolicy.L2_EVICT_FIRST,
        cl.CachePolicy.L2_EVICT_UNCHANGED,
    ):
        raises = pytest.raises(
            Exception,
            match=f"Invalid eviction priority {eviction_priority._name_}."
                  " Accepted values: L2_EVICT_NORMAL, L2_EVICT_LAST",
        )
        compile_kernel(kernel, raises=raises)
        return

    if eviction_priority is not None and memory_space != cl.MemorySpace.GLOBAL:
        expected_message = re.escape(
            f"Pointer in GLOBAL address space is required when eviction priority is specified;"
            f" received a {memory_space._name_} pointer instead.",
        )
        raises = pytest.raises(Exception, match=expected_message)
        compile_kernel(kernel, raises=raises)
        return

    space = {
        cl.MemorySpace.GENERIC: "",
        cl.MemorySpace.GLOBAL: ".global",
        cl.MemorySpace.LOCAL: ".local",
    }[memory_space]
    eviction = "" if eviction_priority is None else eviction_priority.value[2:]
    expected_ptx = f"prefetch{space}.{level.name}{eviction}"
    compile_kernel(
        kernel,
        assert_in_ptx=expected_ptx,
        **HOPPER_TARGET,
    )


def test_prefetch_uniform():
    def kernel():
        generic = cl.address_space_cast(
            cl.shared_array(1, cl.int8).pointer(),
            cl.MemorySpace.GENERIC,
        )
        cl.prefetch_uniform(generic)

    compile_kernel(
        kernel,
        assert_in_ptx="prefetchu.L1",
        **HOPPER_TARGET,
    )


@require_hopper_or_newer()
@pytest.mark.skipif(not cconv_v3_enabled(), reason="Tensor-map arguments require cconv3")
def test_prefetch_tensor_map():
    @cl.kernel
    def kernel(tensor_map):
        cl.prefetch_tensor_map(tensor_map)

    source = torch.empty(16, dtype=torch.int32, device="cuda")
    tensor_map = cl.tensor_map_tiled(source, 16)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor_map,))


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_prefetch_tensor_map_parameter():
    def kernel(tensor_map):
        cl.prefetch_tensor_map(tensor_map)

    compile_kernel(
        kernel,
        signature=KernelSignature(
            [TensorMapConstraint()],
            calling_convention=CallingConvention.cutile_python_v3(),
        ),
        assert_in_ptx="prefetch.tensormap",
        **HOPPER_TARGET,
    )
