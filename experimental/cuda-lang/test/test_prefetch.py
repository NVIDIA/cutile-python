# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature

from test.util import compile_kernel, make_symbolic_tensor

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}

SIG_I32 = KernelSignature([make_symbolic_tensor((16,), cl.int32)])


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
            cl.shared_array(1, cl.int8).get_base_pointer(),
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
            match="Prefetch eviction priority must be L2_EVICT_NORMAL or L2_EVICT_LAST",
        )
        compile_kernel(kernel, raises=raises)
        return

    if eviction_priority is not None and memory_space != cl.MemorySpace.GLOBAL:
        raises = pytest.raises(
            Exception,
            match="cache eviction priority requires a global pointer",
        )
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
            cl.shared_array(1, cl.int8).get_base_pointer(),
            cl.MemorySpace.GENERIC,
        )
        cl.prefetch_uniform(generic)

    compile_kernel(
        kernel,
        assert_in_ptx="prefetchu.L1",
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize("predicated", (False, True))
def test_prefetch_tensor_map(predicated):
    def kernel(x):
        tensor_map = cl.tensor_map_tiled(x, 16)
        predicate = cl.thread_index(0) == 0 if predicated else None
        cl.prefetch_tensor_map(tensor_map, predicate=predicate)

    compile_kernel(
        kernel,
        signature=SIG_I32,
        assert_in_ptx="prefetch.tensormap",
        **HOPPER_TARGET,
    )
