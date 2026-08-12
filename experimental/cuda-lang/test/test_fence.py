# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature
from cuda.lang._enums import MemoryScope, MemoryOrder
from cuda.lang._exception import TypeCheckingError, CompilerExecutionError
from test.util import make_symbolic_tensor, compile_kernel

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}

MEMORY_SCOPE_PTX = {
    MemoryScope.BLOCK: "cta",
    MemoryScope.CLUSTER: "cluster",
    MemoryScope.DEVICE: "gpu",
    MemoryScope.SYS: "sys",
}


def compile_empty_kernel_with_call(func, **kwargs):
    @cl.kernel
    def kernel():
        func()

    compile_kernel(kernel, **kwargs)


@pytest.mark.parametrize(
    "order, scope, expect",
    (
        (
            order,
            scope,
            f"fence.{('sc' if order is MemoryOrder.SEQ_CST else order.value)}."
            f"{scope_suffix}",
        )
        for order in (
            MemoryOrder.ACQUIRE,
            MemoryOrder.RELEASE,
            MemoryOrder.ACQ_REL,
            MemoryOrder.SEQ_CST,
        )
        for scope, scope_suffix in MEMORY_SCOPE_PTX.items()
    ),
)
def test_fence(order, scope, expect):
    def func():
        cl.fence(order, scope)

    compile_empty_kernel_with_call(
        func,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


def test_fence_defaults():
    compile_empty_kernel_with_call(cl.fence, assert_in_ptx="fence.sc.sys")


@pytest.mark.parametrize("order", (MemoryOrder.WEAK, MemoryOrder.RELAXED))
def test_fence_invalid_order(order):
    def func():
        cl.fence(order, MemoryScope.BLOCK)

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(TypeCheckingError, match="Invalid fence memory order"),
    )


def test_fence_invalid_scope():
    def func():
        cl.fence(MemoryOrder.ACQ_REL, MemoryScope.NONE)

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(TypeCheckingError, match="Invalid fence memory scope"),
    )


@pytest.mark.parametrize(
    "order, restrict, expect",
    (
        (
            MemoryOrder.RELEASE,
            cl.FenceRestriction.mbarrier_initialize(),
            "fence.mbarrier_init.release.cluster",
        ),
        (
            MemoryOrder.RELEASE,
            cl.FenceRestriction.shared_block(),
            "fence.release.sync_restrict::shared::cta.cluster",
        ),
        (
            MemoryOrder.ACQUIRE,
            cl.FenceRestriction.shared_cluster(),
            "fence.acquire.sync_restrict::shared::cluster.cluster",
        ),
    ),
)
def test_fence_restricted(order, restrict, expect):
    def func():
        cl.fence(
            order,
            cl.MemoryScope.CLUSTER,
            restriction=restrict,
        )

    compile_empty_kernel_with_call(
        func,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "proxy, restriction, expect",
    (
        (cl.FenceProxy.ALIAS, None, "fence.proxy.alias"),
        (cl.FenceProxy.ASYNC, None, "fence.proxy.async"),
        (
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.global_memory(),
            "fence.proxy.async.global",
        ),
        (
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_block(),
            "fence.proxy.async.shared::cta",
        ),
        (
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_cluster(),
            "fence.proxy.async.shared::cluster",
        ),
    ),
)
def test_fence_proxy_bidirectional(proxy, restriction, expect):
    def func():
        cl.fence_proxy_bidirectional(
            proxy,
            restriction=restriction,
        )

    compile_empty_kernel_with_call(
        func,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize("scope, scope_ptx", MEMORY_SCOPE_PTX.items())
def test_fence_proxy_tensormap_release(scope, scope_ptx):
    def func():
        cl.fence(
            cl.MemoryOrder.RELEASE,
            scope,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.TENSORMAP,
        )

    compile_empty_kernel_with_call(
        func,
        **HOPPER_TARGET,
        assert_in_ptx=f"fence.proxy.tensormap::generic.release.{scope_ptx}",
    )


@pytest.mark.parametrize("scope, scope_ptx", MEMORY_SCOPE_PTX.items())
def test_fence_proxy_tensormap_acquire(scope, scope_ptx):
    @cl.kernel
    def kernel(tensor):
        cl.fence(
            cl.MemoryOrder.ACQUIRE,
            scope,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.TENSORMAP,
            restriction=cl.FenceRestriction.address_range(tensor.get_base_pointer()),
        )

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor((1,), cl.int32)]),
        **HOPPER_TARGET,
        assert_in_ptx=f"fence.proxy.tensormap::generic.acquire.{scope_ptx}",
    )


@pytest.mark.parametrize(
    "order, restrict, expect",
    (
        (
            cl.MemoryOrder.ACQUIRE,
            cl.FenceRestriction.shared_cluster(),
            "fence.proxy.async::generic.acquire."
            "sync_restrict::shared::cluster.cluster",
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.FenceRestriction.shared_block(),
            "fence.proxy.async::generic.release."
            "sync_restrict::shared::cta.cluster",
        ),
    ),
)
def test_fence_proxy_async_split(order, restrict, expect):
    def func():
        cl.fence(
            order,
            cl.MemoryScope.CLUSTER,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.ASYNC,
            restriction=restrict,
        )

    compile_empty_kernel_with_call(
        func,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "order, scope, from_proxy, to_proxy, restriction",
    (
        (
            cl.MemoryOrder.ACQUIRE,
            cl.MemoryScope.CLUSTER,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_block(),
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.DEVICE,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_block(),
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_cluster(),
        ),
        (
            cl.MemoryOrder.ACQUIRE,
            cl.MemoryScope.DEVICE,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.shared_cluster(),
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.GENERIC,
            cl.FenceRestriction.global_memory(),
        ),
        (
            cl.MemoryOrder.ACQUIRE,
            cl.MemoryScope.DEVICE,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.TENSORMAP,
            None,
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.DEVICE,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.TENSORMAP,
            cl.FenceRestriction.shared_block(),
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            cl.FenceProxy.ASYNC,
            cl.FenceProxy.GENERIC,
            cl.FenceRestriction.shared_block(),
        ),
        (
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            cl.FenceProxy.GENERIC,
            cl.FenceProxy.ASYNC,
            cl.FenceRestriction.mbarrier_initialize(),
        ),
    ),
)
def test_fence_invalid_combination(order, scope, from_proxy, to_proxy, restriction):
    def func():
        cl.fence(
            order,
            scope,
            from_proxy=from_proxy,
            to_proxy=to_proxy,
            restriction=restriction,
        )

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(Exception),
        **HOPPER_TARGET,
    )


FABRIC_PROXY_PAIRS = tuple(
    pair
    for pair in product(
        (cl.FenceProxy.GENERIC, cl.FenceProxy.FABRIC),
        repeat=2,
    )
    if cl.FenceProxy.FABRIC in pair
)


@pytest.mark.parametrize("from_proxy, to_proxy", FABRIC_PROXY_PAIRS)
def test_fence_proxy_fabric_unsupported(from_proxy, to_proxy):
    def func():
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.SYS,
            from_proxy=from_proxy,
            to_proxy=to_proxy,
        )

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(Exception),
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "proxy, restriction",
    (
        (cl.FenceProxy.ALIAS, cl.FenceRestriction.global_memory()),
        (cl.FenceProxy.GENERIC, None),
        (cl.FenceProxy.TENSORMAP, None),
        (cl.FenceProxy.FABRIC, None),
    ),
)
def test_fence_proxy_bidirectional_invalid(proxy, restriction):
    def func():
        cl.fence_proxy_bidirectional(proxy, restriction=restriction)

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(Exception),
        **HOPPER_TARGET,
    )


def test_fence_proxy_tensormap_invalid_address_size():
    @cl.kernel
    def kernel(tensor):
        cl.fence(
            cl.MemoryOrder.ACQUIRE,
            cl.MemoryScope.DEVICE,
            from_proxy=cl.FenceProxy.GENERIC,
            to_proxy=cl.FenceProxy.TENSORMAP,
            restriction=cl.FenceRestriction.address_range(
                tensor.get_base_pointer(), 256
            ),
        )

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor((1,), cl.int32)]),
        raises=pytest.raises(TypeCheckingError, match="must have size 128"),
    )


def test_fence_address_restriction_bad_pointer():
    def func():
        cl.fence(restriction=cl.FenceRestriction.address_range(0))

    compile_empty_kernel_with_call(
        func, raises=pytest.raises(TypeCheckingError, match="Expected a pointer")
    )


def test_fence_address_restriction_bad_extent():
    @cl.kernel
    def kernel():
        array = cl.shared_array(1, cl.int32)
        cl.fence(
            restriction=cl.FenceRestriction.address_range(
                array.get_base_pointer(), 128.0
            )
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="Expected an integer constant"),
    )


def test_fence_non_proxy_address_restriction():
    @cl.kernel
    def kernel():
        array = cl.shared_array(1, cl.int32)
        cl.fence(
            cl.MemoryOrder.ACQUIRE,
            cl.MemoryScope.DEVICE,
            restriction=cl.FenceRestriction.address_range(
                array.get_base_pointer()
            ),
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(Exception),
        **HOPPER_TARGET,
    )


def test_fence_proxy_bidirectional_address_restriction():
    @cl.kernel
    def kernel():
        array = cl.shared_array(1, cl.int32)
        cl.fence_proxy_bidirectional(
            cl.FenceProxy.ASYNC,
            restriction=cl.FenceRestriction.address_range(
                array.get_base_pointer()
            ),
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(CompilerExecutionError),
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "from_proxy",
    (
        (cl.FenceProxy.GENERIC,),
        (cl.FenceProxy.GENERIC, cl.FenceProxy.ASYNC),
    ),
)
def test_fence_proxy_tuple_unsupported(from_proxy):
    def func():
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.DEVICE,
            from_proxy=from_proxy,
            to_proxy=cl.FenceProxy.TENSORMAP,
        )

    compile_empty_kernel_with_call(
        func, raises=pytest.raises(TypeCheckingError, match="Expected FenceProxy")
    )


def test_fence_restriction_tuple_unsupported():
    def func():
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=(cl.FenceRestriction.shared_block(),),
        )

    compile_empty_kernel_with_call(func, raises=pytest.raises(TypeCheckingError))


def test_fence_invalid_restriction():
    def func():
        cl.fence(restriction=1)

    compile_empty_kernel_with_call(
        func,
        raises=pytest.raises(
            TypeCheckingError,
            match="Expected FenceRestriction or None",
        ),
    )
