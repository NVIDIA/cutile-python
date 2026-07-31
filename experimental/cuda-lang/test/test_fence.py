# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature, get_compute_capability
from cuda.lang._enums import MemoryScope, MemorySpace, MemoryOrder
from cuda.lang._exception import (
    CompilerExecutionError,
    InvalidValueError,
    TypeCheckingError,
)
from cuda.lang._stub.fence import FenceProxyKind
from test.util import make_symbolic_tensor, compile_kernel

cc = get_compute_capability()
if cc < (9, 0):
    pytest.skip("Requires hopper", True)


MEMORY_SCOPE_PTX = {
    MemoryScope.BLOCK: "cta",
    MemoryScope.CLUSTER: "cluster",
    MemoryScope.DEVICE: "gpu",
    MemoryScope.SYS: "sys",
}

FENCE_SYNC_RESTRICT_RAISES = pytest.raises(
    CompilerExecutionError,
    match="only acquire and release semantics are supported|"
    r"attribute 'order' failed to satisfy constraint: .*\{acquire, release\}",
)
INVALID_SHARED_SPACE_RAISES = pytest.raises(
    InvalidValueError,
    match="Expected one of MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER",
)
NVVM_COMPILER_PROXY_RAISES = pytest.raises(
    CompilerExecutionError,
    match="'nvvm.fence.*' op",
)
MISSING_SHARED_SPACE_RAISES = pytest.raises(
    CompilerExecutionError, match="requires space attribute"
)
INVALID_SCOPE_RAISES = pytest.raises(
    InvalidValueError, match="Expected one of MemoryScope"
)


def compile_empty_kernel_with_call(func, **kwargs):
    @cl.kernel
    def kernel():
        func()

    compile_kernel(kernel, **kwargs)


@pytest.mark.parametrize(
    "order, scope, expect",
    (
        (order, scope, f"fence.{order.value}.{scope_suffix}")
        for order in (
            MemoryOrder.ACQUIRE,
            MemoryOrder.RELEASE,
            MemoryOrder.ACQ_REL,
        )
        for scope, scope_suffix in MEMORY_SCOPE_PTX.items()
    ),
)
def test_fence(order, scope, expect):
    def func():
        cl.fence(order, scope)

    compile_empty_kernel_with_call(func, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "order",
    (MemoryOrder.WEAK, MemoryOrder.RELAXED),
)
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
    "func, expect",
    (
        (cl.fence_mbarrier_initialize, "fence.mbarrier_init.release.cluster"),
        (cl.fence_sc_cluster, "fence.sc.cluster"),
    ),
)
def test_simple_fence(func, expect):
    compile_empty_kernel_with_call(func, assert_in_ptx=expect)


def _fence_sync_restrict_cases():
    for order in tuple(MemoryOrder):
        if order is MemoryOrder.RELEASE:
            yield order, "fence.release.sync_restrict", None
        elif order is MemoryOrder.ACQUIRE:
            yield order, "fence.acquire.sync_restrict", None
        else:
            yield order, None, FENCE_SYNC_RESTRICT_RAISES


@pytest.mark.parametrize("order, expect, raises", _fence_sync_restrict_cases())
def test_fence_sync_restrict(order, expect, raises):
    def func():
        cl.fence_sync_restrict(order)

    compile_empty_kernel_with_call(func, assert_in_ptx=expect, raises=raises)


def _fence_proxy_cases():
    for kind in FenceProxyKind:
        for space in (*list(MemorySpace), None):
            if space not in (None, MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER):
                yield kind, space, None, INVALID_SHARED_SPACE_RAISES
                continue

            if kind is FenceProxyKind.ALIAS:
                expect = "fence.proxy.alias" if space is None else None
                raises = None if space is None else NVVM_COMPILER_PROXY_RAISES
            elif kind is FenceProxyKind.ASYNC:
                expect = "fence.proxy.async" if space is None else None
                raises = None if space is None else NVVM_COMPILER_PROXY_RAISES
            elif kind is FenceProxyKind.ASYNC_GLOBAL:
                expect = "fence.proxy.async.global" if space is None else None
                raises = None if space is None else NVVM_COMPILER_PROXY_RAISES
            elif kind is FenceProxyKind.ASYNC_SHARED:
                if space is MemorySpace.SHARED:
                    expect = "fence.proxy.async.shared::cta"
                    raises = None
                elif space is MemorySpace.SHARED_CLUSTER:
                    expect = "fence.proxy.async.shared::cluster"
                    raises = None
                else:
                    expect = None
                    raises = MISSING_SHARED_SPACE_RAISES
            else:
                expect = None
                raises = NVVM_COMPILER_PROXY_RAISES
            yield kind, space, expect, raises


@pytest.mark.parametrize("kind, space, expect, raises", _fence_proxy_cases())
def test_fence_proxy(kind, space, expect, raises):
    def func():
        cl.fence_proxy(kind, space=space)

    compile_empty_kernel_with_call(func, assert_in_ptx=expect, raises=raises)


def _proxy_pair_cases(release: bool):
    direction = "release" if release else "acquire"
    for scope in MemoryScope:
        scope_ptx = MEMORY_SCOPE_PTX.get(scope)
        for from_proxy in FenceProxyKind:
            for to_proxy in FenceProxyKind:
                valid_scope = scope_ptx is not None
                valid_pair = (
                    from_proxy is FenceProxyKind.GENERIC
                    and to_proxy is FenceProxyKind.TENSORMAP
                )
                if valid_scope and valid_pair:
                    expect = f"fence.proxy.tensormap::generic.{direction}.{scope_ptx}"
                    raises = None
                elif not valid_scope:
                    expect = None
                    raises = INVALID_SCOPE_RAISES
                elif from_proxy is not FenceProxyKind.GENERIC:
                    expect = None
                    raises = NVVM_COMPILER_PROXY_RAISES
                else:
                    expect = None
                    raises = NVVM_COMPILER_PROXY_RAISES
                yield scope, from_proxy, to_proxy, expect, raises


@pytest.mark.parametrize(
    "scope, from_proxy, to_proxy, expect, raises",
    _proxy_pair_cases(release=True),
)
def test_fence_proxy_release(scope, from_proxy, to_proxy, expect, raises):
    def func():
        cl.fence_proxy_release(
            scope=scope,
            from_proxy=from_proxy,
            to_proxy=to_proxy,
        )

    compile_empty_kernel_with_call(func, assert_in_ptx=expect, raises=raises)


@pytest.mark.parametrize(
    "scope, from_proxy, to_proxy, expect, raises",
    _proxy_pair_cases(release=False),
)
def test_fence_proxy_acquire(scope, from_proxy, to_proxy, expect, raises):
    @cl.kernel
    def kernel(tensor):
        cl.fence_proxy_acquire(
            tensor.get_base_pointer(),
            128,
            scope=scope,
            from_proxy=from_proxy,
            to_proxy=to_proxy,
        )

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor((1,), cl.int32)]),
        assert_in_ptx=expect,
        raises=raises,
    )


def _fence_proxy_sync_restrict_cases():
    for order in tuple(MemoryOrder):
        for from_proxy in FenceProxyKind:
            for to_proxy in FenceProxyKind:
                valid_order = order in (MemoryOrder.ACQUIRE, MemoryOrder.RELEASE)
                valid_pair = (
                    from_proxy is FenceProxyKind.GENERIC
                    and to_proxy is FenceProxyKind.ASYNC
                )
                if valid_order and valid_pair:
                    if order is MemoryOrder.ACQUIRE:
                        expect = "fence.proxy.async::generic.acquire.sync_restrict"
                    else:
                        expect = "fence.proxy.async::generic.release.sync_restrict"
                    raises = None
                elif not valid_order:
                    expect = None
                    raises = FENCE_SYNC_RESTRICT_RAISES
                elif from_proxy is not FenceProxyKind.GENERIC:
                    expect = None
                    raises = NVVM_COMPILER_PROXY_RAISES
                else:
                    expect = None
                    raises = NVVM_COMPILER_PROXY_RAISES
                yield order, from_proxy, to_proxy, expect, raises


@pytest.mark.parametrize(
    "order, from_proxy, to_proxy, expect, raises",
    _fence_proxy_sync_restrict_cases(),
)
def test_fence_proxy_sync_restrict(order, from_proxy, to_proxy, expect, raises):
    def func():
        cl.fence_proxy_sync_restrict(
            order,
            from_proxy=from_proxy,
            to_proxy=to_proxy,
        )

    compile_empty_kernel_with_call(func, assert_in_ptx=expect, raises=raises)
