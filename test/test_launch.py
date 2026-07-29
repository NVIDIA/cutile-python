# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.tile as ct
from cuda.tile._compile import format_sm_arch, get_sm_arch


TS = 1024
N = 1 << 20


def _vec_add_impl(a, b, c, ts: ct.Constant[ct.int32]):
    i = ct.bid(0)
    ta = ct.load(a, (i,), (ts,))
    tb = ct.load(b, (i,), (ts,))
    ct.store(c, (i,), ct.add(ta, tb))


def test_launch_on_different_devices(monkeypatch):
    """When launch on different device, a kernel is compiled against the sm_arch
    for the device it is launched on, and reused per sm_arch.
    """

    def _spy_on_compile(monkeypatch):
        """Record the sm_arch of every ct.kernel._compile call."""
        observed = []
        original = ct.kernel._compile

        def spy(self, signature, context, compute_capability):
            observed.append(format_sm_arch(*compute_capability))
            return original(self, signature, context, compute_capability)

        monkeypatch.setattr(ct.kernel, '_compile', spy)
        return observed

    observed = _spy_on_compile(monkeypatch)
    kernel = ct.kernel(_vec_add_impl)
    device_archs = []
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            a = torch.randn(N, device=f'cuda:{dev}')
            b = torch.randn(N, device=f'cuda:{dev}')
            c = torch.empty(N, device=f'cuda:{dev}')
            ct.launch(torch.cuda.current_stream(), (N // TS,), kernel, (a, b, c, TS))
            torch.cuda.synchronize()
        torch.testing.assert_close(c.cpu(), (a + b).cpu(), msg=f"mismatch on device {dev}")
        device_archs.append(get_sm_arch(dev))

    sorted_observed = sorted(observed)
    distinct_archs = sorted(set(device_archs))
    assert sorted_observed == distinct_archs, (
        f"Expected one compile per device with archs {distinct_archs!r}, got {sorted_observed!r}"
    )


def test_launch_on_null_stream():
    """When launch on Null stream, get device context from the array arguments.
    """

    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = torch.empty(N, device='cuda')
    kernel = ct.kernel(_vec_add_impl)
    ct.launch(0, (N // TS,), kernel, (a, b, c, TS))
    torch.testing.assert_close(c, a + b)
