# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from dataclasses import dataclass
from typing import Annotated
from unittest.mock import patch

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile_host import _compile as compile_host
from cuda.lang._exception import TypeCheckingError
from cuda.tile._cext import cconv_v3_enabled


def test_host_jit_no_specialization(capfd):
    @cl.host_entry
    def add(a, b):
        print(a + b)

    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        add(15, 27)
        add(16, 26)

    ctypes.CDLL(None).fflush(None)

    assert compile_host_mock.call_count == 1
    assert capfd.readouterr().out == "42\n42\n"


def test_host_jit_specializes_argument_structure(capfd):
    @cl.host_entry
    def report(value):
        print(42)

    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        report(0)
        report((0,))
        report(1)
        report((1,))

    ctypes.CDLL(None).fflush(None)

    assert compile_host_mock.call_count == 2
    assert capfd.readouterr().out == "42\n42\n42\n42\n"


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_host_jit_specialize_static_shape():
    @cl.kernel
    def write_static_shape(x, out):
        if cl.thread_index(0) == 0:
            out[0] = x.shape[0]

    @cl.host_entry
    def entry(
        stream,
        x: Annotated[
            cl.Array,
            cl.ArrayAnnotation(static_shape_dims=(0,)),
        ],
        out,
    ):
        cl.launch(stream, (1,), (1,), write_static_shape, (x, out))

    stream = torch.cuda.current_stream()
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        entry(stream, torch.zeros(4, dtype=torch.int32, device="cuda"), output)
        assert compile_host_mock.call_count == 1
        assert output.item() == 4

        entry(stream, torch.zeros(8, dtype=torch.int32, device="cuda"), output)
        assert compile_host_mock.call_count == 2
        assert output.item() == 8


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_host_jit_two_kernel_specializations():
    @cl.kernel
    def write(value: cl.Constant[int], output):
        output[0] = value

    @cl.host_entry
    def entry(stream, first, second):
        cl.launch(stream, (1,), (1,), write, (3, first))
        cl.launch(stream, (1,), (1,), write, (7, second))

    stream = torch.cuda.current_stream()
    first = torch.zeros(1, dtype=torch.int32, device="cuda")
    second = torch.zeros(1, dtype=torch.int32, device="cuda")
    with (
        patch(
            "cuda.lang._compile_host._compile",
            wraps=compile_host,
        ) as compile_host_mock,
        patch.object(write, "_compile", wraps=write._compile) as compile_kernel_mock,
    ):
        entry(stream, first, second)
        entry(stream, first, second)

    torch.cuda.synchronize()
    assert (first.item(), second.item()) == (3, 7)
    assert compile_host_mock.call_count == 1
    assert compile_kernel_mock.call_count == 2


@pytest.mark.parametrize(
    "stream_kind",
    (
        "none",
        "int64",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(
                not cconv_v3_enabled(), reason="Requires cconv3 enabled"
            ),
        ),
    ),
)
def test_host_jit_stream_kinds(stream_kind):
    @cl.kernel
    def write(output):
        output[0] = 42

    if stream_kind == "none":
        @cl.host_entry
        def entry(output):
            cl.launch(None, (1,), (1,), write, (output,))
    elif stream_kind == "int64":
        @cl.host_entry
        def entry(stream: cl.ScalarInt64, output):
            cl.launch(stream, (1,), (1,), write, (output,))
    else:
        @cl.host_entry
        def entry(stream, output):
            cl.launch(stream, (1,), (1,), write, (output,))

    torch_stream = torch.cuda.current_stream()
    stream = {
        "int64": torch_stream.cuda_stream,
        "torch": torch_stream,
    }.get(stream_kind)
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    if stream_kind == "none":
        entry(output)
    else:
        entry(stream, output)

    torch_stream.synchronize()
    assert output.item() == 42


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_host_jit_passes_pointer_to_kernel():
    @cl.kernel
    def kernel(pointer):
        cl.static_assert(pointer.memory_space == cl.MemorySpace.GLOBAL)
        pointer.store(42)

    @cl.host_entry
    def launcher(output):
        view = cl.Array.from_parts(
            output.pointer(1), output.shape, output.strides
        )
        cl.static_assert(view.dtype == cl.int32)
        cl.launch(None, (1,), (1,), kernel, (view.pointer(),))

    output = torch.zeros(2, dtype=torch.int32, device="cuda")
    launcher(output)
    torch.cuda.synchronize()
    assert output.tolist() == [0, 42]


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_host_jit_array_permute():
    @cl.kernel
    def kernel(perm_view, out):
        out[0] = perm_view[0, 0]
        out[1] = perm_view[0, 1]
        out[2] = perm_view[1, 0]
        out[3] = perm_view[1, 1]

    @cl.host_entry
    def launcher(A, out):
        perm_view = A.permute()
        cl.launch(None, (1,), (1,), kernel, (perm_view, out))

    A = torch.arange(4, dtype=torch.int32, device="cuda:0").reshape((2, 2))
    out = torch.zeros((4,), dtype=torch.int32, device="cuda:0")
    launcher(A, out)
    torch.cuda.synchronize()
    assert out.tolist() == [0, 2, 1, 3]


def test_host_jit_preserves_argument_constness():
    @cl.kernel
    def kernel(value, output):
        output[0] = value

    @cl.host_entry
    def dynamic_entry(value, output):
        cl.launch(None, (1,), (1,), kernel, (value, output))

    @cl.host_entry
    def constant_entry(value: cl.Constant[int], output):
        cl.launch(None, (1,), (1,), kernel, (value, output))

    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    with patch.object(kernel, "_compile", wraps=kernel._compile) as compile_kernel:
        dynamic_entry(7, output)
        assert output.item() == 7
        assert compile_kernel.call_count == 1

        constant_entry(41, output)
        assert output.item() == 41
        assert compile_kernel.call_count == 2

        constant_entry(42, output)
        assert output.item() == 42
        assert compile_kernel.call_count == 3


def test_host_jit_scalar_annotation_overrides_inferred_constness():
    @cl.kernel
    def kernel(value: cl.ScalarInt64, output):
        output[0] = value

    @cl.host_entry
    def entry(value: cl.Constant[int], output):
        cl.launch(None, (1,), (1,), kernel, (value, output))

    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    with patch.object(kernel, "_compile", wraps=kernel._compile) as compile_kernel:
        entry(41, output)
        entry(42, output)
    torch.cuda.synchronize()
    assert output.item() == 42
    assert compile_kernel.call_count == 1


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_host_jit_preserves_dataclass_field_constness(monkeypatch):
    @dataclass(frozen=True)
    class KernelArgs:
        constant: int
        dynamic: int

    @cl.kernel
    def kernel(args, output):
        output[0] = cl.ensure_constant(args.constant) + args.dynamic

    @cl.host_entry
    def entry(constant: cl.Constant[int], dynamic, output):
        args = KernelArgs(constant, dynamic)
        cl.launch(None, (1,), (1,), kernel, (args, output))

    compile_count = 0
    original_compile = cl.kernel._compile

    def counted_compile(self, *args, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(cl.kernel, "_compile", counted_compile)

    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    entry(40, 2, output)
    torch.cuda.synchronize()
    assert output.item() == 42
    assert compile_count == 1

    entry(40, 3, output)
    torch.cuda.synchronize()
    assert output.item() == 43
    assert compile_count == 1

    entry(41, 2, output)
    torch.cuda.synchronize()
    assert output.item() == 43
    assert compile_count == 2


def test_host_jit_rejects_call_to_host_entry():
    @cl.host_entry
    def inner():
        pass

    @cl.host_entry
    def outer():
        inner()

    with pytest.raises(
        TypeCheckingError,
        match="Cannot call an object of type cuda.lang host entry",
    ):
        outer()
