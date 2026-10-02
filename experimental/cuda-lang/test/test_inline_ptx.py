# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import StaticAssertionError
import torch

from cuda.tile._exception import InvalidValueError
from .util import compile_kernel


@pytest.mark.parametrize("left,right", ((False, False), (False, True), (True, False), (True, True)))
def test_inline_ptx_bool(left, right):
    @cl.kernel
    def kernel(inputs, out):
        predicate = cl._inline_ptx(
            "and.pred %0, %1, %2;", cl.bool_, inputs[0], inputs[1]
        )[0]
        cl.static_assert(cl.dtype_of(predicate) == cl.bool_)
        out[()] = predicate

    inputs = torch.tensor([left, right], dtype=torch.bool).cuda(0)
    out = torch.tensor(False, dtype=torch.bool).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inputs, out))
    assert out.cpu().item() == (left and right)


def test_inline_ptx_multiple_outputs_runtime():
    @cl.kernel
    def kernel(out):
        res0, res1 = cl._inline_ptx(
            """
            add.u32 %0, %2, %3;
            sub.u32 %1, %2, %3;
            """,
            cl.uint32,
            cl.uint32,
            cl.uint32(5),
            cl.uint32(3),
        )
        out[0] = res0
        out[1] = res1

    out = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [8, 2]


def test_inline_ptx_pointer_load():
    @cl.kernel
    def kernel(inp, out):
        inp_ptr = inp.pointer()
        (value,) = cl._inline_ptx(
            "ld.global.u32 %0, [%1];",
            cl.int32,
            inp_ptr,
        )
        out[0] = value

    inp = torch.tensor([42], dtype=torch.int32, device="cuda:0")
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_pointer_output():
    @cl.kernel
    def kernel(inp, out):
        inp_ptr = inp.pointer()
        dtype = cl.pointer_dtype(cl.int32)
        (ptr,) = cl._inline_ptx(
            "mov.u64 %0, %1;",
            dtype,
            inp_ptr,
        )
        cl.static_assert(cl.dtype_of(ptr) == dtype)
        out[0] = ptr.load()

    inp = torch.tensor([42], dtype=torch.int32, device="cuda:0")
    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_shared_pointer_output():
    @cl.kernel
    def kernel(out):
        shared = cl.shared_array(1, cl.int32)
        shared[0] = 42
        shared_ptr = shared.pointer()
        dtype = cl.pointer_dtype(cl.int32, cl.MemorySpace.SHARED)
        (result,) = cl._inline_ptx(
            "mov.u32 %0, %1;",
            dtype,
            shared_ptr
        )
        cl.static_assert(cl.dtype_of(result) == dtype)
        out[0] = result.load()

    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_special_register_operand():
    @cl.kernel
    def kernel():
        clock = cl._nvvm.read_ptx_sreg_clock()
        cl._inline_ptx(
            "mov.u32 %0, %1;",
            cl.int32,
            clock,
        )

    compile_kernel(kernel, assert_in_ptx="%clock")


@pytest.mark.xfail(
    strict=True,
    reason="needs llvm version bump",
)
def test_inline_ptx_escaped_special_register():
    @cl.kernel
    def kernel():
        cl._inline_ptx("mov.u32 %0, %%clock;", cl.int32)

    compile_kernel(
        kernel,
        assert_in_ptx="%clock",
        assert_not_in_ptx="%%clock",
    )


class TestInlinePTXErrors:

    def test_special_register_is_not_supported(self):
        def kernel():
            cl._inline_ptx("mov.u32 %0, %clock;", cl.int32)

        compile_kernel(
            kernel,
            raises=pytest.raises(
                InvalidValueError,
                match="Literal percent signs in inline PTX must be escaped",
            ),
        )


@pytest.mark.parametrize("dtype", (cl.uint32, cl.uint64))
def test_clock(dtype):
    @cl.kernel
    def kernel():
        cl.shared_array(1, dtype)[0] = cl.clock(dtype)

    check = "%clock"
    if dtype is cl.uint64:
        check += "64"

    compile_kernel(kernel, assert_in_ptx=check)


@pytest.mark.parametrize("dtype", (cl.int16, cl.int32, cl.int64, cl.float32, cl.bool_))
def test_clock_invalid_dtype(dtype):
    @cl.kernel
    def kernel():
        cl.shared_array(1, dtype)[0] = cl.clock(dtype)

    compile_kernel(kernel, raises=pytest.raises(StaticAssertionError))
