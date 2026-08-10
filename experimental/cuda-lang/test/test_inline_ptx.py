# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError, StaticAssertionError
import torch

from .util import compile_kernel


def test_inline_ptx_multiple_outputs_runtime():
    @cl.kernel
    def kernel(out):
        res0, res1 = cl._inline_ptx(
            """
            add.u32 %0, %2, %3;
            sub.u32 %1, %2, %3;
            """,
            ("=r", cl.int32),
            ("=r", cl.int32),
            ("r", 5),
            ("r", 3),
        )
        out[0] = res0
        out[1] = res1

    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [8, 2]


def test_inline_ptx_write_only_placeholders_runtime():
    @cl.kernel
    def kernel(out):
        res0, res1 = cl._inline_ptx(
            """
            add.u32 %0, %2, %3;
            sub.u32 %1, %2, %3;
            """,
            ("=r", cl.int32),
            ("=r", cl.int32),
            ("r", 5),
            ("r", 3),
        )
        out[0] = res0
        out[1] = res1

    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [8, 2]


def test_inline_ptx_pointer_load():
    @cl.kernel
    def kernel(inp, out):
        inp_ptr = inp.get_base_pointer()
        (value,) = cl._inline_ptx(
            "ld.global.u32 %0, [%1];",
            ("=r", cl.int32),
            ("p", inp_ptr),
        )
        out[0] = value

    inp = torch.tensor([42], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_pointer_output():
    @cl.kernel
    def kernel(inp, out):
        inp_ptr = inp.get_base_pointer()
        dtype = cl.pointer_dtype(cl.int32)
        (ptr,) = cl._inline_ptx(
            "mov.u64 %0, %1;",
            ("=p", dtype),
            ("p", inp_ptr),
        )
        cl.static_assert(cl.dtype_of(ptr) == dtype)
        out[0] = ptr.load()

    inp = torch.tensor([42], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_shared_pointer_output():
    @cl.kernel
    def kernel(out):
        shared = cl.shared_array(1, cl.int32)
        shared[0] = 42
        shared_ptr = shared.get_base_pointer()
        dtype = cl.pointer_dtype(cl.int32, cl.MemorySpace.SHARED)
        (result,) = cl._inline_ptx(
            "mov.u32 %0, %1;",
            ("=p", dtype),
            ("p", shared_ptr),
        )
        cl.static_assert(cl.dtype_of(result) == dtype)
        out[0] = result.load()

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [42]


def test_inline_ptx_special_register_operand():
    @cl.kernel
    def kernel():
        clock = cl._nvvm.read_ptx_sreg_clock()
        cl._inline_ptx(
            "mov.u32 %0, %1;",
            ("=r", cl.int32),
            ("r", clock),
        )

    compile_kernel(kernel, assert_in_ptx="%clock")


@pytest.mark.xfail(
    strict=True,
    reason="needs llvm version bump",
)
def test_inline_ptx_escaped_special_register():
    @cl.kernel
    def kernel():
        cl._inline_ptx("mov.u32 %0, %%clock;", ("=r", cl.int32))

    compile_kernel(
        kernel,
        assert_in_ptx="%clock",
        assert_not_in_ptx="%%clock",
    )


class TestInlinePTXErrors:

    def test_invalid_type_constraint(self):
        def kernel():
            cl._inline_ptx("add.u32 %0, %1, %1;", ("=x", cl.int32), ("r", 2))

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Unknown constraint dtype 'x'"
            ),
        )

    def test_cuda_c_constraint_is_not_supported(self):
        def kernel():
            cl._inline_ptx("// no operation", ("C", 0))

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Unknown constraint dtype 'C'"
            ),
        )

    def test_pointer_output_requires_pointer_dtype(self):
        def kernel():
            cl._inline_ptx("mov.u64 %0, 0;", ("=p", cl.int64))

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError,
                match="Expected a pointer dtype for constraint =p, got int64",
            ),
        )

    def test_read_write_constraint_is_not_supported(self):
        def kernel():
            cl._inline_ptx("add.u32 %0, %0, 1;", ("+r", 2))

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError,
                match="Read-write inline_ptx constraints are not supported",
            ),
        )

    def test_special_register_is_not_supported(self):
        def kernel():
            cl._inline_ptx("mov.u32 %0, %clock;", ("=r", cl.int32))

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError,
                match="Literal percent signs in inline_ptx must be escaped",
            ),
        )

    def test_invalid_rmw_constraint(self):
        def kernel():
            cl._inline_ptx(
                "add.u32 %0, %1, %1;",
                ("@r", cl.int32),
                ("r", 2),
            )

        compile_kernel(
            kernel,
            raises=pytest.raises(
                TypeCheckingError, match="Unknown constraint rmw modifier '@'"
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
