# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
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
            ("C", inp_ptr),
        )
        out[0] = value

    inp = torch.tensor([42], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [42]


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
