# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import pytest
from cuda.lang._compile import get_function_ir
from cuda.lang._ir.ir import IRContext
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import TypeCheckingError
from cuda.tile._passes.ast2hir import get_function_hir

from .util import filecheck


def filecheck_hir(func_hir: cl.kernel, check_directives: str) -> None:
    func_hir = get_function_hir(func_hir._pyfunc, entry_point=True)
    hir_string = str(func_hir.body)
    filecheck(hir_string, check_directives)


def test_load_store_in_hir():
    @cl.kernel
    def my_kernel(A):
        val = cl.load(A, 0, (1,))
        cl.store(A, 0, val + 1)

    filecheck_hir(
        my_kernel,
        """
        CHECK-LABEL: ^{{[0-9]+}}():
        CHECK: [[LOAD:%[0-9]+]] = <fn:getattr>{{.+}}'load'
        CHECK: %{{[0-9]+}} = [[LOAD]]({{.+}})
        CHECK: [[STORE:%[0-9]+]] = <fn:getattr>{{.+}}'store'
        CHECK: [[STORE]]({{.+}}, %{{[0-9]+}})
        CHECK: return
        """,
    )


def test_hir_error_logging_preserves_original_error(capsys):
    def foo(r):
        cl.static_eval(r)

    @cl.kernel
    def kernel():
        foo(range(1, 2, 3))

    func_hir = get_function_hir(kernel._pyfunc, entry_point=True)
    ctx = IRContext(log_ir_on_error=True)
    match = "Objects of type Range<int32> are not supported at compile time"
    with pytest.raises(TypeCheckingError, match=match):
        get_function_ir(func_hir, KernelSignature(()), ctx)

    stderr = capsys.readouterr().err
    assert "==== HIR for ^" in stderr
    assert "'NoneType' object has no attribute '_value_'" not in stderr
