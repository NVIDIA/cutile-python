# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from cuda.lang._compile import compile_simt, KernelSignature
import cuda.lang as cl

from ..util import filecheck, get_source


def test_ir2mlir():
    # CHECK: builtin.module
    # CHECK: gpu.module
    # CHECK-SAME: sym_name = "kernels"

    # CHECK: llvm.func
    # CHECK-SAME: sym_name = "kernel_Kt1"
    # CHECK-SAME: function_type = !llvm.func<void ()>
    # CHECK: llvm.return

    # NOTE: the mlir wrappers print ops generically,
    # so the attributes are printed at the end of the op.

    # CHECK: nvvm.kernel

    # CHECK: gpu.container_module
    def kernel():
        pass

    result = compile_simt(kernel, [KernelSignature(())], keep_mlir=True)
    filecheck(result.mlir, get_source())


def test_ir2mlir_branch():
    @cl.kernel
    def kernel(cond, res):
        y = res[0]  # noqa: F841
        if cond:
            x = 5.0
        else:
            x = 10.0
        res[0] = x

    res = torch.zeros(1, dtype=torch.float32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (True, res))
    assert res[0] == 5.0

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (False, res))
    assert res[0] == 10.0
