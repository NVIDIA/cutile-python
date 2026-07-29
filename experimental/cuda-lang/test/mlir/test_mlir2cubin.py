# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from types import SimpleNamespace

import cuda.lang._mlir as mlir
import cuda.lang._compile as compile_module
from cuda.lang._compile import mlir2cubin
import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
from cuda.lang._exception import CompilerExecutionError
from cuda.tile import _cext
from cuda.tile._annotated_function import LeafAnnotationNode


class _HackKernel(_cext.TileDispatcher):
    def __init__(self, cubin: bytes, func_name: str, arity: int):
        self._cubin = cubin
        self._func_name = func_name
        annotations = tuple(LeafAnnotationNode(constant=False) for _ in range(arity))
        super().__init__(annotations)

    def _compile(self, signature, ctx, compute_capability):
        return self._cubin, self._func_name, None, []


def test_mlir2cubin_debug_options(monkeypatch):
    compiler_argv = None

    def run(argv, **kwargs):
        nonlocal compiler_argv
        compiler_argv = argv
        return SimpleNamespace(stdout=b"cubin", stderr=b"")

    monkeypatch.setattr(compile_module.subprocess, "run", run)

    result = mlir2cubin(
        "module",
        gpu_name="sm_80",
        arch="compute_80",
        opt_level=1,
        generate_line_info=True,
    )

    assert result.cubin == b"cubin"
    assert compiler_argv is not None
    assert "--opt=1" in compiler_argv
    assert "--generate-device-line-info" in compiler_argv


def construct_1d_memref_from(
    ptr: mlir.Value, size: mlir.Value, stride: mlir.Value, element_type: mlir.Type
) -> mlir.Value:
    ptr_ty = mlir.llvm.LLVMPointerType()
    index_ty = mlir.IntegerType.signless(64)
    shape_ty = mlir.llvm.LLVMArrayType(elementType=index_ty, numElements=1)

    memref_struct_ty = mlir.llvm.LLVMStructType.make_literal(
        [ptr_ty, ptr_ty, index_ty, shape_ty, shape_ty]
    )

    zero_index = mlir.llvm.add_ConstantOp(
        res_type=index_ty, value=mlir.IntegerAttr.make(index_ty, 0)
    )
    shape_arr = mlir.llvm.add_PoisonOp(res_type=shape_ty)
    shape_arr = mlir.llvm.add_InsertValueOp(
        container=shape_arr, value=size, position=(0,)
    )

    stride_arr = mlir.llvm.add_PoisonOp(res_type=shape_ty)
    stride_arr = mlir.llvm.add_InsertValueOp(
        container=stride_arr, value=size, position=(0,)
    )

    memref_struct = mlir.llvm.add_PoisonOp(res_type=memref_struct_ty)
    memref_struct = mlir.llvm.add_InsertValueOp(
        container=memref_struct, value=ptr, position=(0,)
    )
    memref_struct = mlir.llvm.add_InsertValueOp(
        container=memref_struct, value=ptr, position=(1,)
    )
    memref_struct = mlir.llvm.add_InsertValueOp(
        container=memref_struct, value=zero_index, position=(2,)
    )
    memref_struct = mlir.llvm.add_InsertValueOp(
        container=memref_struct, value=shape_arr, position=(3,)
    )
    memref_struct = mlir.llvm.add_InsertValueOp(
        container=memref_struct, value=stride_arr, position=(4,)
    )

    strided_1d_layout = mlir.StridedLayoutAttr(
        offset=0, strides=(mlir.ShapedType.DYNAMIC,)
    )
    memref_ty = mlir.MemRefType(
        shape=(mlir.ShapedType.DYNAMIC,),
        elementType=element_type,
        layout=strided_1d_layout,
        memorySpace=None,
    )
    mr = mlir.add_UnrealizedConversionCastOp(
        outputs_types=[memref_ty], inputs=[memref_struct]
    )[0]
    return mr


def mlir_launch(mlir_module: mlir.Operation, entrypoint: str, args: tuple):
    cc = get_compute_capability()
    cubin = mlir2cubin(str(mlir_module), gpu_name=cc.gpu_name, arch=cc.arch).cubin
    kernel = _HackKernel(cubin, entrypoint, len(args))
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, args)


def test_a_plus_b():
    with mlir.Block().append_here() as top_block:
        module_region = mlir.Region()
        mlir.add_ModuleOp(
            bodyRegion=module_region,
            extra_attributes=[("gpu.container_module", mlir.UnitAttr())],
        )
    module = top_block[0]

    with module_region.new_block().append_here():
        gpu_module_region = mlir.Region()
        mlir.gpu.add_GPUModuleOp(
            sym_name="kernels",
            targets=mlir.ArrayAttr(value=[mlir.nvvm.NVVMTargetAttr()]),
            bodyRegion=gpu_module_region,
        )

    with gpu_module_region.new_block().append_here():
        ptr_ty = mlir.llvm.LLVMPointerType()
        i64_ty = mlir.IntegerType.signless(64)

        array_param_types = (ptr_ty, i64_ty, i64_ty)

        body_region = mlir.Region()
        mlir.gpu.add_GPUFuncOp(
            function_type=mlir.FunctionType(inputs=array_param_types * 3, results=()),
            body=body_region,
            extra_attributes=[
                ("sym_name", mlir.StringAttr(value="add_kernel")),
                ("gpu.kernel", mlir.UnitAttr()),
            ],
        )

    a_ptr = mlir.Value(ptr_ty, "a_ptr")
    a_size = mlir.Value(i64_ty, "a_size")
    a_stride = mlir.Value(i64_ty, "a_stride")

    b_ptr = mlir.Value(ptr_ty, "b_ptr")
    b_size = mlir.Value(i64_ty, "b_size")
    b_stride = mlir.Value(i64_ty, "b_stride")

    c_ptr = mlir.Value(ptr_ty, "c_ptr")
    c_size = mlir.Value(i64_ty, "c_size")
    c_stride = mlir.Value(i64_ty, "c_stride")

    with body_region.new_block(
        args=(a_ptr, a_size, a_stride, b_ptr, b_size, b_stride, c_ptr, c_size, c_stride)
    ).append_here():
        thread_id = mlir.gpu.add_ThreadIdOp(dimension=mlir.gpu.Dimension.x)

        memrefs = []
        for ptr, size, stride in [
            (a_ptr, a_size, a_stride),
            (b_ptr, b_size, b_stride),
            (c_ptr, c_size, c_stride),
        ]:
            memref = construct_1d_memref_from(ptr, size, stride, mlir.Float32Type())
            memrefs.append(memref)

        a, b, c = memrefs

        lhs = mlir.memref.add_LoadOp(memref=a, indices=(thread_id,))
        rhs = mlir.memref.add_LoadOp(memref=b, indices=(thread_id,))
        res = mlir.arith.add_AddFOp(lhs=lhs, rhs=rhs)
        mlir.memref.add_StoreOp(value=res, memref=c, indices=(thread_id,))
        mlir.gpu.add_ReturnOp(operands=())

    x_tensor = torch.arange(10, 138, dtype=torch.float32, device="cuda")
    y_tensor = torch.full((128,), 3.0, dtype=torch.float32, device="cuda")
    result = torch.zeros(128, dtype=torch.float32, device="cuda")
    mlir_launch(module, "add_kernel", (x_tensor, y_tensor, result))
    assert result[0] == 13.0


def test_cond_br():
    entrypoint = "cond_br_kernel"
    with mlir.Block().append_here() as top_block:
        module_region = mlir.Region()
        mlir.add_ModuleOp(
            bodyRegion=module_region,
            extra_attributes=[("gpu.container_module", mlir.UnitAttr())],
        )
    module = top_block[0]

    with module_region.new_block().append_here():
        gpu_module_region = mlir.Region()
        mlir.gpu.add_GPUModuleOp(
            sym_name="kernels",
            targets=mlir.ArrayAttr(value=[mlir.nvvm.NVVMTargetAttr()]),
            bodyRegion=gpu_module_region,
        )

    f32_ty = mlir.Float32Type()
    index_ty = mlir.IndexType()
    i64_ty = mlir.IntegerType.signless(64)
    ptr_ty = mlir.llvm.LLVMPointerType()
    array_param_types = (ptr_ty, i64_ty, i64_ty)
    array_args = tuple(mlir.Value(ty) for ty in array_param_types)

    with gpu_module_region.new_block().append_here():
        body_region = mlir.Region()
        mlir.gpu.add_GPUFuncOp(
            function_type=mlir.FunctionType(
                inputs=array_param_types,
                results=(),
            ),
            body=body_region,
            extra_attributes=[
                ("sym_name", mlir.StringAttr(value=entrypoint)),
                ("gpu.kernel", mlir.UnitAttr()),
            ],
        )

    with body_region.new_block(args=array_args).append_here():
        mr = construct_1d_memref_from(*array_args, mlir.Float32Type())
        c0_f32 = mlir.arith.add_ConstantOp(
            value=mlir.FloatAttr(
                type=f32_ty,
                value=mlir.APFloat(0.0),
            )
        )
        c0_index = mlir.arith.add_ConstantOp(
            value=mlir.IntegerAttr.make(index_ty, 0),
        )
        element = mlir.memref.add_LoadOp(memref=mr, indices=(c0_index,))
        cond = mlir.arith.add_CmpFOp(
            predicate=mlir.arith.CmpFPredicate.OEQ,
            lhs=element,
            rhs=c0_f32,
        )
        mlir.cf.add_CondBranchOp(
            condition=cond,
            trueDestOperands=[],
            falseDestOperands=[],
            trueDest=mlir.BlockLabel("eq0"),
            falseDest=mlir.BlockLabel("ne0"),
        )

    with body_region.new_block(block_id="ne0").append_here():
        c1_f32 = mlir.arith.add_ConstantOp(
            value=mlir.FloatAttr(
                type=f32_ty,
                value=mlir.APFloat(1.0),
            )
        )
        mlir.cf.add_BranchOp(destOperands=[c1_f32], dest=mlir.BlockLabel("exit"))

    with body_region.new_block(block_id="eq0").append_here():
        c2_f32 = mlir.arith.add_ConstantOp(
            value=mlir.FloatAttr(
                type=f32_ty,
                value=mlir.APFloat(2.0),
            )
        )
        mlir.cf.add_BranchOp(destOperands=[c2_f32], dest=mlir.BlockLabel("exit"))

    merged = mlir.Value(f32_ty, "merged")
    with body_region.new_block(block_id="exit", args=(merged,)).append_here():
        mlir.memref.add_StoreOp(value=merged, memref=mr, indices=(c0_index,))
        mlir.gpu.add_ReturnOp(operands=())

    zeros = torch.zeros(1, dtype=torch.float32, device="cuda")
    mlir_launch(module, entrypoint, (zeros,))
    assert zeros[0] == 2.0

    ones = torch.ones(1, dtype=torch.float32, device="cuda")
    mlir_launch(module, entrypoint, (ones,))
    assert ones[0] == 1.0


def test_mlir2cubin_error():
    with pytest.raises(CompilerExecutionError, match="Failed to parse"):
        mlir2cubin("invalid", gpu_name="sm_80", arch="compute_80")
