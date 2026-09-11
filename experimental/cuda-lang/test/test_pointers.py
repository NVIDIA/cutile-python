# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch
from cuda.lang._compile import compile_simt
from cuda.lang._exception import UnsupportedFeatureError, TypeCheckingError
from cuda.lang._ir.ops import AtomicLoad, AtomicStore
from cuda.lang.compilation import KernelSignature

from .util import (
    get_ir,
    make_symbolic_tensor,
    make_symbolic_scalar,
    compile_kernel,
)


def _get_single_op(body, op_type):
    return next(op for op in body.traverse() if isinstance(op, op_type))


@pytest.mark.parametrize(
    "memory_order",
    (cl.MemoryOrder.RELAXED, cl.MemoryOrder.ACQUIRE),
)
@pytest.mark.parametrize(
    "memory_scope",
    (
        cl.MemoryScope.BLOCK,
        cl.MemoryScope.CLUSTER,
        cl.MemoryScope.DEVICE,
        cl.MemoryScope.SYS,
    ),
)
def test_atomic_pointer_load_memory_arguments(memory_order, memory_scope):
    def kernel(source, result):
        result[0] = source.pointer().atomic_load(
            memory_order=memory_order,
            memory_scope=memory_scope,
        )

    body = get_ir(
        kernel,
        (
            make_symbolic_tensor(1, cl.int32),
            make_symbolic_tensor(1, cl.int32),
        ),
    )
    operation = _get_single_op(body, AtomicLoad)
    assert operation.memory_order is memory_order
    assert operation.memory_scope is memory_scope
    assert not operation.mmio
    assert operation.alignment == 4


@pytest.mark.parametrize(
    "memory_order",
    (cl.MemoryOrder.RELAXED, cl.MemoryOrder.RELEASE),
)
@pytest.mark.parametrize(
    "memory_scope",
    (
        cl.MemoryScope.BLOCK,
        cl.MemoryScope.CLUSTER,
        cl.MemoryScope.DEVICE,
        cl.MemoryScope.SYS,
    ),
)
def test_atomic_pointer_store_memory_arguments(memory_order, memory_scope):
    def kernel(result):
        result.pointer().atomic_store(
            cl.int32(1),
            memory_order=memory_order,
            memory_scope=memory_scope,
        )

    body = get_ir(kernel, (make_symbolic_tensor(1, cl.int32),))
    operation = _get_single_op(body, AtomicStore)
    assert operation.memory_order is memory_order
    assert operation.memory_scope is memory_scope
    assert not operation.mmio
    assert operation.alignment == 4


def test_atomic_pointer_mmio():
    def kernel(source, result):
        value = source.pointer().atomic_load(
            memory_order=cl.MemoryOrder.RELAXED,
            memory_scope=cl.MemoryScope.SYS,
            mmio=True,
        )
        result.pointer().atomic_store(
            value,
            memory_order=cl.MemoryOrder.RELAXED,
            memory_scope=cl.MemoryScope.SYS,
            mmio=True,
        )

    sym = make_symbolic_tensor(1, cl.int32)
    compile_kernel(
        kernel,
        signature=KernelSignature((sym, sym)),
        assert_in_nvvm=("load atomic volatile", "store atomic volatile"),
        assert_in_ptx=(
            "ld.mmio.relaxed.sys.global.b32",
            "st.mmio.relaxed.sys.global.b32",
        ),
        gpu_name="sm_100a",
        arch="compute_100a",
    )


@pytest.mark.parametrize(
    "method,memory_order,operation_type",
    (
        ("load", cl.MemoryOrder.RELAXED, AtomicLoad),
        ("load", cl.MemoryOrder.ACQUIRE, AtomicLoad),
        ("store", cl.MemoryOrder.RELAXED, AtomicStore),
        ("store", cl.MemoryOrder.RELEASE, AtomicStore),
    ),
)
def test_atomic_pointer_mmio_memory_order(method, memory_order, operation_type):
    def kernel(data):
        pointer = data.pointer()
        if method == "load":
            pointer.atomic_load(
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )
        else:
            pointer.atomic_store(
                cl.int32(1),
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )

    body = get_ir(kernel, (make_symbolic_tensor(1, cl.int32),))
    operation = _get_single_op(body, operation_type)
    assert operation.memory_order is memory_order
    assert operation.memory_scope is cl.MemoryScope.SYS
    assert operation.mmio


@pytest.mark.xfail(
    strict=True,
    reason=(
        "volatile ldst combination does not lower to expected mmio ptx instructions"
    ),
)
@pytest.mark.parametrize(
    "method,memory_order,instruction",
    (
        (
            "load",
            cl.MemoryOrder.ACQUIRE,
            "ld.mmio.acquire.sys.global.b32",
        ),
        (
            "store",
            cl.MemoryOrder.RELEASE,
            "st.mmio.release.sys.global.b32",
        ),
    ),
)
def test_atomic_ldst_mmio_ptx(method, memory_order, instruction):
    def kernel(data):
        pointer = data.pointer()
        if method == "load":
            pointer.atomic_load(
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )
        else:
            pointer.atomic_store(
                cl.int32(1),
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        assert_in_nvvm=("atomic volatile",),
        assert_in_ptx=instruction,
        gpu_name="sm_100a",
        arch="compute_100a",
    )


@pytest.mark.parametrize(
    "method,memory_order",
    (
        ("load", cl.MemoryOrder.WEAK),
        ("load", cl.MemoryOrder.RELEASE),
        ("load", cl.MemoryOrder.ACQ_REL),
        ("store", cl.MemoryOrder.WEAK),
        ("store", cl.MemoryOrder.ACQUIRE),
        ("store", cl.MemoryOrder.ACQ_REL),
    ),
)
def test_atomic_pointer_mmio_rejects_invalid_memory_order(method, memory_order):
    def kernel(result):
        pointer = result.pointer()
        if method == "load":
            result[0] = pointer.atomic_load(
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )
        else:
            pointer.atomic_store(
                cl.int32(1),
                memory_order=memory_order,
                memory_scope=cl.MemoryScope.SYS,
                mmio=True,
            )

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match="Invalid memory order"),
    )


@pytest.mark.parametrize(
    "method,memory_order",
    (
        ("load", cl.MemoryOrder.RELAXED),
        ("load", cl.MemoryOrder.ACQUIRE),
        ("store", cl.MemoryOrder.RELAXED),
        ("store", cl.MemoryOrder.RELEASE),
    ),
)
@pytest.mark.parametrize(
    "memory_scope",
    (
        cl.MemoryScope.BLOCK,
        cl.MemoryScope.CLUSTER,
        cl.MemoryScope.DEVICE,
    ),
)
def test_atomic_pointer_mmio_requires_system_scope(
    method, memory_order, memory_scope
):
    def kernel(result):
        pointer = result.pointer()
        if method == "load":
            result[0] = pointer.atomic_load(
                memory_order=memory_order,
                memory_scope=memory_scope,
                mmio=True,
            )
        else:
            pointer.atomic_store(
                cl.int32(1),
                memory_order=memory_order,
                memory_scope=memory_scope,
                mmio=True,
            )

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError, match="MMIO requires MemoryScope.SYS"
        ),
    )


def test_atomic_pointer_mmio_rejects_shared_memory():
    def kernel():
        array = cl.shared_array(1, cl.int32)
        array.pointer().atomic_load(
            memory_order=cl.MemoryOrder.RELAXED,
            memory_scope=cl.MemoryScope.SYS,
            mmio=True,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError, match="MMIO requires a pointer to global memory"
        ),
    )


def test_observe_atomic_load_store():
    @cl.kernel
    def kernel(result):
        first = result.pointer(0)
        second = result.pointer(1)
        first.atomic_store(cl.int32(42))
        second.atomic_store(first.atomic_load() + cl.int32(1))

    result = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (result,))
    assert result.cpu().tolist() == [42, 43]


def test_observe_atomic_load_store_free_functions():
    @cl.kernel
    def kernel(result):
        first = result.pointer(0)
        second = result.pointer(1)
        cl.atomic_store(first, cl.int32(42))
        cl.atomic_store(second, cl.atomic_load(first) + cl.int32(1))

    result = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (result,))
    assert result.cpu().tolist() == [42, 43]


@pytest.mark.parametrize(
    "method,memory_order",
    (
        ("load", cl.MemoryOrder.WEAK),
        ("load", cl.MemoryOrder.RELEASE),
        ("load", cl.MemoryOrder.ACQ_REL),
        ("store", cl.MemoryOrder.WEAK),
        ("store", cl.MemoryOrder.ACQUIRE),
        ("store", cl.MemoryOrder.ACQ_REL),
    ),
)
def test_atomic_pointer_rejects_invalid_memory_order(method, memory_order):
    def kernel(result):
        pointer = result.pointer()
        if method == "load":
            result[0] = pointer.atomic_load(memory_order=memory_order)
        else:
            pointer.atomic_store(cl.int32(1), memory_order=memory_order)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match="Invalid memory order"),
    )


@pytest.mark.parametrize("method", ("load", "store"))
def test_atomic_pointer_rejects_invalid_memory_scope(method):
    def kernel(result):
        pointer = result.pointer()
        if method == "load":
            result[0] = pointer.atomic_load(memory_scope=cl.MemoryScope.NONE)
        else:
            pointer.atomic_store(cl.int32(1), memory_scope=cl.MemoryScope.NONE)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match="Invalid memory scope"),
    )


def test_pointer_gep():
    @cl.kernel
    def kernel(A):
        A.pointer((0, 0)).store(1)
        A.pointer((1, 1)).store(2)
        A.pointer((2, 2)).store(3)

    A = torch.zeros(3, 3, dtype=torch.int32).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


def test_ptr_roundtrip():
    @cl.kernel
    def kernel(A):
        B = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        smem = B.pointer()
        B2 = cl.Array.from_parts(smem, 1)
        B2[0] = 1
        A[0] = B[0, 0]
        B2[0] = 2
        A[1] = B[0, 0]

    A = torch.zeros(2, dtype=torch.int32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    assert A.cpu().tolist() == [1, 2]


def test_array_from_parts_with_strides():
    @cl.kernel
    def kernel(array):
        view = cl.Array.from_parts(array.pointer(), (2, 2), (3, 1))
        cl.static_assert(view.shape == (2, 2))
        cl.static_assert(view.strides == (3, 1))
        cl.static_assert(view.dtype == cl.int32)
        view[0, 0] = 10
        view[0, 1] = 11
        view[1, 0] = 12
        view[1, 1] = 13

    array = torch.zeros(5, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (array,))
    assert array.cpu().tolist() == [10, 11, 0, 12, 13]


def test_array_from_parts_with_dynamic_shape():
    @cl.kernel
    def kernel(array, result, m: int, n: int):
        view = cl.Array.from_parts(array.pointer(), (m, n))
        view[1, 1] = 10
        view[2, 0] = 20
        result[0] = view.shape[0]
        result[1] = view.shape[1]
        result[2] = view.strides[0]
        result[3] = view.strides[1]

    array = torch.zeros(9, dtype=torch.int32, device="cuda")
    result = torch.zeros(4, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (array, result, 3, 3))
    assert array.cpu().tolist() == [0, 0, 0, 0, 10, 0, 20, 0, 0]
    assert result.cpu().tolist() == [3, 3, 3, 1]


def test_array_from_parts_with_dynamic_strides():
    @cl.kernel
    def kernel(array, result, m: int, n: int, row_stride: int, column_stride: int):
        view = cl.Array.from_parts(
            array.pointer(), (m, n), (row_stride, column_stride)
        )
        view[0, 1] = 10
        view[1, 0] = 20
        view[1, 1] = 30
        result[0] = view.strides[0]
        result[1] = view.strides[1]

    array = torch.zeros(7, dtype=torch.int32, device="cuda")
    result = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (array, result, 2, 2, 5, 1),
    )
    assert array.cpu().tolist() == [0, 10, 0, 0, 0, 20, 30]
    assert result.cpu().tolist() == [5, 1]


def test_array_from_parts_uses_pointer_dtype():
    @cl.kernel
    def kernel(array):
        pointer = cl.bitcast(
            array.pointer(),
            cl.pointer_dtype(cl.float32, cl.MemorySpace.GLOBAL),
        )
        view = cl.Array.from_parts(pointer, 1)
        cl.static_assert(view.dtype == cl.float32)
        view[0] = 1.5

    array = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (array,))
    assert array.cpu().item() == 0x3FC00000


def test_array_from_parts_rejects_int64_shape():
    def kernel(array):
        cl.Array.from_parts(array.pointer(), (cl.int64(2), 2))

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(4, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match="Invalid array shape: cannot implicitly cast int64 to int32",
        ),
    )


def test_array_from_parts_rejects_int64_stride():
    def kernel(array):
        cl.Array.from_parts(array.pointer(), (2, 2), (cl.int64(2), 1))

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(4, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match="Invalid array strides: cannot implicitly cast int64 to int32",
        ),
    )


def test_array_from_parts_rejects_stride_rank_mismatch():
    def kernel(array):
        cl.Array.from_parts(array.pointer(), (2, 2), (1,))

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(4, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match="Shape and strides must have the same rank, got 2 and 1",
        ),
    )


def test_array_from_parts_rejects_invalid_shape_type():
    def kernel(array):
        cl.Array.from_parts(array.pointer(), (2, 2.0))

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(4, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match=(
                "Expected a signed integer, but item at position #1 has type float32"
            ),
        ),
    )


def test_array_from_parts_rejects_invalid_stride_type():
    def kernel(array):
        cl.Array.from_parts(array.pointer(), (2, 2), (2.0, 1))

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(4, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match=(
                "Expected a signed integer, but item at position #0 has type float32"
            ),
        ),
    )


def test_array_from_parts_rejects_opaque_pointer():
    def kernel(array):
        pointer = cl.bitcast(array.pointer(), cl.opaque_pointer_dtype())
        cl.Array.from_parts(pointer, 1)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_tensor(1, cl.int32),)),
        raises=pytest.raises(
            TypeCheckingError,
            match="Expected concrete pointer type but got opaque_pointer",
        ),
    )


def test_pointer_smem():
    @cl.kernel
    def kernel(A):
        B = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        B.pointer((0, 0)).store(1)
        A[0, 0] = B[0, 0]

    A = torch.zeros(3, 3, dtype=torch.int32).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_pointer_sub_ldst():
    @cl.kernel
    def kernel(A):
        p = A.pointer(3)
        for i in range(A.shape[0]):
            (p - i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [9, 4, 1, 0]


def test_pointer_add_ldst():
    @cl.kernel
    def kernel(A):
        p = A.pointer()
        for i in range(A.shape[0]):
            (p + i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [0, 1, 4, 9]


@pytest.mark.parametrize(
    "offset_dtype,offset",
    (
        (cl.uint8, 255),
        (cl.uint16, 65535),
    ),
)
def test_pointer_add_narrow_unsigned_offset(offset_dtype, offset):
    @cl.kernel
    def kernel(A):
        p = A.pointer() + 1
        p[offset_dtype(offset)] = 7

    A = torch.ones(offset + 2, device="cuda:0")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A[0].item() == 1
    assert A[offset + 1].item() == 7


def test_shared_pointer_add_narrow_unsigned_offset():
    offset_dtype = cl.uint8
    offset = 255

    @cl.kernel
    def kernel(out):
        storage = cl.shared_array(offset + 2, cl.int32)
        p = storage.pointer() + 1
        p[offset_dtype(offset)] = 7
        out[0] = storage[offset + 1]

    out = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (out,),
    )
    assert out.item() == 7


def test_device_alloc_memspace():
    @cl.kernel
    def kernel(memspace):
        A = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        p = A.pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        if cl.thread_index(0) == 0:
            memspace[0] = cl.int32(cl._nvvm.isspacep_local(p))
            memspace[1] = cl.int32(cl._nvvm.isspacep_global(p))
            memspace[2] = cl.int32(cl._nvvm.isspacep_shared(p))

        with cl.local_array(shape=(3, 3), dtype=cl.int32) as B:
            p = B.pointer()
            p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
            if cl.thread_index(0) == 0:
                memspace[3] = cl.int32(cl._nvvm.isspacep_local(p))
                memspace[4] = cl.int32(cl._nvvm.isspacep_global(p))
                memspace[5] = cl.int32(cl._nvvm.isspacep_shared(p))

    memspace = torch.zeros(6, dtype=torch.int32, device="cuda:0")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (memspace,),
    )
    assert memspace.cpu().tolist() == [0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "torch_dtype,cl_dtype",
    [
        (torch.int32, cl.int32),
        (torch.float32, cl.float32),
        (torch.int64, cl.int64),
        (torch.float64, cl.float64),
    ],
)
def test_static_shared_array(torch_dtype, cl_dtype):

    @cl.kernel
    def kernel(out):
        A = cl.shared_array(shape=(3, 3), dtype=cl_dtype)
        p = A.pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        A[0, 0] = cl_dtype(1)
        A[1, 1] = cl_dtype(2)
        A[2, 2] = cl_dtype(3)
        out[0, 0] = A[0, 0]
        out[1, 1] = A[1, 1]
        out[2, 2] = A[2, 2]
        cl.barrier_sync_block_aligned()

    A = torch.zeros(3, 3, dtype=torch_dtype).cuda(0)
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    A = A.cpu()
    assert A[0, 0] == 1
    assert A[1, 1] == 2
    assert A[2, 2] == 3


def test_device_allocation_alignment_lowering():
    @cl.kernel
    def kernel():
        shared = cl.shared_array(shape=(4,), dtype=cl.int32, alignment=128)
        with cl.local_array(shape=(4,), dtype=cl.int32, alignment=16) as local:
            local[0] = cl.int32(1)
            shared[0] = local[0]

    result = compile_simt(
        kernel,
        [KernelSignature(())],
        gpu_name="sm_80",
        arch="compute_80",
        keep_mlir=True,
    )

    assert "alignment = 16 : i64" in result.mlir
    assert "alignment = 128 : i64" in result.mlir


def make_local_array(shape, dtype, alignment):
    with cl.local_array(shape, dtype, alignment):
        pass


@pytest.mark.parametrize("allocator", [make_local_array, cl.shared_array])
@pytest.mark.parametrize("alignment", [0, -1, 3, True])
def test_device_allocation_invalid_alignment(allocator, alignment):
    def kernel():
        allocator(shape=(1,), dtype=cl.int32, alignment=alignment)

    match = (
        "Expected an integer constant"
        if isinstance(alignment, bool)
        else "alignment must be a positive power of two"
    )
    compile_kernel(kernel, raises=pytest.raises(TypeCheckingError, match=match))


@pytest.mark.parametrize("allocator", [make_local_array, cl.shared_array])
def test_device_allocation_alignment_must_be_constant(allocator):
    def kernel(alignment):
        allocator(shape=(1,), dtype=cl.int32, alignment=alignment)

    compile_kernel(
        kernel,
        signature=KernelSignature((make_symbolic_scalar(cl.int32),)),
        raises=pytest.raises(TypeCheckingError, match="Expected an integer constant"),
    )


def test_allocate_shmem_in_runtime_conditional():
    def kernel(tensor):
        if tensor[0]:
            cl.shared_array(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    compile_kernel(
        kernel,
        signature=KernelSignature((tensor_constraint,)),
        raises=pytest.raises(
            UnsupportedFeatureError, match="Memory allocated in dynamic control flow"
        ),
    )


def test_allocate_shmem_in_runtime_loop():
    def kernel(tensor):
        for _ in range(cl.int32(tensor[0])):
            cl.shared_array(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    compile_kernel(
        kernel,
        signature=KernelSignature((tensor_constraint,)),
        raises=pytest.raises(
            UnsupportedFeatureError, match="Memory allocated in dynamic control flow"
        ),
    )


def test_pointer_getitem():
    @cl.kernel
    def kernel(arr):
        arr[0] += arr.pointer()[0]

    arr = torch.tensor([1], dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
    assert arr.cpu().item() == 2


def test_pointer_setitem():
    @cl.kernel
    def kernel(arr):
        p = arr.pointer()
        p[0] = 5

    arr = torch.tensor([1], dtype=torch.int32).cuda(0)
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
    assert arr.cpu().item() == 5


@pytest.mark.parametrize("cluster", (False, True))
def test_map_shared_to_leader_block(cluster):
    """
    The ptx compiler will see the alignment and not mask the lowest bits
    since they will always be zero anyways - this is why we don't see the exact
    mask pattern provided by cl.shared_cluster_leader_bit_mask() in the ptx:

    >>> # What we see in the ptx without the mapa instruction
    >>> print(hex(-16777220 & 0xFFFFFFFF))
    0xfefffffc

    In the cluster case, the alignment is not reflected in the mask, so we
    see the exact value of cl.shared_cluster_leader_bit_mask() in the assembly:

    >>> # What we see in the ptx with the mapa instruction
    >>> print(hex(-16777217 & 0xFFFFFFFF))
    0xfeffffff
    """
    expected_space = cl.MemorySpace.SHARED_CLUSTER if cluster else cl.MemorySpace.SHARED

    @cl.kernel
    def kernel(out):
        pointer = cl.shared_array(1, cl.int32, alignment=4).pointer()
        if cluster:
            pointer = cl.map_shared_to_cluster(pointer, 0)
        mapped = cl.map_shared_to_leader_block(pointer)
        cl.static_assert(cl.dtype_of(mapped) == cl.pointer_dtype(cl.int32, expected_space))
        out[0] = cl.bitcast(mapped, cl.uint32)

    constant = "-16777217" if cluster else "-16777220"
    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor(1, cl.uint32)]),
        filecheck_ptx=f"""
        CHECK: and.b32
        CHECK-SAME: {constant}
        """,
        gpu_name="sm_100a",
        arch="compute_100a",
    )


def test_map_shared_to_leader_block_rejects_global_pointer():
    @cl.kernel
    def kernel(out):
        cl.map_shared_to_leader_block(out.pointer())

    with pytest.raises(TypeCheckingError, match="Expected pointer memory space"):
        compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.uint32)])],
        )


def test_opaque_pointer_arithmetic():
    def kernel(array):
        pointer = cl.bitcast(array.pointer(), cl.opaque_pointer_dtype())
        return pointer + 1

    with pytest.raises(
        TypeCheckingError, match="Opaque pointers do not support pointer arithmetic"
    ):
        get_ir(kernel, (make_symbolic_tensor(1, cl.int32),))


def test_opaque_pointer_getitem():
    @cl.kernel
    def kernel(arr):
        p = arr.pointer()
        p = cl.bitcast(p, cl.opaque_pointer_dtype())
        arr[0] += p[0]

    with pytest.raises(
        TypeCheckingError, match="Expected concrete pointer type but got opaque_pointer"
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda(0)
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))


def test_opaque_pointer_setitem():
    @cl.kernel
    def kernel(arr):
        p = arr.pointer()
        p = cl.bitcast(p, cl.opaque_pointer_dtype())
        p[0] = 5

    with pytest.raises(
        TypeCheckingError,
        match="Expected concrete pointer type but got opaque_pointer",
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda(0)
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))


def test_pointer_access_2d_fails():
    @cl.kernel
    def kernel(arr):
        arr.pointer()[0, 0] = 5

    with pytest.raises(
        TypeCheckingError,
        match="Expected a scalar, but given value has type Tuple",
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda(0)
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
