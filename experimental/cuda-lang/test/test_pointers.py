# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch
from typing import Any
from cuda.lang._compile import compile_simt
from cuda.lang._exception import UnsupportedFeatureError, TypeCheckingError
from cuda.lang.compilation import KernelSignature

from .util import (
    make_symbolic_tensor,
    make_symbolic_scalar,
    compile_kernel,
    require_blackwell_cc100,
)


@cl.function
def loadme(ptr: cl.Pointer[Any], mo: cl.MemoryOrder):
    return ptr.load(memory_order=mo, alignment=16)


@cl.function
def storeme(ptr: cl.Pointer[Any], mo: cl.MemoryOrder):
    ptr.store(0, memory_order=mo, alignment=16)


@pytest.mark.parametrize(
    "ordering,operation",
    [
        (cl.MemoryOrder.ACQUIRE, loadme),
        (cl.MemoryOrder.RELAXED, loadme),
        (cl.MemoryOrder.WEAK, loadme),
        # Store does not support acquire or acq_rel
        (cl.MemoryOrder.RELAXED, storeme),
        (cl.MemoryOrder.RELEASE, storeme),
        (cl.MemoryOrder.WEAK, storeme),
    ],
)
def test_atomic_ptr_ldst(ordering, operation):
    @cl.kernel
    def kernel():
        dtype = cl.int32
        A = cl.shared_array(1, dtype=dtype, alignment=16)
        ptr = A.get_base_pointer()
        operation(ptr, ordering)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_atomic_store_missing_alignment():
    @cl.kernel
    def kernel():
        ptr = cl.shared_array(1, cl.int32).get_base_pointer()
        ptr.store(0, memory_order=cl.MemoryOrder.RELAXED)

    with pytest.raises(TypeCheckingError, match="Expected explicit alignment on atomic"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_atomic_load_missing_alignment():
    @cl.kernel
    def kernel():
        ptr = cl.shared_array(1, cl.int32).get_base_pointer()
        ptr.load(memory_order=cl.MemoryOrder.RELAXED)

    with pytest.raises(TypeCheckingError, match="Expected explicit alignment on atomic"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


@pytest.mark.parametrize(
    "ordering",
    [
        cl.MemoryOrder.RELEASE,
        cl.MemoryOrder.ACQ_REL,
    ],
)
def test_atomic_ptr_load_invalid_ordering(ordering):
    @cl.kernel
    def kernel(out):
        A = cl.shared_array(1, dtype=cl.int32, alignment=16)
        ptr = A.get_base_pointer()
        out[0] = ptr.load(memory_order=ordering, alignment=16)

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeCheckingError, match="Invalid memory order for Pointer.load"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


@pytest.mark.parametrize(
    "ordering",
    [
        cl.MemoryOrder.ACQUIRE,
        cl.MemoryOrder.ACQ_REL,
    ],
)
def test_atomic_ptr_store_invalid_ordering(ordering):
    @cl.kernel
    def kernel(out):
        ptr = out.get_base_pointer()
        ptr.store(cl.int32(0), memory_order=ordering, alignment=4)

    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(TypeCheckingError, match="Invalid memory order for Pointer.store"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))


def test_pointer_gep():
    @cl.kernel
    def kernel(A):
        A.get_element_pointer((0, 0)).store(1)
        A.get_element_pointer((1, 1)).store(2)
        A.get_element_pointer((2, 2)).store(3)

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
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
        smem = B.get_base_pointer()
        B2 = cl.reinterpret_pointer_as_array(smem, cl.int32, 1)
        B2[0] = 1
        A[0] = B[0, 0]
        B2[0] = 2
        A[1] = B[0, 0]

    A = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    assert A.cpu().tolist() == [1, 2]


def test_pointer_smem():
    @cl.kernel
    def kernel(A):
        B = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        B.get_element_pointer((0, 0)).store(1)
        A[0, 0] = B[0, 0]

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
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
        p = A.get_element_pointer(3)
        for i in range(A.shape[0]):
            (p - i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda()
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
        p = A.get_base_pointer()
        for i in range(A.shape[0]):
            (p + i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [0, 1, 4, 9]


def test_device_alloc_memspace():
    @cl.kernel
    def kernel(memspace):
        A = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        if cl.thread_index(0) == 0:
            memspace[0] = cl.int32(cl._nvvm.isspacep_local(p))
            memspace[1] = cl.int32(cl._nvvm.isspacep_global(p))
            memspace[2] = cl.int32(cl._nvvm.isspacep_shared(p))

        with cl.local_array(shape=(3, 3), dtype=cl.int32) as B:
            p = B.get_base_pointer()
            p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
            if cl.thread_index(0) == 0:
                memspace[3] = cl.int32(cl._nvvm.isspacep_local(p))
                memspace[4] = cl.int32(cl._nvvm.isspacep_global(p))
                memspace[5] = cl.int32(cl._nvvm.isspacep_shared(p))

    memspace = torch.zeros(6, dtype=torch.int32, device="cuda")
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
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        A[0, 0] = cl_dtype(1)
        A[1, 1] = cl_dtype(2)
        A[2, 2] = cl_dtype(3)
        out[0, 0] = A[0, 0]
        out[1, 1] = A[1, 1]
        out[2, 2] = A[2, 2]
        cl.barrier_sync_block()

    A = torch.zeros(3, 3, dtype=torch_dtype).cuda()
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
        arr[0] += arr.get_base_pointer()[0]

    arr = torch.tensor([1], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
    assert arr.cpu().item() == 2


def test_pointer_setitem():
    @cl.kernel
    def kernel(arr):
        p = arr.get_base_pointer()
        p[0] = 5

    arr = torch.tensor([1], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
    assert arr.cpu().item() == 5


@require_blackwell_cc100()
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
        pointer = cl.shared_array(1, cl.int32, alignment=4).get_base_pointer()
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
    )


def test_map_shared_to_leader_block_rejects_global_pointer():
    @cl.kernel
    def kernel(out):
        cl.map_shared_to_leader_block(out.get_base_pointer())

    with pytest.raises(TypeCheckingError, match="Expected pointer memory space"):
        compile_simt(
            kernel,
            [KernelSignature([make_symbolic_tensor(1, cl.uint32)])],
        )


def test_opaque_pointer_getitem():
    @cl.kernel
    def kernel(arr):
        p = arr.get_base_pointer()
        p = cl.bitcast(p, cl.opaque_pointer_dtype())
        arr[0] += p[0]

    with pytest.raises(
        TypeCheckingError, match="Expected concrete pointer type but got opaque_pointer"
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda()
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))


def test_opaque_pointer_setitem():
    @cl.kernel
    def kernel(arr):
        p = arr.get_base_pointer()
        p = cl.bitcast(p, cl.opaque_pointer_dtype())
        p[0] = 5

    with pytest.raises(
        TypeCheckingError,
        match="Expected concrete pointer type but got opaque_pointer",
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda()
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))


def test_pointer_access_2d_fails():
    @cl.kernel
    def kernel(arr):
        arr.get_base_pointer()[0, 0] = 5

    with pytest.raises(
        TypeCheckingError,
        match="Expected a scalar, but given value has type Tuple",
    ):
        arr = torch.tensor([1], dtype=torch.int32).cuda()
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (arr,))
