# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature, get_compute_capability
from cuda.lang._exception import (
    TypeCheckingError,
    InvalidValueError,
    CompilerExecutionError,
)
from test.util import make_symbolic_tensor, compile_kernel


cc = get_compute_capability()

if cc.major != 10:
    pytest.skip(reason="Blackwell only", allow_module_level=True)


@pytest.mark.parametrize(
    "mc_mask,cta_group,expect",
    [
        [
            0xAB,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_2,
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64",
        ],
    ],
)
def test_commit(mc_mask, cta_group, expect):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier).get_base_pointer()
        cl.tcgen05_commit(mbar, multicast_mask=mc_mask, cta_group=cta_group)

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
        ],
    ],
)
def test_alloc(cta_group, expect):
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_allocate(p3, 5, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], keep_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


def test_dealloc_requires_tensor_pointer():
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_deallocate(p3, 5)

    with pytest.raises(
        TypeCheckingError,
        match="Expected pointer memory space to be MemorySpace.TENSOR "
        "but got MemorySpace.SHARED",
    ):
        cl.compile_simt(kernel, [KernelSignature([])])


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32",
        ],
    ],
)
def test_dealloc(cta_group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_allocate(smem.get_base_pointer(), 128, cta_group=cta_group)
        tmem_ptr = smem[0]
        cl.tcgen05_deallocate(tmem_ptr, 128, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], keep_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


def test_tmem_offset():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        tmem = cl.tcgen05_tmem_offset(
            smem[0],
            lane_offset=32,
            column_offset=7,
        )
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_32X32B,
            tmem,
            cl.int32(0),
        )

    compile_kernel(kernel, assert_in_ptx="tcgen05.st.sync.aligned.32x32b.x1.b32")


def test_tmem_offset_with_lane_offset():
    @cl.kernel
    def kernel(out):
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        tmem = cl.tcgen05_tmem_offset(
            smem[0],
            lane_offset=5,
            column_offset=7,
        )
        # Use the TMEM value so that the compiler does not remove it.
        out[cl.bitcast(tmem, cl.int32)] = 1

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor(1, cl.int32)]),
        filecheck_nvvm="""
        CHECK: ptrtoint
        CHECK-NEXT: ashr i32
        CHECK-NEXT: and i32
        CHECK-NEXT: add i32
        CHECK-NEXT: add i32
        CHECK-NEXT: shl i32
        CHECK-NEXT: or i32
        CHECK-NEXT: inttoptr
        """,
    )


def test_tmem_offset_without_lane_offset():
    """Ensure we don't do extra pointer arithmetic when the lane offset is
    statically zero."""

    @cl.kernel
    def kernel(out):
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        tmem = cl.tcgen05_tmem_offset(
            smem[0],
            lane_offset=0,
            column_offset=7,
        )
        # Use the TMEM value so that the compiler does not remove it.
        out[cl.bitcast(tmem, cl.int32)] = 1

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor(1, cl.int32)]),
        filecheck_nvvm="""
        CHECK: ptrtoint
        CHECK-NEXT: add i32
        CHECK-NEXT: inttoptr
        """,
    )


def test_tmem_offset_requires_tensor_pointer():
    @cl.kernel
    def kernel():
        smem = cl.shared_array(1, cl.int8)
        cl.tcgen05_tmem_offset(smem.get_base_pointer(), lane_offset=32)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match="Expected pointer memory space to be MemorySpace.TENSOR",
        ),
    )


STORE_VALID_COUNTS_BY_SHAPE = {
    cl.Tcgen05LoadStoreShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    cl.Tcgen05LoadStoreShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    cl.Tcgen05LoadStoreShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    cl.Tcgen05LoadStoreShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}


@pytest.mark.parametrize(
    "shape,count",
    [
        (shape, count)
        for shape, counts in STORE_VALID_COUNTS_BY_SHAPE.items()
        for count in counts
    ],
)
@pytest.mark.parametrize("unpack", (False, True))
def test_store(shape, count, unpack):
    offset = 1 if shape is cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2 else None
    registers_per_count = {
        cl.Tcgen05LoadStoreShape.SHAPE_16X64B: 1,
        cl.Tcgen05LoadStoreShape.SHAPE_16X128B: 2,
        cl.Tcgen05LoadStoreShape.SHAPE_16X256B: 4,
        cl.Tcgen05LoadStoreShape.SHAPE_32X32B: 1,
        cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2: 1,
    }[shape]
    register_count = count * registers_per_count

    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        storage = cl.shared_array(256, cl.int32)
        v = storage.get_element_pointer(0).load(count=register_count)
        cl.tcgen05_allocate(smem.get_base_pointer(), 128)
        cl.tcgen05_store(shape, smem[0], v, unpack=unpack, offset=offset)
        cl.tcgen05_wait_store()
        cl.tcgen05_deallocate(smem[0], 128)

    expect = (
        f"tcgen05.st.sync.aligned.{shape.value}.x{count}"
        + (".unpack::16b" if unpack else "")
        + ".b32"
    )
    compile_kernel(kernel, assert_in_ptx=expect)


def test_store_rejects_wrong_value_dtype():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_16X64B,
            smem[0],
            cl.float32(0),
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="Expected scalar 32-bit integer"),
    )


def test_store_rejects_invalid_register_count():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        value = cl.Vector(cl.int32(0), cl.int32(0), cl.int32(0))

        # 16x128b requires 2 * count registers; three is invalid.
        cl.tcgen05_store(
            cl.Tcgen05LoadStoreShape.SHAPE_16X128B,
            smem[0],
            value,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(InvalidValueError, match="Expected register count"),
    )


@pytest.mark.parametrize(
    "shape,offset",
    (
        (cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2, None),
        (cl.Tcgen05LoadStoreShape.SHAPE_16X64B, 1),
    ),
)
def test_store_offset_validation(shape, offset):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_store(shape, smem[0], cl.int32(0), offset=offset)

    compile_kernel(
        kernel,
        raises=pytest.raises(TypeCheckingError, match="offset"),
    )


@pytest.mark.parametrize("shape", (*tuple(cl.Tcgen05CopyShape), None))
@pytest.mark.parametrize("cta_group", (*tuple(cl.CTAGroup), None))
@pytest.mark.parametrize("multicast", (*tuple(cl.Tcgen05CopyMulticast), None, 5))
@pytest.mark.parametrize("source_format", (*tuple(cl.Tcgen05CopySourceFormat), None, 5))
def test_copy(shape, cta_group, multicast, source_format):
    allocation_group = (
        cta_group if isinstance(cta_group, cl.CTAGroup) else cl.CTAGroup.CTA_1
    )

    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_allocate(smem.get_base_pointer(), 128, cta_group=allocation_group)
        tmem_ptr = smem[0]
        descriptor = cl.uint64(0xDEADBEEF)
        cl.tcgen05_copy(
            tmem_ptr,
            descriptor,
            cta_group=cta_group,
            shape=shape,
            multicast=multicast,
            source_format=source_format,
        )
        cl.tcgen05_deallocate(tmem_ptr, 128, cta_group=allocation_group)

    valid_multicasts = {
        cl.Tcgen05CopyShape.SHAPE_128x256b: (None,),
        cl.Tcgen05CopyShape.SHAPE_4x256b: (None,),
        cl.Tcgen05CopyShape.SHAPE_128x128b: (None,),
        cl.Tcgen05CopyShape.SHAPE_64x128b: (
            cl.Tcgen05CopyMulticast.WARPX2_02_13,
            cl.Tcgen05CopyMulticast.WARPX2_01_23,
        ),
        cl.Tcgen05CopyShape.SHAPE_32x128b: (cl.Tcgen05CopyMulticast.WARPX4,),
    }

    expect = None
    raises = None
    if cta_group not in tuple(cl.CTAGroup):
        raises = pytest.raises(TypeCheckingError, match="Expected CTAGroup")
    elif shape not in tuple(cl.Tcgen05CopyShape):
        raises = pytest.raises(TypeCheckingError, match="Expected Tcgen05CopyShape")
    elif multicast not in (*tuple(cl.Tcgen05CopyMulticast), None):
        raises = pytest.raises(TypeCheckingError, match="Expected Tcgen05CopyMulticast")
    elif source_format not in (*tuple(cl.Tcgen05CopySourceFormat), None):
        raises = pytest.raises(
            TypeCheckingError, match="Expected Tcgen05CopySourceFormat"
        )
    elif multicast not in valid_multicasts[shape]:
        raises = pytest.raises(CompilerExecutionError)
    else:
        shape_str = shape.name.removeprefix("SHAPE_")
        group_str = "cta_group::" + str(1 if cta_group is cl.CTAGroup.CTA_1 else 2)
        multicast_str = {
            None: "",
            cl.Tcgen05CopyMulticast.WARPX2_02_13: ".warpx2::02_13",
            cl.Tcgen05CopyMulticast.WARPX2_01_23: ".warpx2::01_23",
            cl.Tcgen05CopyMulticast.WARPX4: ".warpx4",
        }[multicast]
        source_format_str = {
            None: "",
            cl.Tcgen05CopySourceFormat.B6x16_P32: ".b8x16.b6x16_p32",
            cl.Tcgen05CopySourceFormat.B4x16_P64: ".b8x16.b4x16_p64",
        }[source_format]
        expect = (
            f"tcgen05.cp.{group_str}.{shape_str}" + multicast_str + source_format_str
        )

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


@pytest.mark.parametrize("shape", tuple(cl.Tcgen05LoadStoreShape))
@pytest.mark.parametrize("count", (1, 2, 4, 8, 16, 32, 64, 128))
@pytest.mark.parametrize("pack", (True, False, None))
@pytest.mark.parametrize("offset", (None, 0, 1))
def test_load(shape, count, pack, offset):
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_allocate(smem.get_base_pointer(), 128)
        tmem_ptr = smem[0]
        cl.tcgen05_load(shape, tmem_ptr, count=count, pack=pack, offset=offset)
        cl.tcgen05_deallocate(tmem_ptr, 128)

    bad_args = (
        offset is not None and shape is not cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2
    )
    bad_args |= shape is cl.Tcgen05LoadStoreShape.SHAPE_16X256B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
    )
    bad_args |= shape is cl.Tcgen05LoadStoreShape.SHAPE_16X32BX2 and offset is None
    bad_args |= shape is cl.Tcgen05LoadStoreShape.SHAPE_16X128B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    compile_kernel(
        kernel,
        assert_in_ptx=None if bad_args else ("tcgen05.ld.sync.aligned", shape.value),
        raises=pytest.raises((TypeCheckingError, InvalidValueError))
        if bad_args
        else None,
    )


ORDINARY_MMA_KINDS = (
    cl.Tcgen05MMAKind.I8,
    cl.Tcgen05MMAKind.F8F6F4,
    cl.Tcgen05MMAKind.F16,
    cl.Tcgen05MMAKind.TF32,
)


@pytest.mark.parametrize("sparse", (False, True))
@pytest.mark.parametrize("kind", ORDINARY_MMA_KINDS)
@pytest.mark.parametrize("cta_group", tuple(cl.CTAGroup))
@pytest.mark.parametrize("collector_op", tuple(cl.Tcgen05MMACollectorOp))
def test_mma_valid_enum_combinations(sparse, kind, cta_group, collector_op, request):

    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem = cl.shared_array(2, tmem_dtype, alignment=4)
        sparse_metadata = tmem[1] if sparse else None
        cl.tcgen05_mma(
            kind,
            tmem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            accumulate=False,
            sparse_metadata=sparse_metadata,
            cta_group=cta_group,
            collector_op=collector_op,
        )

    compile_kernel(
        kernel,
        assert_in_ptx="tcgen05.mma.sp" if sparse else "tcgen05.mma",
        assert_not_in_ptx=None if sparse else "tcgen05.mma.sp",
    )


MMA_BLOCK_SCALE_CASES = (
    (
        cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
        cl.Tcgen05MMAScaleVectorSize.DEFAULT,
    ),
    (
        cl.Tcgen05MMABlockScaleKind.MXF8F6F4,
        cl.Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
    (
        cl.Tcgen05MMABlockScaleKind.MXF4,
        cl.Tcgen05MMAScaleVectorSize.DEFAULT,
    ),
    (
        cl.Tcgen05MMABlockScaleKind.MXF4,
        cl.Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
    (
        cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
        cl.Tcgen05MMAScaleVectorSize.BLOCK_16,
    ),
    (
        cl.Tcgen05MMABlockScaleKind.MXF4NVF4,
        cl.Tcgen05MMAScaleVectorSize.BLOCK_32,
    ),
)


@pytest.mark.parametrize("sparse", (False, True))
@pytest.mark.parametrize("kind", tuple(cl.Tcgen05MMABlockScaleKind))
@pytest.mark.parametrize(
    "scale_vector_size",
    tuple(cl.Tcgen05MMAScaleVectorSize),
)
@pytest.mark.parametrize("cta_group", tuple(cl.CTAGroup))
@pytest.mark.parametrize(
    "collector_op",
    tuple(cl.Tcgen05MMACollectorOp),
)
def test_mma_block_scale_valid_enum_combinations(
    sparse,
    kind,
    scale_vector_size,
    cta_group,
    collector_op,
):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem = cl.shared_array(4, tmem_dtype, alignment=4)
        sparse_metadata = tmem[3] if sparse else None

        cl.tcgen05_mma_block_scale(
            kind,
            tmem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            tmem[1],
            tmem[2],
            accumulate=False,
            sparse_metadata=sparse_metadata,
            cta_group=cta_group,
            scale_vector_size=scale_vector_size,
            collector_op=collector_op,
        )

    if (kind, scale_vector_size) in MMA_BLOCK_SCALE_CASES:
        compile_kernel(
            kernel,
            assert_not_in_ptx=None if sparse else "tcgen05.mma.sp",
            filecheck_ptx=f"""CHECK: {"tcgen05.mma.sp" if sparse else "tcgen05.mma"}
                             CHECK-SAME: .block_scale""",
        )
    else:
        compile_kernel(
            kernel,
            raises=pytest.raises(InvalidValueError),
        )


@pytest.mark.parametrize("sparse", (False, True))
@pytest.mark.parametrize("kind", ORDINARY_MMA_KINDS)
@pytest.mark.parametrize("collector_b_buffer", tuple(cl.Tcgen05MMACollectorBBuffer))
@pytest.mark.parametrize("collector_op", tuple(cl.Tcgen05MMACollectorOp))
def test_mma_weight_stationary_valid_enum_combinations(
    sparse,
    kind,
    collector_b_buffer,
    collector_op,
):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem = cl.shared_array(2, tmem_dtype, alignment=4)
        sparse_metadata = tmem[1] if sparse else None

        cl.tcgen05_mma_weight_stationary(
            kind,
            tmem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            accumulate=False,
            sparse_metadata=sparse_metadata,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
        )

    compile_kernel(
        kernel,
        assert_in_ptx="tcgen05.mma.ws.sp" if sparse else "tcgen05.mma.ws",
        assert_not_in_ptx=None if sparse else "tcgen05.mma.ws.sp",
    )


@pytest.mark.parametrize("cta_group", tuple(cl.CTAGroup))
@pytest.mark.parametrize("scale_input_d", (None, 0, 15))
@pytest.mark.parametrize("disable_output_lane", (False, True))
def test_mma_optional_operands(cta_group, scale_input_d, disable_output_lane):

    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        if disable_output_lane:
            if cta_group == cl.CTAGroup.CTA_1:
                disable_output_lane_value = cl.Vector(
                    cl.int32(0), cl.int32(0), cl.int32(0), cl.int32(0)
                )
            else:
                disable_output_lane_value = cl.Vector(
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                )
        else:
            disable_output_lane_value = None

        cl.tcgen05_mma(
            cl.Tcgen05MMAKind.F16,
            tmem_smem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            accumulate=False,
            cta_group=cta_group,
            collector_op=cl.Tcgen05MMACollectorOp.DISCARD,
            disable_output_lane=disable_output_lane_value,
            scale_input_d=cl.int64(scale_input_d)
            if scale_input_d is not None
            else None,
        )

    compile_kernel(kernel, assert_in_ptx="tcgen05.mma")


def test_mma_matrix_a_validation():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_mma(
            cl.Tcgen05MMAKind.F16,
            tmem_smem[0],
            cl.int32(0),  # wrong type!
            cl.int64(0),
            cl.int32(0),
            accumulate=False,
            cta_group=cl.CTAGroup.CTA_1,
        )

    match = (
        "Expected a tensor memory pointer or a shared memory descriptor "
        "encoded as a 64 bit integer but got int32"
    )

    compile_kernel(kernel, raises=pytest.raises(TypeCheckingError, match=match))


@pytest.mark.parametrize(
    "op,expect",
    (
        (cl.tcgen05_wait_load, "tcgen05.wait::ld.sync.aligned"),
        (cl.tcgen05_wait_store, "tcgen05.wait::st.sync.aligned"),
    ),
)
def test_wait(op, expect):
    def kernel():
        op()

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "op,expect",
    (
        (
            cl.tcgen05_fence_before_thread_sync,
            "tcgen05.fence::before_thread_sync",
        ),
        (
            cl.tcgen05_fence_after_thread_sync,
            "tcgen05.fence::after_thread_sync",
        ),
    ),
)
def test_fence(op, expect):
    def kernel():
        op()

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "group,expect",
    (
        (
            cl.CTAGroup.CTA_1,
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        ),
        (
            cl.CTAGroup.CTA_2,
            "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned",
        ),
    ),
)
def test_relinquish(group, expect):
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(group)

    compile_kernel(kernel, assert_in_ptx=expect)


def test_relinquish_bad_group():
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(0xDEADBEEF)

    compile_kernel(kernel, raises=pytest.raises(Exception))


@pytest.mark.parametrize(
    "group,expect",
    (
        (cl.CTAGroup.CTA_1, "tcgen05.shift.cta_group::1.down"),
        (cl.CTAGroup.CTA_2, "tcgen05.shift.cta_group::2.down"),
    ),
)
def test_shift(group, expect):
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], group)

    compile_kernel(kernel, assert_in_ptx=expect)


def test_shift_bad_group():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], 0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])])


def test_shift_bad_address_space_shared():
    def kernel():
        ptr = cl.shared_array(1, cl.int8).get_base_pointer()
        cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

    compile_kernel(kernel, raises=pytest.raises(Exception))


def test_shift_bad_address_space_local():

    def kernel():
        with cl.local_array(1, cl.int8) as arr:
            ptr = arr.get_base_pointer()
            cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

    compile_kernel(kernel, raises=pytest.raises(Exception))


def test_shift_bad_address_space_global():
    def kernel(arr):
        ptr = arr.get_base_pointer()
        cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

    compile_kernel(
        kernel,
        signature=KernelSignature([make_symbolic_tensor(1, cl.int8)]),
        raises=pytest.raises(Exception),
    )
