# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature, get_compute_capability
from cuda.lang._exception import TileTypeError, TileValueError
from test.util import make_symbolic_tensor


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
def test_commit(log_ptx, mc_mask, cta_group, expect):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier).get_base_pointer()
        cl.tcgen05_commit(mbar, multicast_mask=mc_mask, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx


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
def test_alloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_alloc(p3, 5, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


def test_dealloc_requires_tensor_pointer():
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_dealloc(p3, 5)

    with pytest.raises(
        TileTypeError,
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
def test_dealloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128, cta_group=cta_group)
        tmem_ptr = smem[0]
        cl.tcgen05_dealloc(tmem_ptr, 128, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


@pytest.mark.parametrize("shape", cl.Tcgen05LdStShape._member_map_.values())
@pytest.mark.parametrize("count", (1, 2, 4, 8, 16, 32, 64, 128))
@pytest.mark.parametrize("pack", (True, False, None))
@pytest.mark.parametrize("offset", (None, 0, 1))
def test_ld(log_ptx, shape, count, pack, offset):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128)
        tmem_ptr = smem[0]
        cl.tcgen05_ld(shape, tmem_ptr, count=count, pack=pack, offset=offset)
        cl.tcgen05_dealloc(tmem_ptr, 128)

    def do_compile():
        compiled = cl.compile_simt(kernel, [KernelSignature([])])
        ptx = compiled.ptx
        assert ptx is not None
        assert "tcgen05.ld.sync.aligned" in ptx and shape.value in ptx, ptx

    bad_args = offset is not None and shape is not cl.Tcgen05LdStShape.SHAPE_16X32BX2
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X256B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
    )
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X32BX2 and offset is None
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X128B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    if bad_args:
        with pytest.raises((TileTypeError, TileValueError)):
            do_compile()
    else:
        do_compile()


@pytest.mark.parametrize("kind", cl.Tcgen05MMAKind._member_map_.values())
@pytest.mark.parametrize("cta_group", cl.CTAGroup._member_map_.values())
@pytest.mark.parametrize("collector_op", cl.Tcgen05MMACollectorOp._member_map_.values())
def test_mma_valid_enum_combinations(kind, cta_group, collector_op):
    if kind in (
        cl.Tcgen05MMAKind.I8,
        cl.Tcgen05MMAKind.MXF8F6F4,
        cl.Tcgen05MMAKind.MXF4,
        cl.Tcgen05MMAKind.MXF4NVF4,
    ):
        pytest.skip("needs updated mlir bindings")

    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_mma(
            kind,
            cta_group,
            tmem_smem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            False,
            collector_op=collector_op,
        )

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert "tcgen05.mma" in ptx, ptx


@pytest.mark.parametrize("cta_group", cl.CTAGroup._member_map_.values())
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
            cta_group,
            tmem_smem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            False,
            collector_op=cl.Tcgen05MMACollectorOp.DISCARD,
            disable_output_lane=disable_output_lane_value,
            scale_input_d=cl.int64(scale_input_d)
            if scale_input_d is not None
            else None,
        )

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert "tcgen05.mma" in ptx, ptx


def test_mma_matrix_a_validation():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_mma(
            cl.Tcgen05MMAKind.F16,
            cl.CTAGroup.CTA_1,
            tmem_smem[0],
            cl.int32(0),  # wrong type!
            cl.int64(0),
            cl.int32(0),
            False,
        )

    match = (
        "Expected a tensor memory pointer or a shared memory descriptor "
        "encoded as a 64 bit integer but got int32"
    )
    with pytest.raises(TileTypeError, match=match):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


@pytest.mark.parametrize(
    "kind,expect",
    (
        (cl.Tcgen05WaitKind.LOAD, "tcgen05.wait::ld.sync.aligned"),
        (cl.Tcgen05WaitKind.STORE, "tcgen05.wait::st.sync.aligned"),
    ),
)
def test_wait(kind, expect):
    @cl.kernel
    def kernel():
        cl.tcgen05_wait(kind)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    assert expect in compiled.ptx, compiled.ptx


def test_wait_bad_kind():
    @cl.kernel
    def kernel():
        cl.tcgen05_wait(0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


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
    @cl.kernel
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    assert expect in compiled.ptx, compiled.ptx


def test_relinquish_bad_group():
    @cl.kernel
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


@pytest.mark.parametrize(
    "group,expect",
    (
        (cl.CTAGroup.CTA_1, "tcgen05.shift.cta_group::1.down"),
        (cl.CTAGroup.CTA_2, "tcgen05.shift.cta_group::2.down"),
    ),
)
def test_shift(group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    assert expect in compiled.ptx, compiled.ptx


def test_shift_bad_group():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], 0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


def test_shift_bad_address_space(subtests):
    with subtests.test("shared"):

        @cl.kernel
        def kernel():
            ptr = cl.shared_array(1, cl.int8).get_base_pointer()
            cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)

    with subtests.test("local"):

        @cl.kernel
        def kernel():
            with cl.local_array(1, cl.int8) as arr:
                ptr = arr.get_base_pointer()
                cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)

    with subtests.test("global"):

        @cl.kernel
        def kernel(arr):
            ptr = arr.get_base_pointer()
            cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(
                kernel,
                [KernelSignature([make_symbolic_tensor(1, cl.int8)])],
                log_ptx=True,
            )
