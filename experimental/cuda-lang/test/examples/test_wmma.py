# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import dataclasses
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import cuda.lang as cl
import torch
from cuda.lang._datatype import DType, is_integral


__doc__ = """

- Mirrors CUDA's mma.h nvcuda::wmma namespace.
- CUDA Samples analogues: simpleTensorCoreGemm and cudaTensorCoreGemm.
- See CuTe's MmaAtom

.. note::

    WMMA is not used to achieve peak performance on newer GPUs, so this code
    is better-suited as tests demonstrating what the programming model is
    capable of, as opposed to a an API surface we want to support.
    See this comment from the developer forum:

    https://forums.developer.nvidia.com/t/implement-all-supported-matrix-shapes-for-wmma-bmma-sync/361063/3

    > WMMA is now considered a compatibility / fallback
    > interface, not the main vehicle for exposing all tensor core
    > capabilities or for reaching peak performance on newer GPUs. For
    > Turing/Ampere as well, the lower-level mma PTX instructions that
    > operate directly on fragment layouts are the preferred way to fully
    > control performance characteristics (e.g., avoiding shared-memory
    > bank conflicts) beyond what WMMA can express.

"""


class FragmentUse(Enum):
    MATRIX_A = "a"
    MATRIX_B = "b"
    ACCUMULATOR = "accumulator"


class Layout(Enum):
    # corresponds to template parameter Layout = void in mma.h
    NO_LAYOUT = "void"

    ROW_MAJOR = "row"
    COL_MAJOR = "col"


@dataclass(frozen=True)
class FragmentDescriptor:
    use: FragmentUse
    m: int
    n: int
    k: int
    dtype: Any
    layout: Layout = Layout.NO_LAYOUT

    @property
    def shape(self):
        return (self.m, self.n, self.k)


@dataclass(frozen=True)
class Fragment:
    spec: FragmentDescriptor
    regs: Any


@dataclass(frozen=True)
class WmmaDescriptor:
    shape: tuple[int, int, int]
    a_dtype: DType
    b_dtype: DType
    acc_dtype: DType
    a_regs: int
    b_regs: int
    c_regs: int

    @property
    def shape_name(self):
        m, n, k = self.shape
        return f"m{m}n{n}k{k}"

    @property
    def mma_suffix(self):
        # again, float16 types get special names
        if self.a_dtype is cl.float16 and self.b_dtype is cl.float16:
            acc = _dtype_name(self.acc_dtype)
            return f"{acc}_{acc}"
        return _dtype_name(self.a_dtype)


WMMA_DESCRIPTORS = (
    # Match CUDA C++ nvcuda::wmma shapes for half/bfloat/int/tf32/f64
    # from mma.h.
    # Sub-byte modes are left out as they are in an experimental
    # namespace in the official cuda headers.
    WmmaDescriptor((16, 16, 16), cl.float16, cl.float16, cl.float32, 8, 8, 8),
    WmmaDescriptor((32, 8, 16), cl.float16, cl.float16, cl.float32, 8, 8, 8),
    WmmaDescriptor((8, 32, 16), cl.float16, cl.float16, cl.float32, 8, 8, 8),
    WmmaDescriptor((16, 16, 16), cl.float16, cl.float16, cl.float16, 8, 8, 4),
    WmmaDescriptor((32, 8, 16), cl.float16, cl.float16, cl.float16, 8, 8, 4),
    WmmaDescriptor((8, 32, 16), cl.float16, cl.float16, cl.float16, 8, 8, 4),
    WmmaDescriptor((16, 16, 16), cl.bfloat16, cl.bfloat16, cl.float32, 4, 4, 8),
    WmmaDescriptor((32, 8, 16), cl.bfloat16, cl.bfloat16, cl.float32, 8, 2, 8),
    WmmaDescriptor((8, 32, 16), cl.bfloat16, cl.bfloat16, cl.float32, 2, 8, 8),
    WmmaDescriptor((16, 16, 16), cl.int8, cl.int8, cl.int32, 2, 2, 8),
    WmmaDescriptor((32, 8, 16), cl.int8, cl.int8, cl.int32, 4, 1, 8),
    WmmaDescriptor((8, 32, 16), cl.int8, cl.int8, cl.int32, 1, 4, 8),
    WmmaDescriptor((16, 16, 16), cl.uint8, cl.uint8, cl.int32, 2, 2, 8),
    WmmaDescriptor((32, 8, 16), cl.uint8, cl.uint8, cl.int32, 4, 1, 8),
    WmmaDescriptor((8, 32, 16), cl.uint8, cl.uint8, cl.int32, 1, 4, 8),
    WmmaDescriptor((16, 16, 8), cl.tfloat32, cl.tfloat32, cl.float32, 4, 4, 8),
    WmmaDescriptor((8, 8, 4), cl.float64, cl.float64, cl.float64, 1, 1, 2),
)


@cl.function
def fragment(use, m: int, n: int, k: int, dtype, layout=Layout.NO_LAYOUT):
    return FragmentDescriptor(use, m, n, k, dtype, layout)


def _dtype_name(dtype):
    # string used in the nvvm intrinsic name for the given datatype.
    match dtype:
        case cl.bfloat16:
            return "bf16"
        case cl.float16:
            return "f16"
        case cl.float32:
            return "f32"
        case cl.float64:
            return "f64"
        case cl.tfloat32:
            return "tf32"
        case cl.int8:
            return "s8"
        case cl.int32:
            return "s32"
        case cl.uint8:
            return "u8"
        case _:
            raise TypeError(f"unsupported WMMA dtype {dtype}")


def _find_fragment_descriptor(spec):
    for op in WMMA_DESCRIPTORS:
        if op.shape != spec.shape:
            continue
        if spec.use == FragmentUse.MATRIX_A and op.a_dtype == spec.dtype:
            return op
        if spec.use == FragmentUse.MATRIX_B and op.b_dtype == spec.dtype:
            return op
        if spec.use == FragmentUse.ACCUMULATOR and op.acc_dtype == spec.dtype:
            return op
    raise TypeError(f"unsupported WMMA fragment spec {spec}")


def _find_mma_op(a_spec, b_spec, c_spec):
    if a_spec.use != FragmentUse.MATRIX_A:
        raise TypeError("mma_sync expects a matrix_a fragment as its first input")
    if b_spec.use != FragmentUse.MATRIX_B:
        raise TypeError("mma_sync expects a matrix_b fragment as its second input")
    if c_spec.use != FragmentUse.ACCUMULATOR:
        raise TypeError("mma_sync expects an accumulator fragment as its third input")
    if a_spec.shape != b_spec.shape or a_spec.shape != c_spec.shape:
        raise TypeError(
            f"mma_sync shape mismatch: A={a_spec.shape}, B={b_spec.shape}, C={c_spec.shape}"
        )

    for op in WMMA_DESCRIPTORS:
        if (
            op.shape == a_spec.shape
            and op.a_dtype == a_spec.dtype
            and op.b_dtype == b_spec.dtype
            and op.acc_dtype == c_spec.dtype
        ):
            return op
    raise TypeError(f"unsupported WMMA mma_sync spec {a_spec}, {b_spec}, {c_spec}")


def _find_load_intrinsic(spec, memory_layout):
    op = _find_fragment_descriptor(spec)
    use = "c" if spec.use == FragmentUse.ACCUMULATOR else spec.use.value
    layout = (
        memory_layout.value
        if spec.use == FragmentUse.ACCUMULATOR
        else spec.layout.value
    )
    name = f"wmma_{op.shape_name}_load_{use}_{layout}_stride_{_dtype_name(spec.dtype)}"
    return getattr(cl._nvvm, name)


def _find_mma_intrinsic(a_spec, b_spec, c_spec, satf):
    op = _find_mma_op(a_spec, b_spec, c_spec)
    a_layout = a_spec.layout.value
    b_layout = b_spec.layout.value
    name = f"wmma_{op.shape_name}_mma_{a_layout}_{b_layout}_{op.mma_suffix}"
    if satf:
        name += "_satfinite"
    return getattr(cl._nvvm, name), (op.a_regs, op.b_regs, op.c_regs)


def _to_tuple(v):
    # intrinsics that return multiple results are unpacked to tuples, but
    # single returns are just scalars. we want to pass all registers to the
    # intrinsic with one call: `intrin(*a_regs, *b_regs, *c_regs)`
    # so we convert everything to tuples.
    return v if isinstance(v, tuple) else (v,)


@cl.function
def load_matrix_sync(spec, ptr, ldm, layout=Layout.NO_LAYOUT):
    fn = cl.static_eval(_find_load_intrinsic(spec, layout))
    regs = fn(ptr, ldm)
    regs = cl.static_eval(_to_tuple(regs))
    return Fragment(spec, regs)


@cl.function
def mma_sync(a, b, c, satf=False):
    fn, reg_counts = cl.static_eval(_find_mma_intrinsic(a.spec, b.spec, c.spec, satf))
    regs = (*a.regs, *b.regs, *c.regs)
    expected_regs = cl.static_eval(sum(reg_counts))
    cl.static_assert(len(regs) == expected_regs, "mma_sync register count mismatch")
    results = fn(*regs)
    return dataclasses.replace(c, regs=results)


@cl.function
def store_matrix_sync(ptr, frag, ldm, layout):
    cl.static_assert(
        frag.spec.use == FragmentUse.ACCUMULATOR,
        "store_matrix_sync expects an accumulator fragment",
    )

    op = cl.static_eval(_find_fragment_descriptor(frag.spec))
    dtype_name = cl.static_eval(_dtype_name(frag.spec.dtype))
    layout_name = cl.static_eval(layout.value)
    fn = cl.static_eval(
        getattr(
            cl._nvvm, f"wmma_{op.shape_name}_store_d_{layout_name}_stride_{dtype_name}"
        )
    )
    cl.static_assert(
        len(frag.regs) == op.c_regs,
        f"store_matrix_sync expected {op.c_regs} registers",
    )
    fn(ptr, *frag.regs, ldm)


class wmma:
    FragmentUse = FragmentUse
    Layout = Layout

    fragment = staticmethod(fragment)
    load_matrix_sync = staticmethod(load_matrix_sync)
    mma_sync = staticmethod(mma_sync)
    store_matrix_sync = staticmethod(store_matrix_sync)


@dataclass(frozen=True)
class WmmaConfig:
    shape: tuple[int, int, int]
    global_shape: tuple[int, int, int]
    a_dtype: DType
    b_dtype: DType
    acc_dtype: DType
    a_layout: Layout = Layout.ROW_MAJOR
    b_layout: Layout = Layout.COL_MAJOR
    acc_layout: Layout = Layout.ROW_MAJOR
    atol: float = 1e-3
    rtol: float = 1e-3
    satf: bool = False

    def __str__(self):
        m, n, k = self.shape
        satf = "_satfinite" if self.satf else ""
        return (
            f"m{m}n{n}k{k}_"
            f"{_dtype_name(self.a_dtype)}x{_dtype_name(self.b_dtype)}_"
            f"acc{_dtype_name(self.acc_dtype)}_"
            f"a{self.a_layout.value}_b{self.b_layout.value}_"
            f"mem{self.acc_layout.value}{satf}"
        )


@lru_cache
def wmma_gemm_kernel(cfg: WmmaConfig) -> cl.kernel:
    @cl.kernel
    def kernel(a, b, c, d):
        config = cl.static_eval(cfg)
        wmma_m, wmma_n, wmma_k = config.shape
        global_m, global_n, global_k = config.global_shape

        tid = cl.thread_index(0)
        bx, by = cl.block_index(0), cl.block_index(1)

        a_frag_t = wmma.fragment(
            wmma.FragmentUse.MATRIX_A,
            wmma_m,
            wmma_n,
            wmma_k,
            config.a_dtype,
            config.a_layout,
        )
        b_frag_t = wmma.fragment(
            wmma.FragmentUse.MATRIX_B,
            wmma_m,
            wmma_n,
            wmma_k,
            config.b_dtype,
            config.b_layout,
        )
        acc_frag_t = wmma.fragment(
            wmma.FragmentUse.ACCUMULATOR,
            wmma_m,
            wmma_n,
            wmma_k,
            config.acc_dtype,
        )

        tile_row = by * wmma_m
        tile_col = bx * wmma_n

        # all wmma ops are warp-level, so limit to 32 threads.
        # we launch with block size 32 so this will never happen, but early
        # return in case the launch params change.
        if tid >= 32:
            return

        if config.acc_layout == wmma.Layout.COL_MAJOR:
            c_ptr = c.get_element_pointer((tile_col, tile_row))
            acc_ldm = global_m
        else:
            c_ptr = c.get_element_pointer((tile_row, tile_col))
            acc_ldm = global_n

        acc = wmma.load_matrix_sync(
            acc_frag_t,
            c_ptr,
            acc_ldm,
            config.acc_layout,
        )

        for kk in range(0, global_k, wmma_k):
            if config.a_layout == wmma.Layout.COL_MAJOR:
                a_ptr = a.get_element_pointer((kk, tile_row))
                a_ldm = global_m
            else:
                a_ptr = a.get_element_pointer((tile_row, kk))
                a_ldm = global_k

            if config.b_layout == wmma.Layout.ROW_MAJOR:
                b_ptr = b.get_element_pointer((kk, tile_col))
                b_ldm = global_n
            else:
                b_ptr = b.get_element_pointer((tile_col, kk))
                b_ldm = global_k

            acc = wmma.mma_sync(
                wmma.load_matrix_sync(a_frag_t, a_ptr, a_ldm),
                wmma.load_matrix_sync(b_frag_t, b_ptr, b_ldm),
                Fragment(acc_frag_t, acc.regs),
                config.satf,
            )

        if config.acc_layout == wmma.Layout.COL_MAJOR:
            d_ptr = d.get_element_pointer((tile_col, tile_row))
        else:
            d_ptr = d.get_element_pointer((tile_row, tile_col))

        wmma.store_matrix_sync(
            d_ptr,
            Fragment(acc_frag_t, acc.regs),
            acc_ldm,
            config.acc_layout,
        )

    return kernel


@dataclass(frozen=True)
class WmmaGemm:
    config: WmmaConfig
    a: Any
    b: Any
    c: Any
    d: Any

    @property
    def kernel(self):
        return wmma_gemm_kernel(self.config)

    def __call__(self):
        m_global, n_global, _ = self.config.global_shape
        wmma_m, wmma_n, _ = self.config.shape
        cl.launch(
            torch.cuda.current_stream(),
            (n_global // wmma_n, m_global // wmma_m),
            (32,),
            self.kernel,
            (self.a, self.b, self.c, self.d),
        )
        return self.d

    @property
    def expected(self):
        expected_dtype = (
            torch.float64 if self.config.acc_dtype is cl.float64 else torch.float32
        )
        a = self.a.cpu().to(expected_dtype)
        b = self.b.cpu().to(expected_dtype)
        c = self.c.cpu().to(expected_dtype)
        if self.config.a_layout == Layout.COL_MAJOR:
            a = a.T
        if self.config.b_layout == Layout.COL_MAJOR:
            b = b.T
        if self.config.acc_layout == Layout.COL_MAJOR:
            c = c.T
        expected = a @ b + c
        return expected.to(CL2TORCH[self.config.acc_dtype])

    def assert_close(self):
        actual = self.d.cpu()
        if self.config.acc_layout == Layout.COL_MAJOR:
            actual = actual.T
        expected = self.expected
        error = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
        # need to filter zero values so the geomean does not go to zero
        # TODO(ajm): review error formula
        positive_error = error[error > 0]
        max_error = error.max()
        geomean_error = (
            positive_error.log().mean().exp()
            if positive_error.numel() > 0
            else torch.zeros((), dtype=error.dtype)
        )
        print(
            f"\n{self.config}:\nmaxerror={max_error.item():.6g} "
            f"geomean={geomean_error.item():.6g}"
        )
        torch.testing.assert_close(
            actual,
            expected,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )


def has_satfinite(op, a_layout, b_layout):
    if op.a_dtype not in (cl.int8, cl.uint8):
        return False
    name = (
        f"wmma_{op.shape_name}_mma_"
        f"{a_layout.value}_{b_layout.value}_"
        f"{op.mma_suffix}_satfinite"
    )
    return hasattr(cl._nvvm, name)


def all_test_cases():
    for op in WMMA_DESCRIPTORS:
        m, n, k = op.shape
        for a_layout in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
            for b_layout in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
                for acc_layout in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
                    # These tolerances were roughly taken from these samples
                    # cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/
                    # cuda-samples/Samples/3_CUDA_Features/cudaTensorCoreGemm/
                    # cuda-samples/Samples/3_CUDA_Features/dmmaTensorCoreGemm/
                    # cuda-samples/Samples/3_CUDA_Features/immaTensorCoreGemm/
                    # cuda-samples/Samples/3_CUDA_Features/tf32TensorCoreGemm/
                    if is_integral(op.acc_dtype):
                        tol = 0
                    elif op.a_dtype is cl.float64:
                        tol = 1e-10
                    elif op.acc_dtype is cl.float16:
                        tol = 1e-2
                    elif op.a_dtype in (cl.bfloat16, cl.tfloat32):
                        tol = 5e-2
                    else:
                        tol = 1e-3
                    cfg = WmmaConfig(
                        op.shape,
                        (2 * m, 2 * n, 2 * k),
                        op.a_dtype,
                        op.b_dtype,
                        op.acc_dtype,
                        a_layout,
                        b_layout,
                        acc_layout,
                        atol=tol,
                        rtol=tol,
                    )
                    yield cfg
                    if has_satfinite(op, a_layout, b_layout):
                        yield dataclasses.replace(cfg, satf=True)


TEST_CASES = tuple(all_test_cases())


CL2TORCH = {
    cl.float16: torch.float16,
    cl.bfloat16: torch.bfloat16,
    cl.float32: torch.float32,
    cl.tfloat32: torch.float32,
    cl.float64: torch.float64,
    cl.int8: torch.int8,
    cl.uint8: torch.uint8,
    cl.int32: torch.int32,
}


def make_matrix(shape, dtype, generator):
    torch_dtype = CL2TORCH[dtype]
    if dtype is cl.uint8:
        return torch.randint(
            0, 17, shape, generator=generator, dtype=torch_dtype, device="cuda"
        )
    if dtype in (cl.int8, cl.int32):
        return torch.randint(
            -8, 9, shape, generator=generator, dtype=torch_dtype, device="cuda"
        )
    values = torch.rand(shape, generator=generator, dtype=torch.float32, device="cuda")
    return (values * 4.0 - 2.0).to(torch_dtype)


@pytest.mark.parametrize("case", TEST_CASES, ids=str)
def test_wmma(case):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    m_global, n_global, k_global = case.global_shape

    a_shape = (
        (k_global, m_global)
        if case.a_layout == Layout.COL_MAJOR
        else (m_global, k_global)
    )
    b_shape = (
        (k_global, n_global)
        if case.b_layout == Layout.ROW_MAJOR
        else (n_global, k_global)
    )
    acc_shape = (
        (n_global, m_global)
        if case.acc_layout == Layout.COL_MAJOR
        else (m_global, n_global)
    )

    a = make_matrix(a_shape, case.a_dtype, generator)
    b = make_matrix(b_shape, case.b_dtype, generator)
    c = make_matrix(acc_shape, case.acc_dtype, generator)
    d = torch.zeros(
        acc_shape,
        dtype=CL2TORCH[case.acc_dtype],
        device="cuda",
    )

    gemm = WmmaGemm(case, a, b, c, d)
    gemm()
    gemm.assert_close()


if __name__ == "__main__":
    for case in TEST_CASES:
        test_wmma(case)
