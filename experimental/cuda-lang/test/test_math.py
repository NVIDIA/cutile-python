# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from test.util import compile_kernel
import cuda.lang as cl
import cuda.lang._datatype as datatype
import builtins
import itertools
import math as host_math
import operator
import sys
import torch
import pytest
from cuda.lang import compile_simt
from cuda.lang._compile import get_compute_capability
from cuda.lang._stub import math as device_math
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import CompilerExecutionError, TypeCheckingError
from cuda.lang._fp_utils import _FLOAT_SMALLEST_NORMAL, isnormal
from cuda.lang._target import TargetFeature, TargetInfo
from .util import make_symbolic_tensor


rng = torch.Generator().manual_seed(0)
FLOAT_TYPES = (
    cl.float16,
    cl.float32,
    cl.float64,
)
FLOAT_TOLERANCES = {
    cl.float16: dict(rel=1e-2, abs=1e-2),
    cl.float32: dict(rel=1e-5, abs=1e-5),
    cl.float64: dict(rel=1e-10, abs=1e-10),
}


SIGNED_INT_TYPES = datatype.signed_integral_dtypes
UNSIGNED_INT_TYPES = datatype.unsigned_integral_dtypes

UNARY_FLOAT_OPS = (
    (device_math.ceil, host_math.ceil),
    (device_math.exp, host_math.exp),
    (device_math.sin, host_math.sin),
    (device_math.cos, host_math.cos),
    (device_math.tan, host_math.tan),
    (device_math.sinh, host_math.sinh),
    (device_math.cosh, host_math.cosh),
    (device_math.tanh, host_math.tanh),
    (device_math.sqrt, host_math.sqrt),
    (device_math.rsqrt, lambda x: 1 / host_math.sqrt(x)),
    (device_math.floor, host_math.floor),
    (device_math.log, host_math.log),
    (device_math.log2, host_math.log2),
    (device_math.abs, builtins.abs),
)

APPROX_UNARY_MATH_OPS = (
    (device_math.exp, "math.exp", "__nv{approx}_expf"),
    (device_math.sin, "math.sin", "__nv{approx}_sinf"),
    (device_math.cos, "math.cos", "__nv{approx}_cosf"),
    (device_math.tan, "math.tan", "__nv{approx}_tanf"),
    (device_math.log, "math.log", "__nv{approx}_logf"),
    (device_math.log2, "math.log2", "__nv{approx}_log2f"),
)

BINARY_FLOAT_OPS = ((device_math.atan2, host_math.atan2),)

OPERATOR_ALIAS_BINARY_OPS = (
    (device_math.add, operator.add, cl.float32),
    (device_math.sub, operator.sub, cl.float32),
    (device_math.mul, operator.mul, cl.float32),
    (device_math.truediv, operator.truediv, cl.float32),
    (device_math.floordiv, operator.floordiv, cl.int32),
    (device_math.mod, operator.mod, cl.int32),
    (device_math.floordiv, operator.floordiv, cl.float16),
    (device_math.mod, operator.mod, cl.float16),
    (device_math.floordiv, operator.floordiv, cl.float32),
    (device_math.mod, operator.mod, cl.float32),
    (device_math.floordiv, operator.floordiv, cl.float64),
    (device_math.mod, operator.mod, cl.float64),
    (device_math.bitwise_and, operator.and_, cl.int32),
    (device_math.bitwise_or, operator.or_, cl.int32),
    (device_math.bitwise_xor, operator.xor, cl.int32),
    (device_math.greater, operator.gt, cl.int32),
    (device_math.greater_equal, operator.ge, cl.int32),
    (device_math.less, operator.lt, cl.int32),
    (device_math.less_equal, operator.le, cl.int32),
    (device_math.equal, operator.eq, cl.int32),
    (device_math.not_equal, operator.ne, cl.int32),
)

FPCLASS_OPS = (
    (device_math.isinf, host_math.isinf),
    (device_math.isnan, host_math.isnan),
    (device_math.isfinite, host_math.isfinite),
    (device_math.isnormal, isnormal),
)


def assert_close_float(actual, expected, dtype):
    tol = FLOAT_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol["rel"], atol=tol["abs"])


def approx_float(expected, dtype):
    tol = FLOAT_TOLERANCES[dtype]
    return pytest.approx(expected, rel=tol["rel"], abs=tol["abs"])


def assert_special_float_values(actual, expected):
    for got, want in zip(actual.tolist(), expected, strict=True):
        if host_math.isnan(want):
            assert host_math.isnan(got)
        else:
            assert got == want
            if want == 0.0:
                assert host_math.copysign(1.0, got) == host_math.copysign(1.0, want)


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", FPCLASS_OPS)
@pytest.mark.parametrize(
    "input",
    (
        float("-0.0"),
        float("0.0"),
        float("inf"),
        float("-inf"),
        float("nan"),
        "subnormal",
    ),
)
@pytest.mark.parametrize("vector", (True, False))
def test_math_fpclass(dtype, device_op, host_op, input, vector):
    subnormal = input == "subnormal"
    if subnormal:
        smallest = _FLOAT_SMALLEST_NORMAL[dtype.bitwidth]
        input = smallest / 2

    @cl.kernel
    def kernel(out, inp):
        if vector:
            v = device_op(inp.get_base_pointer().load(count=2))
            out[0] = v[0]
        else:
            out[0] = device_op(inp[0])

    out = torch.zeros(1, dtype=torch.bool).cuda()
    inp = torch.tensor([input, input], dtype=datatype.to_torch_dtype(dtype)).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out, inp))
    if host_op == isnormal:
        expect = host_op(input, dtype.bitwidth)
    else:
        expect = host_op(input)
    got = out.cpu().item()
    assert got == expect, f"{host_op}({input}) {expect=} {got=}"


def test_isnormal_non_arithmetic_float():
    @cl.kernel
    def kernel():
        device_math.isnormal(cl.float8_e4m3fn(float("inf")))

    with pytest.raises(
        TypeCheckingError,
        match="Expected scalar or vector to satisfy constraint is_unrestricted_float",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", UNARY_FLOAT_OPS)
def test_math_unary_float(dtype, device_op, host_op):
    @cl.kernel
    def kernel(inp, out):
        out[0] = device_op(inp[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    expected = host_op(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == approx_float(expected, dtype)


@pytest.mark.parametrize(
    "device_op, op_name, libdevice_name_template", APPROX_UNARY_MATH_OPS
)
@pytest.mark.parametrize("approx", (False, True))
@pytest.mark.parametrize("vector", (False, True))
def test_math_unary_approx_fastmath(
    device_op, op_name, libdevice_name_template, approx, vector
):
    def kernel(inp, out):
        if vector:
            value = inp.get_base_pointer().load(count=2)
            out.get_base_pointer().store(device_op(value, approx=approx))
        else:
            out[0] = device_op(inp[0], approx=approx)

    count = 2 if vector else 1
    inp = make_symbolic_tensor([count], datatype.float32)
    out = make_symbolic_tensor([count], datatype.float32)
    fastmath = "afn" if approx else "none"
    approx_suffix = "_fast" if approx else ""
    libdevice_name = libdevice_name_template.format(approx=approx_suffix)
    filecheck = (
        "CHECK: " + op_name + "{{.+}}" + f"fastmath = #arith<fastmath <{fastmath}>>"
    )
    compile_kernel(
        kernel,
        signature=KernelSignature([inp, out]),
        filecheck_mlir=filecheck,
        assert_in_nvvm=f"call float @{libdevice_name}(",
    )


@pytest.mark.parametrize("approx", (False, True))
def test_math_sincos_approx_fastmath(approx):
    def kernel(inp, sin_out, cos_out):
        sin, cos = device_math.sincos(inp[0], approx=approx)
        sin_out[0] = sin
        cos_out[0] = cos

    inp = make_symbolic_tensor([1], datatype.float32)
    sin_out = make_symbolic_tensor([1], datatype.float32)
    cos_out = make_symbolic_tensor([1], datatype.float32)
    fastmath = "afn" if approx else "none"
    approx_suffix = "_fast" if approx else ""
    filecheck = (
        "CHECK: math.sincos{{.+}}" + f"fastmath = #arith<fastmath <{fastmath}>>"
    )
    compile_kernel(
        kernel,
        signature=KernelSignature([inp, sin_out, cos_out]),
        filecheck_mlir=filecheck,
        assert_in_nvvm=f"call void @__nv{approx_suffix}_sincosf(",
    )


@pytest.mark.parametrize("approx", (False, True))
@pytest.mark.parametrize("vector", (False, True))
def test_math_truediv_approx(approx, vector):
    def kernel(lhs, rhs, out):
        if vector:
            lhs_value = lhs.get_base_pointer().load(count=2)
            rhs_value = rhs.get_base_pointer().load(count=2)
            value = device_math.truediv(lhs_value, rhs_value, approx=approx)
            out.get_base_pointer().store(value)
        else:
            out[0] = device_math.truediv(lhs[0], rhs[0], approx=approx)

    count = 2 if vector else 1
    lhs = make_symbolic_tensor([count], datatype.float32)
    rhs = make_symbolic_tensor([count], datatype.float32)
    out = make_symbolic_tensor([count], datatype.float32)
    if approx:
        assert_in_nvvm = "call float @__nv_fast_fdividef("
    elif vector:
        assert_in_nvvm = "fdiv <2 x float>"
    else:
        assert_in_nvvm = "fdiv float"
    compile_kernel(
        kernel,
        signature=KernelSignature([lhs, rhs, out]),
        assert_in_nvvm=assert_in_nvvm,
    )


def _pow_test_values(dtype):
    if datatype.is_integral(dtype):
        return (2, 3, 4, 5)
    return (1.25, 1.5, 1.75, 2.0)


@pytest.mark.parametrize(
    "lhs_dt, rhs_dt, result_dt",
    (
        (cl.int32, cl.int32, cl.float32),
        (cl.uint32, cl.uint32, cl.float32),
        (cl.float16, cl.int32, cl.float16),
        (cl.float32, cl.int32, cl.float32),
        (cl.float64, cl.int32, cl.float64),
        (cl.int32, cl.float32, cl.float32),
        (cl.int32, cl.float64, cl.float64),
        (cl.float16, cl.float16, cl.float16),
        (cl.float32, cl.float32, cl.float32),
        (cl.float64, cl.float64, cl.float64),
        (cl.float16, cl.float32, cl.float32),
        (cl.float16, cl.float64, cl.float64),
        (cl.float32, cl.float64, cl.float64),
        (cl.float64, cl.float32, cl.float64),
    ),
)
@pytest.mark.parametrize("vector", (False, True))
def test_pow(lhs_dt, rhs_dt, result_dt, vector):
    @cl.kernel
    def kernel(lhs, rhs, out, operator_out):
        if vector:
            lhs_v = lhs.get_base_pointer().load(count=4)
            rhs_v = rhs.get_base_pointer().load(count=4)
            v = device_math.pow(lhs_v, rhs_v)
            operator_v = lhs_v**rhs_v
            for i in range(4):
                out[i] = out.dtype(v[i])
                operator_out[i] = operator_out.dtype(operator_v[i])
        else:
            out[0] = out.dtype(device_math.pow(lhs[0], rhs[0]))
            operator_out[0] = operator_out.dtype(lhs[0] ** rhs[0])

    lhs_torch_dt = datatype.to_torch_dtype(lhs_dt)
    rhs_torch_dt = datatype.to_torch_dtype(rhs_dt)
    result_torch_dt = datatype.to_torch_dtype(result_dt)
    count = 4 if vector else 1
    lhs = torch.tensor(_pow_test_values(lhs_dt)[:count], dtype=lhs_torch_dt).cuda()
    rhs = torch.tensor(_pow_test_values(rhs_dt)[:count], dtype=rhs_torch_dt).cuda()
    out = torch.zeros(count, dtype=result_torch_dt).cuda()
    operator_out = torch.zeros(count, dtype=result_torch_dt).cuda()

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (lhs, rhs, out, operator_out),
    )

    lhs_values = lhs.cpu().tolist()
    rhs_values = rhs.cpu().tolist()
    expected_values = [x**y for x, y in zip(lhs_values, rhs_values, strict=True)]
    expected = torch.tensor(expected_values, dtype=result_torch_dt)
    if datatype.is_float(result_dt):
        assert_close_float(out.cpu(), expected, result_dt)
        assert_close_float(operator_out.cpu(), expected, result_dt)
    else:
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(operator_out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "lhs_dt,rhs_dt,result_dt,vector_side",
    (
        (cl.float32, cl.int32, cl.float32, "lhs"),
        (cl.float64, cl.int32, cl.float64, "lhs"),
        (cl.int32, cl.float32, cl.float32, "rhs"),
        (cl.int32, cl.float64, cl.float64, "rhs"),
    ),
)
def test_pow_scalar_vector_broadcast(lhs_dt, rhs_dt, result_dt, vector_side):
    lhs_vector = vector_side == "lhs"

    @cl.kernel
    def kernel(lhs, rhs, out, operator_out):
        if lhs_vector:
            lhs_value = lhs.get_base_pointer().load(count=4)
            rhs_value = rhs[0]
        else:
            lhs_value = lhs[0]
            rhs_value = rhs.get_base_pointer().load(count=4)
        out.get_base_pointer().store(device_math.pow(lhs_value, rhs_value))
        operator_out.get_base_pointer().store(lhs_value**rhs_value)

    lhs_count = 4 if lhs_vector else 1
    rhs_count = 1 if lhs_vector else 4
    lhs_torch_dt = datatype.to_torch_dtype(lhs_dt)
    rhs_torch_dt = datatype.to_torch_dtype(rhs_dt)
    result_torch_dt = datatype.to_torch_dtype(result_dt)
    lhs = torch.tensor(_pow_test_values(lhs_dt)[:lhs_count], dtype=lhs_torch_dt).cuda()
    rhs = torch.tensor(_pow_test_values(rhs_dt)[:rhs_count], dtype=rhs_torch_dt).cuda()
    out = torch.zeros(4, dtype=result_torch_dt).cuda()
    operator_out = torch.zeros(4, dtype=result_torch_dt).cuda()

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (lhs, rhs, out, operator_out),
    )

    lhs_values = lhs.cpu().tolist()
    rhs_values = rhs.cpu().tolist()
    if lhs_vector:
        expected_values = [value ** rhs_values[0] for value in lhs_values]
    else:
        expected_values = [lhs_values[0] ** value for value in rhs_values]
    expected = torch.tensor(expected_values, dtype=result_torch_dt)
    assert_close_float(out.cpu(), expected, result_dt)
    assert_close_float(operator_out.cpu(), expected, result_dt)


@pytest.mark.parametrize(
    "lhs_dt, rhs_dt, op_name, libdevice_name_template",
    (
        # Floating-point exponent without promotion.
        (datatype.float32, datatype.float32, "math.powf", "__nv{approx}_powf"),
        (datatype.float64, datatype.float64, "math.powf", "__nv_pow"),
        # Integer exponent without promotion.
        (datatype.float32, datatype.int32, "math.fpowi", "__nv_powif"),
        (datatype.float64, datatype.int32, "math.fpowi", "__nv_powi"),
        # Integer exponent cast to i32.
        (datatype.float64, datatype.int8, "math.fpowi", "__nv_powi"),
        (datatype.float64, datatype.int16, "math.fpowi", "__nv_powi"),
        (datatype.float64, datatype.int64, "math.fpowi", "__nv_powi"),
        # promote floats to same type
        (datatype.float32, datatype.float64, "math.powf", "__nv_pow"),
        (datatype.float64, datatype.float32, "math.powf", "__nv_pow"),
        # half precision promotion
        (datatype.float16, datatype.float16, "math.powf", "__nv{approx}_powf"),
    ),
)
@pytest.mark.parametrize("approx", (False, True))
def test_pow_math_dialect(
    lhs_dt, rhs_dt, op_name, libdevice_name_template, approx
):
    def kernel(lhs, rhs, out):
        out[0] = out.dtype(device_math.pow(lhs[0], rhs[0], approx=approx))

    lhs = make_symbolic_tensor([1], lhs_dt)
    rhs = make_symbolic_tensor([1], rhs_dt)
    out = make_symbolic_tensor([1], lhs_dt)
    fastmath = "afn" if approx else "none"
    approx_suffix = "_fast" if approx else ""
    libdevice_name = libdevice_name_template.format(approx=approx_suffix)
    return_type = (
        "double" if libdevice_name in ("__nv_pow", "__nv_powi") else "float"
    )
    filecheck = (
        "CHECK: " + op_name + "{{.+}}" + f"fastmath = #arith<fastmath <{fastmath}>>"
    )
    compile_kernel(
        kernel,
        signature=KernelSignature([lhs, rhs, out]),
        filecheck_mlir=filecheck,
        assert_in_nvvm=f"call {return_type} @{libdevice_name}(",
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="math.exp2 requires Python 3.11+"
)
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_math_exp2(dtype):
    from math import exp2

    @cl.kernel
    def kernel(inp, out):
        out[0] = device_math.exp2(inp[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    expected = exp2(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == approx_float(expected, dtype)


@pytest.mark.parametrize("flush_to_zero", (False, True))
@pytest.mark.parametrize("vector", (False, True))
def test_math_exp2_ptx(flush_to_zero, vector):
    def kernel():
        arr = cl.shared_array(1, cl.float32)
        ptr = arr.get_base_pointer()
        if vector:
            value = ptr.load(count=2)
            result = device_math.exp2(value, flush_to_zero=flush_to_zero)
            ptr.store(result)
        else:
            arr[0] = device_math.exp2(arr[0], flush_to_zero=flush_to_zero)

    ftz = ".ftz" if flush_to_zero else ""
    nftz = "" if flush_to_zero else ".ftz"
    instruction = f"ex2.approx{ftz}.f32"
    other_instruction = f"ex2.approx{nftz}.f32"

    ptx_checks = f'CHECK-COUNT-{2 if vector else 1}: {instruction}'
    compile_kernel(
        kernel,
        filecheck_ptx=ptx_checks,
        assert_not_in_ptx=other_instruction,
    )


@pytest.mark.parametrize(
    "dtype", (datatype.float16, datatype.bfloat16, datatype.float64)
)
def test_math_exp2_flush_to_zero_requires_float32(dtype):
    def kernel():
        arr = cl.shared_array(1, dtype)
        arr[0] = device_math.exp2(arr[0], flush_to_zero=True)

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match="Flush to zero for exp2 requires float32 operands",
        ),
    )


def test_math_vector_splat():
    vector_dtype = cl.float32
    scalar_dtype = cl.float64

    @cl.kernel
    def kernel(inp, out):
        with cl.local_array(4, vector_dtype) as arr:
            arr[0] = 0.5
            arr[1] = 1.5
            arr[2] = 2.5
            arr[3] = 3.5
            v = arr.get_base_pointer().load(count=4)
            v = device_math.atan2(v, scalar_dtype(inp[0]))
            out.get_base_pointer().store(v)

    scalar_torch_dt = datatype.to_torch_dtype(scalar_dtype)
    out_torch_dt = datatype.to_torch_dtype(scalar_dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    inp = torch.tensor([host_inp], dtype=scalar_torch_dt, device="cuda")
    out = torch.zeros(4, dtype=out_torch_dt, device="cuda")
    scalar = inp.cpu().item()
    expected = [host_math.atan2(x, scalar) for x in (0.5, 1.5, 2.5, 3.5)]
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert_close_float(
        out.cpu(), torch.tensor(expected, dtype=out_torch_dt), scalar_dtype
    )


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", BINARY_FLOAT_OPS)
def test_math_binary_float(dtype, device_op, host_op):

    @cl.kernel
    def kernel(lhs, rhs, out):
        out[0] = device_op(lhs[0], rhs[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_lhs = torch.rand((), generator=rng).item() + 0.5
    host_rhs = torch.rand((), generator=rng).item() + 0.5
    expected = host_op(host_lhs, host_rhs)
    lhs = torch.tensor([host_lhs], dtype=torch_dt, device="cuda")
    rhs = torch.tensor([host_rhs], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    assert out[0].item() == approx_float(expected, dtype)


def test_math_binary_float_promotion():
    dt1, dt2 = cl.float16, cl.float64

    @cl.kernel
    def kernel(lhs, rhs, out):
        out[0] = device_math.atan2(dt1(lhs[0]), dt2(rhs[0]))

    tdt1 = datatype.to_torch_dtype(dt1)
    tdt2 = datatype.to_torch_dtype(dt2)
    host_lhs = torch.rand((), generator=rng).item() + 0.5
    host_rhs = torch.rand((), generator=rng).item() + 0.5
    lhs = torch.tensor([host_lhs], dtype=tdt1, device="cuda")
    rhs = torch.tensor([host_rhs], dtype=tdt2, device="cuda")
    out = torch.tensor([0.0], dtype=tdt2, device="cuda")
    expected = host_math.atan2(lhs.cpu().item(), rhs.cpu().item())
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    assert out[0].item() == approx_float(expected, dt2)


@pytest.mark.parametrize("vector", (False, True))
def test_math_fma_ir(vector):
    def kernel(x, y, z, out):
        if vector:
            xv = x.get_base_pointer().load(count=2)
            yv = y.get_base_pointer().load(count=2)
            zv = z.get_base_pointer().load(count=2)
            out.get_base_pointer().store(cl.fma(xv, yv, zv))
        else:
            out[0] = cl.fma(x[0], y[0], z[0])

    count = 2 if vector else 1
    x = make_symbolic_tensor([count], cl.float16)
    y = make_symbolic_tensor([count], cl.float32)
    z = make_symbolic_tensor([count], cl.float64)
    out = make_symbolic_tensor([count], cl.float64)
    compile_kernel(
        kernel,
        signature=KernelSignature([x, y, z, out]),
        filecheck_nvvm=(
            f"CHECK-COUNT-{count}: call double @llvm.nvvm.fma.rn.d("
        ),
        filecheck_ptx=f"CHECK-COUNT-{count}: fma.rn.f64",
    )


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("vector", (False, 2, 3, 4))
def test_math_fma(dtype, vector):
    """
    https://docs.nvidia.com/cuda/archive/12.2.1/floating-point/index.html#the-fused-multiply-add-fma

    From the docs above, a fused multiply-add has one rounding while a non-fused
    multiply and add will round twice. We can check the precision to determine
    the number of times the operation was rounded, telling us if an fma was
    really used.
    """

    @cl.kernel
    def kernel(x, y, z, out):
        if vector:
            xv = x.get_base_pointer().load(count=vector)
            yv = y.get_base_pointer().load(count=vector)
            zv = z.get_base_pointer().load(count=vector)
            out.get_base_pointer().store(cl.fma(xv, yv, zv))
        else:
            out[0] = cl.fma(x[0], y[0], z[0])

    count = vector if vector else 1
    torch_dtype = datatype.to_torch_dtype(dtype)
    eps = torch.finfo(torch_dtype).eps
    scale = 32.0
    delta = scale * eps
    x = torch.full(
        (count,),
        scale + delta,
        dtype=torch_dtype,
        device="cuda",
    )
    y = torch.full(
        (count,),
        scale - delta,
        dtype=torch_dtype,
        device="cuda",
    )
    z = torch.full(
        (count,),
        -(scale**2),
        dtype=torch_dtype,
        device="cuda",
    )
    out = torch.zeros(count, dtype=torch_dtype, device="cuda")
    expected = torch.full(
        (count,),
        -(delta**2),
        dtype=torch_dtype,
    )

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (x, y, z, out))
    assert torch.equal(out.cpu(), expected)


_ROUNDING_MODES = (
    cl.RoundingMode.RM,
    cl.RoundingMode.RN,
    cl.RoundingMode.RP,
    cl.RoundingMode.RZ,
)
_FMA_SATURATION_MODES = (
    cl.SaturationMode.NONE,
    cl.SaturationMode.SATFINITE,
    cl.SaturationMode.SAT,
)
_FMA_DATA_TYPE_CASES = tuple(
    itertools.product(
        (cl.float16, cl.bfloat16, cl.float32, cl.float64),
        (False, True),
    )
)


def _fma_supported_options(dtype):
    if dtype == cl.float16:
        yield from itertools.product(
            (cl.RoundingMode.RN,),
            (cl.SaturationMode.NONE, cl.SaturationMode.SAT),
            (False, True),
            (False,),
            (False,),
        )
        yield from itertools.product(
            (cl.RoundingMode.RN,),
            (cl.SaturationMode.NONE,),
            (False, True),
            (True,),
            (False,),
        )
        yield from itertools.product(
            (cl.RoundingMode.RN,),
            (cl.SaturationMode.NONE,),
            (False,),
            (False, True),
            (True,),
        )
    elif dtype == cl.bfloat16:
        yield from itertools.product(
            (cl.RoundingMode.RN,),
            (cl.SaturationMode.NONE,),
            (False,),
            (False, True),
            (False, True),
        )
    elif dtype == cl.float32:
        yield from itertools.product(
            _ROUNDING_MODES,
            (cl.SaturationMode.NONE, cl.SaturationMode.SAT),
            (False, True),
            (False,),
            (False,),
        )
    elif dtype == cl.float64:
        yield from itertools.product(
            _ROUNDING_MODES,
            (cl.SaturationMode.NONE,),
            (False,),
            (False,),
            (False,),
        )


def _fma_supported_cases():
    for dtype, vector in _FMA_DATA_TYPE_CASES:
        for options in _fma_supported_options(dtype):
            yield dtype, vector, *options


def _fma_all_options():
    yield from itertools.product(
        _ROUNDING_MODES,
        _FMA_SATURATION_MODES,
        (False, True),
        (False, True),
        (False, True),
    )


def _fma_unsupported_cases():
    for dtype, vector in _FMA_DATA_TYPE_CASES:
        supported = set(_fma_supported_options(dtype))
        for options in _fma_all_options():
            if options not in supported:
                yield dtype, vector, *options


def _fma_ptx_type(dtype, vector, saturation_mode):
    ptx_type = dtype.name.replace("float", "f")
    packed_f32x2 = (
        dtype == cl.float32
        and saturation_mode == cl.SaturationMode.NONE
        and get_compute_capability() >= (10, 0)
    )
    if vector and (dtype in (cl.float16, cl.bfloat16) or packed_f32x2):
        ptx_type += "x2"
    return ptx_type


def _compile_math_fma_mode(
    dtype,
    vector,
    rounding_mode,
    saturation_mode,
    flush_to_zero,
    relu,
    oob,
    *,
    assert_in_ptx=None,
    raises=None,
):
    def kernel(values, out):
        if vector:
            value = values.get_base_pointer().load(count=2)
            value = cl.fma(
                value,
                value,
                value,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                flush_to_zero=flush_to_zero,
                relu=relu,
                oob=oob,
            )
            out.get_base_pointer().store(value)
        else:
            value = values[0]
            out[0] = cl.fma(
                value,
                value,
                value,
                rounding_mode=rounding_mode,
                saturation_mode=saturation_mode,
                flush_to_zero=flush_to_zero,
                relu=relu,
                oob=oob,
            )

    count = 2 if vector else 1
    values = make_symbolic_tensor([count], dtype)
    out = make_symbolic_tensor([count], dtype)
    compile_kernel(
        kernel,
        signature=KernelSignature([values, out]),
        assert_in_ptx=assert_in_ptx,
        raises=raises,
    )


@pytest.mark.parametrize(
    (
        "dtype,vector,rounding_mode,saturation_mode,flush_to_zero,relu,oob"
    ),
    _fma_supported_cases(),
)
def test_math_fma_modes(
    dtype,
    vector,
    rounding_mode,
    saturation_mode,
    flush_to_zero,
    relu,
    oob,
):
    if oob and get_compute_capability() < (9, 0):
        pytest.skip("OOB FMA requires Hopper or newer")

    ptx_modifiers = [rounding_mode.name.lower()]
    if flush_to_zero:
        ptx_modifiers.append("ftz")
    if saturation_mode == cl.SaturationMode.SAT:
        ptx_modifiers.append("sat")
    if oob:
        ptx_modifiers.append("oob")
    if relu:
        ptx_modifiers.append("relu")
    ptx_type = _fma_ptx_type(dtype, vector, saturation_mode)
    ptx_instruction = f"fma.{'.'.join(ptx_modifiers)}.{ptx_type}"

    _compile_math_fma_mode(
        dtype,
        vector,
        rounding_mode,
        saturation_mode,
        flush_to_zero,
        relu,
        oob,
        assert_in_ptx=ptx_instruction,
    )


@pytest.mark.parametrize(
    (
        "dtype,vector,rounding_mode,saturation_mode,flush_to_zero,relu,oob"
    ),
    _fma_unsupported_cases(),
)
def test_math_fma_unsupported_modes(
    dtype,
    vector,
    rounding_mode,
    saturation_mode,
    flush_to_zero,
    relu,
    oob,
):
    exception = (
        TypeCheckingError
        if saturation_mode == cl.SaturationMode.SATFINITE
        else CompilerExecutionError
    )
    _compile_math_fma_mode(
        dtype,
        vector,
        rounding_mode,
        saturation_mode,
        flush_to_zero,
        relu,
        oob,
        raises=pytest.raises(exception),
    )


@pytest.mark.parametrize(
    "rounding_mode",
    (
        cl.RoundingMode.RA,
        cl.RoundingMode.FULL,
        cl.RoundingMode.APPROX,
        cl.RoundingMode.RZI,
    ),
)
def test_math_fma_invalid_rounding_mode(rounding_mode):
    def kernel():
        values = cl.shared_array(1, cl.float32)
        value = values[0]
        values[0] = cl.fma(
            value,
            value,
            value,
            rounding_mode=rounding_mode,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(
            TypeCheckingError,
            match="fma does not support RoundingMode",
        ),
    )


@pytest.mark.parametrize("device_op,python_op,dtype", OPERATOR_ALIAS_BINARY_OPS)
@pytest.mark.parametrize("vector", (False, True))
def test_operator_alias_binary_math(device_op, python_op, dtype, vector):
    lhs_values = (-7, 9, 5, -1, 5, 5, 6)
    rhs_values = (3, 2, -2, -4, 5, 6, 5)
    count = 4 if vector else 1

    @cl.kernel
    def kernel(lhs, rhs, out, operator_out):
        if vector:
            lhs_value = lhs.get_base_pointer().load(count=4)
            rhs_value = rhs.get_base_pointer().load(count=4)
            out.get_base_pointer().store(device_op(lhs_value, rhs_value))
            operator_out.get_base_pointer().store(python_op(lhs_value, rhs_value))
        else:
            out[0] = device_op(lhs[0], rhs[0])
            operator_out[0] = python_op(lhs[0], rhs[0])

    torch_dtype = datatype.to_torch_dtype(dtype)
    lhs = torch.tensor(lhs_values[:count], dtype=torch_dtype, device="cuda")
    rhs = torch.tensor(rhs_values[:count], dtype=torch_dtype, device="cuda")
    out = torch.zeros(count, dtype=torch_dtype, device="cuda")
    operator_out = torch.zeros(count, dtype=torch_dtype, device="cuda")
    expected = torch.tensor(
        [
            python_op(lhs_value, rhs_value)
            for lhs_value, rhs_value in zip(
                lhs_values[:count], rhs_values[:count], strict=True
            )
        ],
        dtype=torch_dtype,
    )

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (lhs, rhs, out, operator_out),
    )
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(operator_out.cpu(), expected)


@pytest.mark.parametrize(
    "operation,lhs_values,rhs_values,expected",
    (
        (
            "floordiv",
            (-0.0, 0.0, 1.0, 0.0, float("inf"), float("-inf"), 1.0, -1.0),
            (2.0, -2.0, 0.0, 0.0, 2.0, 2.0, float("inf"), float("inf")),
            (
                -0.0,
                -0.0,
                float("inf"),
                float("nan"),
                float("inf"),
                float("-inf"),
                0.0,
                -0.0,
            ),
        ),
        (
            "mod",
            (
                -4.0,
                4.0,
                -0.0,
                0.0,
                float("inf"),
                float("-inf"),
                3.0,
                -3.0,
                1.0,
                0.0,
                5.5,
                -5.5,
            ),
            (
                2.0,
                -2.0,
                2.0,
                -2.0,
                2.0,
                2.0,
                float("-inf"),
                float("inf"),
                0.0,
                0.0,
                -2.0,
                2.0,
            ),
            (
                0.0,
                -0.0,
                0.0,
                -0.0,
                float("nan"),
                float("nan"),
                float("-inf"),
                float("inf"),
                float("nan"),
                float("nan"),
                -0.5,
                0.5,
            ),
        ),
    ),
)
def test_float_division_edge_cases(operation, lhs_values, rhs_values, expected):
    count = len(lhs_values)

    @cl.kernel
    def kernel(lhs, rhs, out, operator_out):
        lhs_value = lhs.get_base_pointer().load(count=count)
        rhs_value = rhs.get_base_pointer().load(count=count)
        if operation == "floordiv":
            out.get_base_pointer().store(device_math.floordiv(lhs_value, rhs_value))
            operator_out.get_base_pointer().store(lhs_value // rhs_value)
        else:
            out.get_base_pointer().store(device_math.mod(lhs_value, rhs_value))
            operator_out.get_base_pointer().store(lhs_value % rhs_value)

    lhs = torch.tensor(lhs_values, dtype=torch.float64, device="cuda")
    rhs = torch.tensor(rhs_values, dtype=torch.float64, device="cuda")
    out = torch.zeros(count, dtype=torch.float64, device="cuda")
    operator_out = torch.zeros(count, dtype=torch.float64, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (lhs, rhs, out, operator_out),
    )
    assert_special_float_values(out.cpu(), expected)
    assert_special_float_values(operator_out.cpu(), expected)


@pytest.mark.parametrize(
    "device_op,python_op",
    (
        (device_math.floordiv, operator.floordiv),
        (device_math.mod, operator.mod),
    ),
)
@pytest.mark.parametrize("vector_side", ("lhs", "rhs"))
def test_float_division_scalar_vector_broadcast(device_op, python_op, vector_side):
    lhs_vector = vector_side == "lhs"
    lhs_values = (-7.5, 7.5, 5.5, -5.5) if lhs_vector else (-7.5,)
    rhs_values = (-2.0,) if lhs_vector else (2.0, -2.0, 3.0, -3.0)

    @cl.kernel
    def kernel(lhs, rhs, out, operator_out):
        if lhs_vector:
            lhs_value = lhs.get_base_pointer().load(count=4)
            rhs_value = rhs[0]
        else:
            lhs_value = lhs[0]
            rhs_value = rhs.get_base_pointer().load(count=4)
        out.get_base_pointer().store(device_op(lhs_value, rhs_value))
        operator_out.get_base_pointer().store(python_op(lhs_value, rhs_value))

    lhs = torch.tensor(lhs_values, dtype=torch.float32, device="cuda")
    rhs = torch.tensor(rhs_values, dtype=torch.float64, device="cuda")
    out = torch.zeros(4, dtype=torch.float64, device="cuda")
    operator_out = torch.zeros(4, dtype=torch.float64, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (lhs, rhs, out, operator_out),
    )

    lhs_host = lhs.cpu().tolist()
    rhs_host = rhs.cpu().tolist()
    if lhs_vector:
        expected_values = [python_op(value, rhs_host[0]) for value in lhs_host]
    else:
        expected_values = [python_op(lhs_host[0], value) for value in rhs_host]
    expected = torch.tensor(expected_values, dtype=torch.float64)
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(operator_out.cpu(), expected)


def test_cdiv_reexport():
    assert cl.cdiv(9, 4) == 3
    assert cl.cdiv(8, 4) == 2


@pytest.mark.parametrize(
    "dtype,lhs_values,rhs_values",
    (
        (cl.int32, (-30, 100, 30, -100), (13, -23, -13, 23)),
        (cl.uint32, (30, 100, 31, 99), (13, 23, 13, 23)),
    ),
)
@pytest.mark.parametrize("mode", ("scalar", "vector", "vector_scalar"))
def test_cdiv(dtype, lhs_values, rhs_values, mode):
    count = 1 if mode == "scalar" else 4

    @cl.kernel
    def kernel(lhs, rhs, out):
        if mode == "scalar":
            out[0] = cl.cdiv(lhs[0], rhs[0])
        else:
            lhs_value = lhs.get_base_pointer().load(count=4)
            rhs_value = (
                rhs.get_base_pointer().load(count=4) if mode == "vector" else rhs[0]
            )
            out.get_base_pointer().store(cl.cdiv(lhs_value, rhs_value))

    torch_dtype = datatype.to_torch_dtype(dtype)
    lhs = torch.tensor(lhs_values[:count], dtype=torch_dtype, device="cuda")
    rhs_count = count if mode != "vector_scalar" else 1
    rhs = torch.tensor(rhs_values[:rhs_count], dtype=torch_dtype, device="cuda")
    out = torch.zeros(count, dtype=torch_dtype, device="cuda")

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))

    rhs_host = rhs.cpu().tolist()
    expected = [
        host_math.ceil(lhs_value / rhs_host[i if mode == "vector" else 0])
        for i, lhs_value in enumerate(lhs.cpu().tolist())
    ]
    assert out.cpu().tolist() == expected


@pytest.mark.parametrize(
    "dtype,lhs_values,rhs_values,expected",
    (
        (
            cl.int32,
            (-5, 5, -5, 5),
            (3, -3, -3, 3),
            (-2, 2, -2, 2),
        ),
        (
            cl.uint32,
            (5, 6, 8, 9),
            (3, 3, 4, 5),
            (2, 0, 0, 4),
        ),
    ),
)
@pytest.mark.parametrize("vector", (False, True))
def test_integer_remainder(dtype, lhs_values, rhs_values, expected, vector):
    count = 4 if vector else 1

    @cl.kernel
    def kernel(lhs, rhs, out):
        if vector:
            lhs_value = lhs.get_base_pointer().load(count=4)
            rhs_value = rhs.get_base_pointer().load(count=4)
            out.get_base_pointer().store(
                cl.integer_remainder(lhs_value, rhs_value)
            )
        else:
            out[0] = cl.integer_remainder(lhs[0], rhs[0])

    torch_dtype = datatype.to_torch_dtype(dtype)
    lhs = torch.tensor(lhs_values[:count], dtype=torch_dtype, device="cuda")
    rhs = torch.tensor(rhs_values[:count], dtype=torch_dtype, device="cuda")
    out = torch.zeros(count, dtype=torch_dtype, device="cuda")

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))

    assert out.cpu().tolist() == list(expected[:count])


@pytest.mark.parametrize("vector_side", ("lhs", "rhs"))
def test_integer_remainder_broadcast(vector_side):
    lhs_vector = vector_side == "lhs"
    lhs_values = (-5, 5, -7, 7) if lhs_vector else (-5,)
    rhs_values = (3,) if lhs_vector else (3, -3, 2, -2)
    expected = (-2, 2, -1, 1) if lhs_vector else (-2, -2, -1, -1)

    @cl.kernel
    def kernel(lhs, rhs, out):
        lhsp, rhsp = lhs.get_base_pointer(), rhs.get_base_pointer()
        lhs_value = lhsp.load(count=4) if lhs_vector else lhs[0]
        rhs_value = rhs[0] if lhs_vector else rhsp.load(count=4)
        result = cl.integer_remainder(lhs_value, rhs_value)
        out.get_base_pointer().store(result)

    lhs = torch.tensor(lhs_values, dtype=torch.int32, device="cuda")
    rhs = torch.tensor(rhs_values, dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))

    assert out.cpu().tolist() == list(expected)


def test_integer_remainder_constants():
    @cl.kernel
    def kernel(out):
        out[0] = cl.integer_remainder(-5, 3)
        out[1] = cl.integer_remainder(5, -3)
        out[2] = cl.integer_remainder(-5, -3)
        out[3] = cl.integer_remainder(6, 3)

    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))

    assert out.cpu().tolist() == [-2, 2, -2, 0]


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype",
    (
        (cl.float32, cl.float32),
        (cl.int32, cl.float32),
        (cl.float32, cl.int32),
        (cl.bool_, cl.bool_),
    ),
)
def test_integer_remainder_requires_integer(lhs_dtype, rhs_dtype):
    def kernel():
        cl.integer_remainder(lhs_dtype(5), rhs_dtype(3))

    err = pytest.raises(TypeCheckingError, match="constraint is_integral")
    compile_kernel(kernel, raises=err)


@pytest.mark.parametrize(
    "operation,dtype,check",
    (
        (operator.floordiv, cl.float32, ("arith.divf", "math.floor")),
        (operator.mod, cl.float32, "callee = @__nv_fmodf"),
        (cl.cdiv, cl.int32, "arith.ceildivsi"),
        (cl.cdiv, cl.uint32, "arith.ceildivui"),
        (cl.integer_remainder, cl.int32, "arith.remsi"),
        (cl.integer_remainder, cl.uint32, "arith.remui"),
    ),
)
def test_division_mlir(operation, dtype, check):
    def kernel(lhs, rhs, out):
        out[0] = operation(lhs[0], rhs[0])

    lhs = make_symbolic_tensor([1], dtype)
    rhs = make_symbolic_tensor([1], dtype)
    out = make_symbolic_tensor([1], dtype)
    compile_kernel(
        kernel,
        signature=KernelSignature([lhs, rhs, out]),
        assert_in_mlir=check,
    )


@pytest.mark.parametrize("dtype", (cl.int32, cl.float32))
@pytest.mark.parametrize("vector", (False, True))
def test_operator_alias_negative(dtype, vector):
    input_values = (-7, 9, 5, -1)
    count = 4 if vector else 1

    @cl.kernel
    def kernel(inp, out, operator_out):
        if vector:
            value = inp.get_base_pointer().load(count=4)
            out.get_base_pointer().store(device_math.negative(value))
            operator_out.get_base_pointer().store(-value)
        else:
            out[0] = device_math.negative(inp[0])
            operator_out[0] = -inp[0]

    torch_dtype = datatype.to_torch_dtype(dtype)
    inp = torch.tensor(input_values[:count], dtype=torch_dtype, device="cuda")
    out = torch.zeros(count, dtype=torch_dtype, device="cuda")
    operator_out = torch.zeros(count, dtype=torch_dtype, device="cuda")
    expected = -torch.tensor(input_values[:count], dtype=torch_dtype)

    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (inp, out, operator_out),
    )
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(operator_out.cpu(), expected)


@pytest.mark.parametrize("dtype", SIGNED_INT_TYPES)
@pytest.mark.parametrize("host_inp", (-5, 0, 5))
def test_math_abs_signed_int(dtype, host_inp):
    @cl.kernel
    def kernel(inp, out):
        out[0] = device_math.abs(dtype(inp[0]))

    torch_dt = datatype.to_torch_dtype(dtype)
    expected = builtins.abs(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == expected


def test_math_abs_unsigned_int():
    # absolute value of unsigned number should be identity
    @cl.kernel
    def kernel():
        device_math.abs(cl.uint32(5.0))

    result = compile_simt(kernel, [KernelSignature([])], keep_mlir=True)
    assert "math.abs" not in result.mlir


def test_vector():
    @cl.kernel
    def kernel(out):
        with cl.local_array(4, cl.float32) as arr:
            arr[0] = 0.5
            arr[1] = 1.5
            arr[2] = 2.5
            arr[3] = 3.5
            v = arr.get_base_pointer().load(count=4)
            v = device_math.floor(v)
            out.get_base_pointer().store(v)

    out = torch.zeros(4, dtype=torch.float32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    print(out.cpu().tolist())
    torch.testing.assert_close(out.cpu().tolist(), [0.0, 1.0, 2.0, 3.0])


def test_type_error():
    @cl.kernel
    def kernel():
        device_math.sin(cl.int32(5.0))

    with pytest.raises(
        TypeCheckingError,
        match="Expected scalar or vector to satisfy constraint is_float but got int32",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


MINMAX_OPS = (
    (device_math.maximum, builtins.max),
    (device_math.minimum, builtins.min),
)

MINMAX_DTYPES = (*FLOAT_TYPES, *SIGNED_INT_TYPES, *UNSIGNED_INT_TYPES)


@pytest.mark.parametrize("dtype", MINMAX_DTYPES)
@pytest.mark.parametrize("device_op, host_op", MINMAX_OPS)
@pytest.mark.parametrize("vector", (False, True))
def test_minmax_basic(dtype, device_op, host_op, vector):
    count = 4 if vector else 1
    lhs_vals = [1, 5, 3, 8][:count]
    rhs_vals = [4, 2, 3, 6][:count]

    @cl.kernel
    def kernel(lhs, rhs, out):
        if vector:
            lhs_v = lhs.get_base_pointer().load(count=4)
            rhs_v = rhs.get_base_pointer().load(count=4)
            out.get_base_pointer().store(device_op(lhs_v, rhs_v))
        else:
            out[0] = device_op(lhs[0], rhs[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    lhs = torch.tensor(lhs_vals, dtype=torch_dt, device="cuda")
    rhs = torch.tensor(rhs_vals, dtype=torch_dt, device="cuda")
    out = torch.zeros(count, dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    expected = [host_op(a, b) for a, b in zip(lhs_vals, rhs_vals, strict=True)]
    assert out.cpu().tolist() == expected


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op", (device_math.maximum, device_math.minimum))
@pytest.mark.parametrize("propagate_nan", (False, True))
def test_minmax_nan(dtype, device_op, propagate_nan):
    @cl.kernel
    def kernel(lhs, rhs, out):
        out[0] = device_op(lhs[0], rhs[0], propagate_nan=propagate_nan)

    torch_dt = datatype.to_torch_dtype(dtype)
    lhs = torch.tensor([float("nan")], dtype=torch_dt, device="cuda")
    rhs = torch.tensor([3.0], dtype=torch_dt, device="cuda")
    out = torch.zeros(1, dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    got = out.cpu().item()
    if propagate_nan:
        assert host_math.isnan(got), f"expected NaN, got {got}"
    else:
        assert got == 3.0, f"expected 3.0, got {got}"


def test_bitwise_not():
    @cl.kernel
    def kernel(inp, out):
        tid = cl.thread_index(0)
        out[tid] = cl.bitwise_not(inp[tid])

    input = torch.tensor([0, 1, 0xffff, 13, -1], dtype=torch.int32, device="cuda")
    expected = torch.bitwise_not(input)
    output = torch.zeros_like(input)
    cl.launch(torch.cuda.current_stream(), (1,), (len(input),), kernel, (input, output))
    assert expected.tolist() == output.tolist()


@pytest.mark.parametrize("divmod_func", [divmod, cl.divmod])
def test_divmod(divmod_func):
    @cl.kernel
    def kernel(lhs, rhs, out_q, out_r):
        i = cl.block_index(0)
        j = cl.thread_index(0)
        out_q[i, j], out_r[i, j] = divmod_func(lhs[i], rhs[j])

    lhs = torch.arange(-64, 64, dtype=torch.int32, device="cuda")
    rhs = torch.arange(-8, 8, dtype=torch.int32, device="cuda")
    rhs = torch.where(rhs == 0, 1, rhs)  # avoid division by zero

    expected_q, expected_r = lhs[:, None] // rhs, lhs[:, None] % rhs
    output_q, output_r = torch.zeros_like(expected_q), torch.zeros_like(expected_r)
    cl.launch(torch.cuda.current_stream(), (len(lhs),), (len(rhs),), kernel,
              (lhs, rhs, output_q, output_r))
    assert expected_q.tolist() == output_q.tolist()
    assert expected_r.tolist() == output_r.tolist()


@pytest.mark.parametrize("vector_length", (2, 4, 8))
def test_fma_f32x2_target_lowering(vector_length):
    def kernel():
        values = cl.shared_array(vector_length * 2, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=vector_length * 2)
        ptr.store(
            cl.fma(
                value[vector_length: vector_length * 2],
                cl.float32(2.0),
                cl.float32(1.0),
            )
        )

    compile_kernel(
        kernel,
        filecheck_mlir="CHECK-COUNT-1: nvvm.fma.packed.f32x2",
        filecheck_nvvm="CHECK-COUNT-1: llvm.nvvm.fma.packed.f32x2",
        filecheck_ptx=(
            f"CHECK-COUNT-{vector_length // 2}: fma.rn.f32x2"
        ),
        gpu_name="sm_100a",
        arch="compute_100a",
    )
    compile_kernel(
        kernel,
        filecheck_mlir=f"""
            CHECK-COUNT-{max(1, vector_length // 2)}: nvvm.fma
            CHECK-NOT: nvvm.fma.packed.f32x2
        """,
        filecheck_nvvm=(
            f"CHECK-COUNT-{vector_length}: llvm.nvvm.fma.rn.f"
        ),
        filecheck_ptx=f"CHECK-COUNT-{vector_length}: fma.rn.f32",
        gpu_name="sm_100a",
        arch="compute_90",
    )


@pytest.mark.parametrize("vector_length", (2, 4, 8))
def test_add_f32x2_target_lowering(vector_length):
    def kernel():
        values = cl.shared_array(vector_length, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=vector_length)
        ptr.store(value + value)

    compile_kernel(
        kernel,
        filecheck_mlir="""
            CHECK: arith.addf
            CHECK-NOT: nvvm.add.packed.f32x2
        """,
        filecheck_ptx=f"CHECK-COUNT-{vector_length // 2}: add.f32x2",
        gpu_name="sm_100a",
        arch="compute_100a",
    )

    compile_kernel(
        kernel,
        filecheck_mlir="""
            CHECK: arith.addf
            CHECK-NOT: nvvm.add.packed.f32x2
        """,
        filecheck_ptx=f"CHECK-COUNT-{vector_length}: add.f32",
        gpu_name="sm_100a",
        arch="compute_90",
    )


@pytest.mark.parametrize(
    "op_name,device_op",
    (("add", cl.add), ("sub", cl.sub)),
)
@pytest.mark.parametrize("rounding_mode", (*_ROUNDING_MODES, None))
@pytest.mark.parametrize("flush_to_zero", (False, True))
def test_arith_f32x2_modes(op_name, device_op, rounding_mode, flush_to_zero):
    def kernel():
        values = cl.shared_array(2, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=2)
        ptr.store(device_op(
            value,
            value,
            rounding_mode=rounding_mode,
            flush_to_zero=flush_to_zero,
        ))

    rnd = "rn" if rounding_mode is None else rounding_mode.name.lower()
    ftz = ".ftz" if flush_to_zero else ""
    explicit_packed = rounding_mode is not None or flush_to_zero
    if explicit_packed:
        filecheck_mlir = f"CHECK-COUNT-1: nvvm.{op_name}.packed.f32x2"
        filecheck_nvvm = (
            f"CHECK-COUNT-1: llvm.nvvm.{op_name}{ftz}.packed.f32x2"
        )
        filecheck_ptx = f"CHECK-COUNT-1: {op_name}.{rnd}{ftz}.f32x2"
    else:
        filecheck_mlir = f"""
            CHECK: arith.{op_name}f
        """
        filecheck_nvvm = None
        filecheck_ptx = f"CHECK-COUNT-1: {op_name}.f32x2"
    compile_kernel(
        kernel,
        filecheck_mlir=filecheck_mlir,
        filecheck_nvvm=filecheck_nvvm,
        filecheck_ptx=filecheck_ptx,
        gpu_name="sm_100a",
        arch="compute_100a",
    )


@pytest.mark.parametrize(
    "op_name,device_op",
    (("add", cl.add), ("sub", cl.sub)),
)
@pytest.mark.parametrize("rounding_mode", _ROUNDING_MODES)
@pytest.mark.parametrize("flush_to_zero", (False, True))
def test_arith_f32_scalar_modes(
    op_name, device_op, rounding_mode, flush_to_zero
):
    def kernel():
        values = cl.shared_array(1, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load()
        ptr.store(device_op(
            value,
            value,
            rounding_mode=rounding_mode,
            flush_to_zero=flush_to_zero,
        ))

    rnd = rounding_mode.name.lower()
    ftz = ".ftz" if flush_to_zero else ""
    compile_kernel(
        kernel,
        assert_in_mlir=f"nvvm.{op_name}f",
        filecheck_ptx=f"CHECK-COUNT-1: {op_name}.{rnd}{ftz}.f32",
        gpu_name="sm_100a",
        arch="compute_100a",
    )


@pytest.mark.parametrize(
    "op_name,device_op",
    (("add", cl.add), ("sub", cl.sub)),
)
@pytest.mark.parametrize("rounding_mode", _ROUNDING_MODES)
@pytest.mark.parametrize("flush_to_zero", (False, True))
def test_arith_f32x2_nvvm_toolchain_packing(
    monkeypatch, op_name, device_op, rounding_mode, flush_to_zero
):
    original_supports = TargetInfo.supports

    def supports_without_packed_f32x2(self, feature):
        if feature is TargetFeature.PACKED_F32X2:
            return False
        return original_supports(self, feature)

    monkeypatch.setattr(TargetInfo, "supports", supports_without_packed_f32x2)

    def kernel():
        values = cl.shared_array(2, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=2)
        ptr.store(device_op(
            value,
            value,
            rounding_mode=rounding_mode,
            flush_to_zero=flush_to_zero,
        ))

    rnd = rounding_mode.name.lower()
    ftz = ".ftz" if flush_to_zero else ""
    # If this starts emitting packed PTX, remove the special PACKED_F32X2
    # target lowering and rely on the toolchain to pack nvvm.addf/subf.
    compile_kernel(
        kernel,
        assert_in_mlir=f"nvvm.{op_name}f",
        assert_not_in_ptx="f32x2",
        filecheck_ptx=f"CHECK-COUNT-2: {op_name}.{rnd}{ftz}.f32",
        gpu_name="sm_100a",
        arch="compute_100a",
    )


def test_add_f32x2_odd_vector_fallback():
    vector_length = 3

    def kernel():
        values = cl.shared_array(vector_length, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=vector_length)
        ptr.store(cl.add(value, value))

    compile_kernel(
        kernel,
        filecheck_mlir="""
            CHECK: arith.addf
            CHECK-NOT: nvvm.add.packed.f32x2
        """,
        gpu_name="sm_100a",
        arch="compute_100a",
    )


@pytest.mark.parametrize(
    "dtype,ptx_type",
    (
        (cl.float16, "f16"),
        (cl.bfloat16, "bf16"),
    ),
)
@pytest.mark.parametrize("vector_length", (3, 4))
def test_fma_long_vector_pair_lowering(dtype, ptx_type, vector_length):
    def kernel():
        values = cl.shared_array(vector_length, dtype)
        ptr = values.get_base_pointer()
        value = ptr.load(count=vector_length)
        ptr.store(cl.fma(value, value, value))

    pair_count = vector_length // 2
    operation_count = pair_count + vector_length % 2
    compile_kernel(
        kernel,
        filecheck_mlir=f"""
            CHECK-COUNT-{operation_count}: nvvm.fma
            CHECK-NOT: nvvm.fma.packed.f32x2
        """,
        filecheck_nvvm=(
            f"CHECK-COUNT-{pair_count}: llvm.nvvm.fma.rn.{ptx_type}x2"
        ),
        filecheck_ptx=(
            f"CHECK-COUNT-{pair_count}: fma.rn.{ptx_type}x2"
        ),
        gpu_name="sm_100a",
        arch="compute_100a",
    )


def test_fma_f32x2_odd_vector_fallback():
    vector_length = 3

    def kernel():
        values = cl.shared_array(vector_length, cl.float32)
        ptr = values.get_base_pointer()
        value = ptr.load(count=vector_length)
        ptr.store(cl.fma(value, value, value))

    compile_kernel(
        kernel,
        filecheck_mlir="""
            CHECK-COUNT-2: nvvm.fma
            CHECK-NOT: nvvm.fma.packed.f32x2
        """,
        filecheck_nvvm="CHECK-COUNT-3: llvm.nvvm.fma.rn.f",
        filecheck_ptx="CHECK-COUNT-3: fma.rn.f32",
        gpu_name="sm_100a",
        arch="compute_100a",
    )
