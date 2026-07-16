# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._ir.op_defs import RawNVVMIntrinsic, BitCast
from ..type import (
    DTypeConstructor,
    ScalarTy,
    PointerTy,
    VectorTy,
    type_bitwidth,
)
from cuda.lang._ir.type_checking_helpers import require_dtype_spec
from cuda.lang._stub import core_api
from cuda.lang._exception import TypeCheckingError
import cuda.lang._datatype as datatype
from cuda.tile._datatype import int32
from cuda.tile._ir.ir import Var, add_operation
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_int


_registry = ImplRegistry()
impl = _registry.impl


def core_api_impl_registry() -> ImplRegistry:
    return _registry


@impl(core_api.thread_index, fixed_args=["tid"])
@impl(core_api.thread_count, fixed_args=["ntid"])
@impl(core_api.block_index, fixed_args=["ctaid"])
@impl(core_api.block_count, fixed_args=["nctaid"])
@impl(core_api.cluster_index, fixed_args=["clusterid"])
@impl(core_api.cluster_count, fixed_args=["nclusterid"])
@impl(core_api.block_in_cluster_index, fixed_args=["cluster.ctaid"])
@impl(core_api.block_in_cluster_count, fixed_args=["cluster.nctaid"])
def read_gridlike_special_register_impl(sreg_name: str, axis: Var) -> Var:
    axis = require_constant_int(axis)
    if axis not in (0, 1, 2):
        raise TypeCheckingError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    axis_name = "xyz"[axis]
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(int32),
        intrinsic=f"llvm.nvvm.read.ptx.sreg.{sreg_name}.{axis_name}",
        operands_=()
    )


def bitcast(x: Var[ScalarTy | PointerTy | VectorTy], dtype: datatype.DType):
    x_ty = x.get_type()
    x_dtype = x_ty.tensor_dtype()
    if isinstance(dtype, VectorTy):
        # dead code for now - users have no way to construct vector dtypes
        raise TypeCheckingError("bitcast to vector is not supported")
    if datatype.bool_ in (dtype, x_dtype):
        raise TypeCheckingError("bitcast to or from bool is not supported")
    x_bitwidth = type_bitwidth(x_ty)
    if x_bitwidth != dtype.bitwidth:
        raise TypeCheckingError(
            "bitcast requires input value's type and output type to have the "
            f"same bitwidth, but input type is {x_bitwidth} bits and output "
            f"dtype has {dtype.bitwidth} bits"
        )

    # at the mlir level, we only have bitcast, inttoptr, and ptrtoint. If we
    # have a pointer, cast it to an int first then to the real dst type.
    # If we are casting *to* a pointer, first cast to int then the real dst
    # type. If both src and dst are pointer types, use a regular bitcast.
    # ir2mlir will use an address space cast.

    src_dtype, dst_dtype = x_dtype, dtype
    src_is_ptr = datatype.is_pointer_dtype(src_dtype)
    dst_is_ptr = datatype.is_pointer_dtype(dst_dtype)
    src_is_int_scalar = isinstance(x_ty, ScalarTy) and datatype.is_integral(src_dtype)
    dst_is_int_scalar = datatype.is_integral(dst_dtype)

    def direct_bitcast():
        res_ty = PointerTy(dtype) if datatype.is_pointer_dtype(dtype) else ScalarTy(dtype)
        return add_operation(BitCast, res_ty, x=x)

    def bitcast_through_int():
        intermediate_type = getattr(datatype, f'int{x_bitwidth}')
        first = bitcast(x, intermediate_type)
        return bitcast(first, dtype)

    if src_is_ptr and dst_is_ptr:
        return direct_bitcast()

    if src_is_ptr:
        if dst_is_int_scalar:
            return direct_bitcast()
        return bitcast_through_int()

    if dst_is_ptr:
        if src_is_int_scalar:
            return direct_bitcast()
        return bitcast_through_int()

    # no pointer involved: direct bitcast
    return direct_bitcast()


@impl(core_api.bitcast)
def bitcast_impl(x: Var[ScalarTy | PointerTy | VectorTy], dtype: Var[DTypeConstructor]):
    dtype = require_dtype_spec(dtype)
    return bitcast(x, dtype)
