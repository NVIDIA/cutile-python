# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import CachePolicy
from cuda.lang._exception import InvalidValueError, TypeCheckingError
from cuda.lang._ir.op_defs import RawLLVMIntrinsic, BitCast
from cuda.tile import MemoryScope
from .inline_ptx_impl import inline_ptx
from ..type import (
    DTypeConstructor,
    MemorySpace,
    ScalarTy,
    PointerTy,
    VectorTy,
    make_rank0_ty,
    type_bitwidth,
)
from cuda.lang._ir.type_checking_helpers import (
    require_constant_enum,
    require_dtype_spec,
    require_integral_scalar_type,
    require_pointer_in_memory_space,
    require_pointer_type,
    require_scalar_type,
)
from cuda.lang._stub import core_api, cache_policy
from cuda.lang._stub.types import Vector
import cuda.lang._datatype as datatype
from cuda.tile._datatype import int32, opaque_pointer_dtype, pointer_dtype
from cuda.tile._ir.arithmetic_ops import astype, binary_bitwise_tensorlike
from cuda.tile._ir.core_ops import bind_method, strictly_typed_const
from cuda.tile._ir.ir import Var, add_operation, add_operation_variadic
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
        RawLLVMIntrinsic,
        ScalarTy(int32),
        intrinsic=f"llvm.nvvm.read.ptx.sreg.{sreg_name}.{axis_name}",
        operands_=()
    )


def reinterpret_to(x: Var, result_ty: VectorTy | ScalarTy | PointerTy):
    """Reinterpret ``x``'s bits as ``result_ty``, which must have the same total
    bitwidth. Scalars, vectors, and pointers may be mixed freely on either
    side; bytes are packed and re-split little-endian."""
    x_ty = x.get_type()
    x_dtype = x_ty.tensor_dtype()
    dst_dtype = result_ty.tensor_dtype()
    if datatype.bool_ in (dst_dtype, x_dtype):
        raise TypeCheckingError("bitcast to or from bool is not supported")
    x_bitwidth = type_bitwidth(x_ty)
    dst_bitwidth = type_bitwidth(result_ty)
    if x_bitwidth != dst_bitwidth:
        raise TypeCheckingError(
            "bitcast requires input value's type and output type to have the "
            f"same bitwidth, but input type is {x_bitwidth} bits and output "
            f"dtype has {dst_bitwidth} bits"
        )

    src_is_ptr = datatype.is_pointer_dtype(x_dtype)
    dst_is_ptr = datatype.is_pointer_dtype(dst_dtype)

    def integer_type_like(ty: ScalarTy | PointerTy | VectorTy):
        dtype = getattr(datatype, f"int{ty.tensor_dtype().bitwidth}")
        if isinstance(ty, VectorTy):
            return VectorTy(dtype, ty.length)
        return ScalarTy(dtype)

    if src_is_ptr and dst_is_ptr and x_ty.tensor_shape() == result_ty.tensor_shape():
        return add_operation(BitCast, result_ty, x=x)
    if src_is_ptr:
        integer = add_operation(BitCast, integer_type_like(x_ty), x=x)
        return reinterpret_to(integer, result_ty)
    if dst_is_ptr:
        integer = reinterpret_to(x, integer_type_like(result_ty))
        return add_operation(BitCast, result_ty, x=integer)
    return add_operation(BitCast, result_ty, x=x)


def bitcast(x: Var[ScalarTy | PointerTy | VectorTy], dtype: datatype.DType):
    x_ty = x.get_type()
    if isinstance(x_ty, VectorTy):
        elem_dtype = x_ty.element_dtype
        if elem_dtype.bitwidth != dtype.bitwidth:
            raise TypeCheckingError(
                "Vector element and target dtype must have the same bitwidth "
                f"(element is {elem_dtype.bitwidth} bits, target is {dtype.bitwidth} bits)"
            )
        return reinterpret_to(x, VectorTy(dtype, x_ty.length))
    return reinterpret_to(x, make_rank0_ty(dtype))


def reinterpret_as_scalar(x: Var[VectorTy], dtype: datatype.DType):
    """Reinterpret a whole vector's bits as a single scalar of ``dtype``, whose
    bitwidth must equal the vector's total bitwidth. Bytes are packed
    little-endian."""
    if datatype.is_pointer_dtype(dtype):
        raise TypeCheckingError(
            "reinterpret_as_scalar only accepts a scalar dtype."
        )
    return reinterpret_to(x, ScalarTy(dtype))


def reinterpret_as_vector(x: Var[VectorTy], dtype: datatype.DType, length: int):
    """Reinterpret a whole vector's bits as ``Vector[dtype, length]``, whose
    total bitwidth must equal the source vector's total bitwidth. Bytes are
    re-split little-endian."""
    return reinterpret_to(x, VectorTy(dtype, length))


@impl(core_api.bitcast)
def bitcast_impl(x: Var[ScalarTy | PointerTy | VectorTy], dtype: Var[DTypeConstructor]):
    return bitcast(x, require_dtype_spec(dtype))


@impl(getattr, overload=(VectorTy, "bitcast"))
def getattr_vector_bitcast(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.bitcast)


@impl(Vector.bitcast)
def vector_bitcast_impl(self: Var[VectorTy], dtype: Var[DTypeConstructor]):
    return bitcast(self, require_dtype_spec(dtype))


@impl(getattr, overload=(VectorTy, "reinterpret_as_scalar"))
def getattr_vector_reinterpret_as_scalar(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.reinterpret_as_scalar)


@impl(Vector.reinterpret_as_scalar)
def vector_reinterpret_as_scalar_impl(self: Var[VectorTy], dtype: Var[DTypeConstructor]):
    return reinterpret_as_scalar(self, require_dtype_spec(dtype))


@impl(getattr, overload=(VectorTy, "reinterpret_as_vector"))
def getattr_vector_reinterpret_as_vector(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.reinterpret_as_vector)


@impl(Vector.reinterpret_as_vector)
def vector_reinterpret_as_vector_impl(
    self: Var[VectorTy], dtype: Var[DTypeConstructor], length: Var
):
    return reinterpret_as_vector(
        self, require_dtype_spec(dtype), require_constant_int(length)
    )


@impl(core_api.map_shared_to_cluster)
def map_shared_to_cluster_impl(pointer: Var, rank: Var):
    ptr_ty = require_pointer_type(pointer)
    rank = astype(rank, datatype.int32)
    require_pointer_in_memory_space(pointer, (MemorySpace.SHARED,))
    if ptr_ty.opaque:
        result_dtype = opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER)
    else:
        result_dtype = pointer_dtype(ptr_ty.pointee_dtype, MemorySpace.SHARED_CLUSTER)
    result_ty = PointerTy(result_dtype)
    return add_operation(
        RawLLVMIntrinsic,
        result_ty,
        intrinsic="llvm.nvvm.mapa.shared.cluster",
        operands_=(pointer, rank),
    )


@impl(core_api.map_shared_to_leader_block)
def map_shared_to_leader_block(pointer: Var):
    spaces = (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER)
    pointer_type = require_pointer_in_memory_space(pointer, spaces)
    int_value = bitcast(pointer, datatype.uint32)
    mask = core_api.shared_cluster_leader_bit_mask()
    mask = strictly_typed_const(mask, ScalarTy(datatype.uint32))
    mapped = binary_bitwise_tensorlike("and_", int_value, mask)
    # TODO: should this be shared_cluster memory space?
    return bitcast(mapped, pointer_type.pointer_dtype)


@impl(core_api.setmaxregister_decrease)
def impl_setmaxregister_decrease(number_of_registers: Var[ScalarTy]):
    value = require_constant_int(number_of_registers)
    inline_ptx(f"setmaxnreg.dec.sync.aligned.u32 {value};")


@impl(core_api.setmaxregister_increase)
def impl_setmaxregister_increase(number_of_registers: Var[ScalarTy]):
    value = require_constant_int(number_of_registers)
    inline_ptx(f"setmaxnreg.inc.sync.aligned.u32 {value};",)


@impl(cache_policy.create_range_cache_policy)
def impl_create_range_cache_policy(
    base_address,
    primary_size,
    total_size,
    primary_policy,
    secondary_policy,
):
    require_integral_scalar_type(primary_size)
    primary_size = astype(primary_size, datatype.int32)
    require_integral_scalar_type(total_size)
    total_size = astype(total_size, datatype.int32)
    require_pointer_type(base_address)
    primary_policy = require_constant_enum(primary_policy, CachePolicy)
    secondary_policy = require_constant_enum(secondary_policy, CachePolicy)
    valid = (CachePolicy.L2_EVICT_FIRST, CachePolicy.L2_EVICT_UNCHANGED)
    if secondary_policy not in valid:
        raise InvalidValueError(
            "Secondary cache policy may only be " + " or ".join(str(i) for i in valid)
        )
    return inline_ptx(
        f"createpolicy.range.{primary_policy.value}.{secondary_policy.value}.b64"
        f" %0, [%1], %2, %3;",
        datatype.int64, base_address, primary_size, total_size
    )[0]


@impl(cache_policy.create_fractional_cache_policy)
def impl_create_fractional_cache_policy(
    primary_policy,
    fraction,
    secondary_policy,
):
    primary_policy = require_constant_enum(primary_policy, CachePolicy)
    require_scalar_type(fraction, datatype.is_unrestricted_float)
    fraction = astype(fraction, datatype.float32)
    secondary_policy = require_constant_enum(secondary_policy, CachePolicy)
    valid = (CachePolicy.L2_EVICT_FIRST, CachePolicy.L2_EVICT_UNCHANGED)
    if secondary_policy not in valid:
        raise InvalidValueError(
            "Secondary cache policy may only be " + " or ".join(str(i) for i in valid)
        )
    return inline_ptx(
        f"createpolicy.fractional.{primary_policy.value}.{secondary_policy.value}.b64 %0, %1;",
        datatype.int64, fraction
    )[0]


@impl(core_api.memory_barrier)
def memory_barrier_impl(scope: Var) -> None:
    scope2intrin = {
        MemoryScope.BLOCK: "llvm.nvvm.membar.cta",
        MemoryScope.CLUSTER: "llvm.nvvm.fence_sc_cluster",
        MemoryScope.DEVICE: "llvm.nvvm.membar.gl",
        MemoryScope.SYS: "llvm.nvvm.membar.sys",
    }
    scope = require_constant_enum(scope, MemoryScope)
    intrinsic = scope2intrin.get(scope)
    if intrinsic is None:
        valid = ", ".join(x._name_ for x in scope2intrin.keys())
        raise InvalidValueError(f"Invalid memory scope '{scope}'. Valid values: {valid}")
    add_operation_variadic(
        RawLLVMIntrinsic, (),
        intrinsic=intrinsic,
        operands_=()
    )


@impl(core_api.grid_dependency_control_wait, fixed_args=["wait"])
@impl(core_api.grid_dependency_control_launch_dependents, fixed_args=["launch.dependents"])
def grid_dependency_control_action_impl(action: str) -> None:
    add_operation_variadic(
        RawLLVMIntrinsic, (),
        intrinsic="llvm.nvvm."
                  "griddepcontrol." + action,
        operands_=()
    )
