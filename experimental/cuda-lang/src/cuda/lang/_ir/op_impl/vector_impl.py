# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import operator
from typing import Callable

from cuda.tile._datatype import is_pointer_dtype
from cuda.tile._ir.op_impl import (
    WILDCARD,
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_dtype_spec,
    require_constant_slice,
)
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile._ir.core_ops import bind_method, build_tuple, loosely_typed_const
from cuda.tile._ir.ops import strictly_typed_const, slice_impl
from cuda.tile._ir.ops_utils import promote_dtypes
from cuda.tile._ir.type import LooselyTypedScalar
from cuda.lang._exception import InternalError, TypeCheckingError, InvalidValueError
import cuda.lang._datatype as datatype
from cuda.lang._enums import VectorReduction
from ..type_checking_helpers import require_vector_type, require_scalar_type
from ..op_defs import VectorConstruct, VectorGetItem, VectorInsert, VectorReduce
from ..type import PointerTy, ScalarTy, VectorTy, SliceType
from ..._stub.types import Vector
from ..ir import Var, add_operation


_registry = ImplRegistry()
impl = _registry.impl


def vector_impl_registry() -> ImplRegistry:
    return _registry


def vector_construct(
    res_type: VectorTy, elements: tuple[Var[ScalarTy | PointerTy], ...]
) -> Var[VectorTy]:
    if len(elements) != res_type.length:
        raise InternalError(
            f"Expected {res_type.length} vector elements but got {len(elements)}"
        )
    return add_operation(VectorConstruct, res_type, elements=elements)


def vector_with_item(
    vector: Var[VectorTy],
    key: int | Var[ScalarTy],
    value: Var[ScalarTy | PointerTy],
):
    ty = require_vector_type(vector)
    if isinstance(key, int):
        key = strictly_typed_const(key, ScalarTy(datatype.int32))
    key = implicit_cast(key, datatype.int32, "vector setitem index to int32")
    value = implicit_cast(
        value, ty.element_dtype, "vector setitem cast RHS to value type"
    )
    return add_operation(
        VectorInsert,
        vector.get_type(),
        vector=vector,
        value=value,
        index=key,
    )


def _vector_constructor_element_dtype(elements: tuple[Var, ...]) -> datatype.DType:
    loose_types = [element.get_loose_type() for element in elements]
    concrete_dtypes = [
        lt.tensor_dtype()
        for lt in loose_types
        if not isinstance(lt, LooselyTypedScalar)
    ]

    if not concrete_dtypes:
        element_dtype = loose_types[0].tensor_dtype()
        for lt in loose_types[1:]:
            element_dtype = promote_dtypes(element_dtype, lt.tensor_dtype())
        return element_dtype

    element_dtype = concrete_dtypes[0]
    for concrete_dtype in concrete_dtypes[1:]:
        element_dtype = promote_dtypes(element_dtype, concrete_dtype)
    return element_dtype


def _optional_vector_constructor_dtype(dtype: Var) -> datatype.DType | None:
    if dtype.is_constant() and dtype.get_constant() is None:
        return None
    return require_dtype_spec(dtype)


def _require_vector_constructor_element(element: Var, index: int) -> None:
    try:
        require_scalar_type(element)
    except TypeCheckingError as e:
        raise TypeCheckingError(f"Vector() element {index}: {str(e)}")


@impl(Vector)
def vector_constructor_impl(elements: tuple[Var, ...], dtype: Var) -> Var[VectorTy]:
    if not elements:
        raise TypeCheckingError("Vector() expects at least one element")

    for index, element in enumerate(elements):
        _require_vector_constructor_element(element, index)

    explicit_dtype = _optional_vector_constructor_dtype(dtype)
    element_dtype = (
        explicit_dtype
        if explicit_dtype is not None
        else _vector_constructor_element_dtype(elements)
    )
    values = tuple(
        implicit_cast(element, element_dtype, f"Vector() element {index}")
        for index, element in enumerate(elements)
    )
    return vector_construct(VectorTy(element_dtype, len(values)), values)


impl(slice)(slice_impl)


@impl(operator.getitem, overload=(VectorTy, SliceType))
def vector_slice_impl(object: Var[VectorTy], key: Var[SliceType]):
    s = require_constant_slice(key)
    vt = require_vector_type(object)

    if s.step == 0:
        raise InvalidValueError("Slice step cannot be zero")

    start, stop, step = s.indices(vt.length)
    indices = tuple(range(start, stop, step))
    if len(indices) == 0:
        raise InvalidValueError(
            "Slice is invalid because slice would result in length-0 vector"
        )

    new_vt = VectorTy(vt.element_dtype, length=len(indices))
    elements = tuple(
        vector_getitem(object, loosely_typed_const(src_index))
        for src_index in indices
    )
    return vector_construct(new_vt, elements)


@impl(tuple, overload=(VectorTy,))
def vector_tuple_impl(iterable: Var[VectorTy]):
    length = iterable.get_type().length
    return build_tuple(
        vector_getitem(
            iterable,
            strictly_typed_const(index, ScalarTy(datatype.int32)),
        )
        for index in range(length)
    )


@impl(len, overload=(VectorTy,))
def vector_len_impl(x: Var[VectorTy]):
    return loosely_typed_const(x.get_type().length)


def vector_elementwise_apply(
    callable: Callable[[Var[ScalarTy], ...], Var[ScalarTy]], *vectors
):
    vector_types = [require_vector_type(v) for v in vectors]
    if len(vectors) == 0:
        raise InternalError("Expected at least one vector")
    length = vector_types[0].length
    if not all(v.length == length for v in vector_types[1:]):
        raise InternalError("Expected all vectors to have same length")

    def apply_one(i: int):
        index = strictly_typed_const(i, ScalarTy(datatype.int32))
        operands = [vector_getitem(x, index) for x in vectors]
        element = callable(*operands)
        return element

    elements = tuple(apply_one(i) for i in range(length))
    element_type = elements[0].get_type()
    if not isinstance(element_type, ScalarTy):
        raise InternalError(
            "Expected elementwise application of function to vector to "
            f"return a scalar but got {element_type}"
        )
    if not all(element.get_type() == element_type for element in elements[1:]):
        raise InternalError("Expected all elementwise results to have the same type")
    return vector_construct(VectorTy(element_type.dtype, length), elements)


@impl(operator.setitem, overload=(VectorTy, WILDCARD, WILDCARD))
def vector_setitem_impl(object: Var[VectorTy], key: Var, value: Var):
    raise TypeCheckingError(
        "Vectors are immutable. Consider calling vector.with_item() instead"
    )


@impl(getattr, overload=(VectorTy, "with_item"))
def getattr_vector_with_item(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.with_item)


@impl(Vector.with_item)
def vector_with_item_impl(
    self: Var[VectorTy],
    index: Var[ScalarTy],
    value: Var[ScalarTy | PointerTy],
) -> Var[VectorTy]:
    return vector_with_item(self, index, value)


@impl(getattr, overload=(VectorTy, "astype"))
def getattr_vector_astype(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.astype)


@impl(Vector.astype)
def vector_astype_impl(self: Var[VectorTy], dtype: Var) -> Var[VectorTy]:
    dtype = require_dtype_spec(dtype)
    if is_pointer_dtype(self.get_type().element_dtype) or is_pointer_dtype(dtype):
        return implicit_cast(self, dtype, "Vector.astype")
    return astype(self, dtype)


@impl(getattr, overload=(VectorTy, "reduce"))
def getattr_vector_reduce(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.reduce)


@impl(Vector.reduce)
def vector_reduce_impl(
    self: Var[VectorTy], op: Var, propagate_nan: Var, reassociate: Var
) -> Var[ScalarTy]:
    vector_type = require_vector_type(self)
    kind = require_constant_enum(op, VectorReduction)
    propagate_nan_value = require_constant_bool(propagate_nan)
    reassociate_value = require_constant_bool(reassociate)
    dtype = vector_type.element_dtype

    if propagate_nan_value and kind not in (
        VectorReduction.max,
        VectorReduction.min,
    ):
        raise TypeCheckingError(
            "propagate_nan is valid only for min and max reductions"
        )

    bitwise_kinds = (
        VectorReduction.bitwise_and,
        VectorReduction.bitwise_or,
        VectorReduction.bitwise_xor,
    )
    if datatype.is_boolean(dtype):
        supported = kind in bitwise_kinds
    elif datatype.is_integral(dtype):
        supported = True
    elif datatype.is_unrestricted_float(dtype):
        supported = kind not in bitwise_kinds
    else:
        supported = False

    if not supported:
        raise TypeCheckingError(
            f"Vector reduction {kind.value} does not support {dtype}"
        )

    if reassociate_value and not (
        datatype.is_unrestricted_float(dtype)
        and kind in (VectorReduction.add, VectorReduction.mul)
    ):
        raise TypeCheckingError(
            "reassociate is valid only for floating-point add and multiply "
            "vector reductions"
        )

    return add_operation(
        VectorReduce,
        ScalarTy(dtype),
        x=self,
        kind=kind,
        propagate_nan=propagate_nan_value,
        reassociate=reassociate_value,
    )


@impl(operator.getitem, overload=(VectorTy, WILDCARD))
def vector_getitem(
    object: Var[VectorTy], key: Var[ScalarTy]
) -> Var[ScalarTy | PointerTy]:
    result_dtype = object.get_type().element_dtype
    index = implicit_cast(key, datatype.int32, "vector getitem index")
    result_ty = (
        PointerTy(result_dtype)
        if is_pointer_dtype(result_dtype)
        else ScalarTy(result_dtype)
    )
    return add_operation(
        VectorGetItem,
        result_ty,
        x=object,
        index=index,
    )
