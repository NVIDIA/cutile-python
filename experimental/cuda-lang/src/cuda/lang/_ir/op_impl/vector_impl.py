# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import operator
from typing import Callable
from cuda.tile._ir.op_impl import (
    WILDCARD,
    ImplRegistry,
    require_dtype_spec,
)
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile._ir.core_ops import bind_method, build_tuple, loosely_typed_const
from cuda.tile._ir.ops import strictly_typed_const
from cuda.tile._ir.ops_utils import promote_dtypes
from cuda.tile._ir.type import LooselyTypedScalar
from cuda.lang._exception import InternalError, TypeCheckingError
import cuda.lang._datatype as datatype
from ..type_checking_helpers import require_vector_type, require_scalar_type
from ..op_defs import RawMLIROperation, VectorGetItem
from ..type import ScalarTy, Type, VectorTy
from ..._stub.types import Vector
from ..ir import Var, add_operation


_registry = ImplRegistry()
impl = _registry.impl


def vector_impl_registry() -> ImplRegistry:
    return _registry


def vector_undef(res_type: Type):
    return add_operation(
        RawMLIROperation, res_type, op_name="llvm.mlir.undef", operands_=()
    )


def vector_with_item(vector: Var[VectorTy], key: int | Var[ScalarTy], value: Var[ScalarTy]):
    ty = require_vector_type(vector)
    if isinstance(key, int):
        key = strictly_typed_const(key, ScalarTy(datatype.int32))
    key = implicit_cast(key, datatype.int32, "vector setitem index to int32")
    value = implicit_cast(
        value, ty.element_dtype, "vector setitem cast RHS to value type"
    )
    return add_operation(
        RawMLIROperation,
        vector.get_type(),
        op_name="llvm.insertelement",
        operands_=(vector, value, key),
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
    res = vector_undef(VectorTy(element_dtype, len(elements)))
    for index, element in enumerate(elements):
        value = implicit_cast(element, element_dtype, f"Vector() element {index}")
        res = vector_with_item(res, index, value)
    return res


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

    first_element = apply_one(0)
    element_type = first_element.get_type()
    if not isinstance(element_type, ScalarTy):
        raise InternalError(
            "Expected elementwise application of function to vector to "
            f"return a scalar but got {element_type}"
        )

    res = vector_undef(VectorTy(element_type.dtype, length))
    res = vector_with_item(res, 0, first_element)
    for i in range(1, length):
        element = apply_one(i)
        res = vector_with_item(res, i, element)

    return res


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
    self: Var[VectorTy], index: Var[ScalarTy], value: Var[ScalarTy]
) -> Var[VectorTy]:
    return vector_with_item(self, index, value)


@impl(getattr, overload=(VectorTy, "astype"))
def getattr_vector_astype(object: Var[VectorTy], name: Var):
    return bind_method(object, Vector.astype)


@impl(Vector.astype)
def vector_astype_impl(self: Var[VectorTy], dtype: Var) -> Var[VectorTy]:
    return astype(self, require_dtype_spec(dtype))


@impl(operator.getitem, overload=(VectorTy, WILDCARD))
def vector_getitem(object: Var[VectorTy], key: Var[ScalarTy]) -> Var[ScalarTy]:
    result_dtype = object.get_type().element_dtype
    index = implicit_cast(key, datatype.int32, "vector getitem index")
    return add_operation(
        VectorGetItem,
        ScalarTy(result_dtype),
        x=object,
        index=index,
    )
