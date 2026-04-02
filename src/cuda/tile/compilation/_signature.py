# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, Iterator, TypeAlias, Any

from cuda.tile._execution import kernel
from cuda.tile._cext import CallingConvention, get_parameter_constraints_from_pyargs
from cuda.tile._datatype import DType


@dataclass(frozen=True, init=False)
class ScalarConstraint:
    """
    Describes a scalar kernel parameter and associated compile-time assumptions.

    Args:
        dtype: Data type of the scalar.
    """
    dtype: DType

    def __init__(self, dtype: DType):
        if not isinstance(dtype, DType):
            raise TypeError(f"Expected a DType for the `dtype` parameter, got '{dtype}'")
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True, init=False)
class ArrayConstraint:
    """
    Describes an array kernel parameter and associated compile-time assumptions.

    Args:
        dtype (DType):
            Data type of the array.
        ndim (int):
            Number of dimensions of the array, also known as rank.
        stride_lower_bound_incl (Sequence[int | None] | int | None):
            For each dimension of the array, an optional inclusive lower bound for its stride.
            If all dimensions have the same lower bound, a single number can be passed
            instead of a sequence. For example, passing `0` specifies that all strides
            are non-negative.
        alias_groups (Sequence[str]):
            When set to an empty sequence, specifies that this array may not alias
            any other parameter. Otherwise, it must be a sequence of arbitrary strings,
            referred to as "alias groups". Two parameters are allowed to alias
            each other if and only if they have an alias group in common.
        may_alias_internally (bool):
            Indicates whether two distinct in-bounds indices are allowed
            to point to the same memory location. For example, this can happen if the
            array has a zero stride. For most arrays produced by major tensor libraries,
            this can be assumed to be false. Setting this to True may disable certain
            optimizations of loads and stores to/from this array.
        stride_constant (Sequence[int | None] | None):
            For each dimension of the array, an optional constant value of its stride.
            For example, if the array is known to have a C-contiguous layout, the stride of
            the last dimension can be set to 1, which may enable certain optimizations of loads
            and stores from/to this array. Can be set to `None` if none of the dimensions
            have known strides (this is the default).
        stride_divisible_by (Sequence[int] | int):
            For each dimension of the array, a factor by which its stride is assumed
            to be divisible. The value is given in array elements, not bytes.
            For example, a value of `8` for a `float16` array indicates divisibility by
            16 bytes, since each element of the array is 2 bytes wide. Value of 1
            indicates that no assumption is made regarding the stride divisibility
            (this is the default).
        shape_divisible_by (Sequence[int] | int):
            For each dimension of the array, a factor by which its length is assumed
            to be divisible. The value is given in array elements, not bytes.
            For example, a value of `8` for a `float16` array indicates divisibility by
            16 bytes, since each element of the array is 2 bytes wide. Value of 1
            indicates that no assumption is made regarding the shape divisibility
            (this is the default).
        base_addr_divisible_by (int): Factor by which the array's base address is assumed
            to be divisible. Value of 1 indicates that no assumption is made regarding
            the base address divisibility (this is the default).
    """
    dtype: DType
    ndim: int
    stride_lower_bound_incl: tuple[int | None, ...]
    alias_groups: tuple[str, ...]
    may_alias_internally: bool
    stride_constant: tuple[int | None, ...]
    stride_divisible_by: tuple[int, ...]
    shape_divisible_by: tuple[int, ...]
    base_addr_divisible_by: int

    def __init__(self,
                 dtype: DType,
                 ndim: int,
                 *,
                 stride_lower_bound_incl: Sequence[int | None] | int | None,
                 alias_groups: Sequence[str],
                 may_alias_internally: bool,
                 stride_constant: Sequence[int | None] | None = None,
                 stride_divisible_by: Sequence[int] | int = 1,
                 shape_divisible_by: Sequence[int] | int = 1,
                 base_addr_divisible_by: int = 1):
        # dtype
        if not isinstance(dtype, DType):
            raise TypeError(f"Expected a DType for the `dtype` parameter, got '{dtype}'")

        # ndim
        if not isinstance(ndim, int):
            raise TypeError(f"Expected an integer for `ndim`, got '{ndim}'")
        if ndim < 0:
            raise ValueError("`ndim` cannot be negative")

        # stride_constant
        stride_constant = _parse_assumption_tuple(
                stride_constant, ndim, "stride_constant", None, _check_optional_int)

        # stride_lower_bound
        stride_lower_bound_incl = _parse_assumption_tuple(
            stride_lower_bound_incl, ndim, "stride_lower_bound_incl", None, _check_optional_int)
        stride_lower_bound_incl = _remove_redundant_lower_bounds(
            stride_constant, stride_lower_bound_incl, "stride_lower_bound_incl")

        # stride_divisible_by
        stride_divisible_by = _parse_assumption_tuple(
                stride_divisible_by, ndim, "stride_divisible_by", 1, _check_divisibility)
        stride_divisible_by = _remove_redundant_divisibility_constraints(
                stride_constant, stride_divisible_by, "stride_constant")

        # shape_divisible_by
        shape_divisible_by = _parse_assumption_tuple(
                shape_divisible_by, ndim, "shape_divisible_by", 1, _check_divisibility)

        # base_addr_divisible_by_bytes
        if not isinstance(base_addr_divisible_by, int):
            raise TypeError("Expected an integer for the `base_addr_divisible_by`"
                            f" parameter, got {base_addr_divisible_by}")
        if base_addr_divisible_by <= 0:
            raise ValueError(f"`base_addr_divisible_by` must be strictly positive,"
                             f" got {base_addr_divisible_by}")

        # may_alias_internally
        may_alias_internally = bool(may_alias_internally)

        # alias_groups
        alias_groups = _parse_alias_groups(alias_groups)

        for field in dataclasses.fields(ArrayConstraint):
            object.__setattr__(self, field.name, locals()[field.name])


def _parse_alias_groups(alias_groups: Sequence[str]):
    if isinstance(alias_groups, str):
        raise TypeError("`alias_groups` cannot be a string")
    alias_groups = tuple(alias_groups)
    for i, ag in enumerate(alias_groups):
        if not isinstance(ag, str):
            raise TypeError(f"Element #{i} of `alias_groups` has non-string type {type(ag)}")
    return alias_groups


@dataclass(frozen=True, init=False)
class ListConstraint:
    """
    Describes a list kernel parameter and associated compile-time assumptions.

    Args:
        element (ArrayConstraint):
            Describes the element of this list. Currently, this must be an `ArrayConstraint`,
            since only lists of arrays are supported as kernel arguments.
        alias_groups (Sequence[str]):
            Describes which other parameters the storage of this list is allowed to alias.
            Note that this is different from ``element.alias_groups``, which sets aliasing
            assumptions on the list elements.
            When set to an empty sequence, specifies that this list may not alias
            any other parameter. Otherwise, it must be a sequence of arbitrary strings,
            referred to as "alias groups". Two parameters are allowed to alias
            each other if and only if they have an alias group in common.
        elements_may_alias (bool):
            Specifies whether two distinct elements of this list are allowed to alias each other.
    """
    element: "ParameterConstraint"
    alias_groups: tuple[str, ...]
    elements_may_alias: bool

    def __init__(self,
                 element: ArrayConstraint,
                 *,
                 alias_groups: Sequence[str],
                 elements_may_alias: bool):
        if not isinstance(element, ArrayConstraint):
            raise TypeError(f"ListConstraint only supports an ArrayParameter"
                            f" as the `element` constraint, got '{element}'")
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "alias_groups", _parse_alias_groups(alias_groups))
        object.__setattr__(self, "elements_may_alias", bool(elements_may_alias))


@dataclass(frozen=False, eq=False)
class ConstantConstraint:
    """
    Specifies the constant value of a kernel parameter
    marked with :py:class:`ct.Constant <cuda.tile.Constant>`.

    Args:
        value (bool | int | float):
            The value of the compile-time constant.
    """
    value: bool | int | float

    def __post_init__(self):
        if not isinstance(self.value, bool | int | float):
            raise TypeError(f"Unexpected constant value type {type(self.value)}")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        # Take the type into account, so that 1, True and 1.0 are treated as three distinct things
        if type(self.value) is not type(other.value):
            return False

        # For floats, compare bit representations so that NaN is treated as equal to itself
        if isinstance(self.value, float):
            return struct.pack("=d", self.value) == struct.pack("=d", other.value)

        return self.value == other.value


ParameterConstraint: TypeAlias = (ScalarConstraint | ArrayConstraint | ListConstraint
                                  | ConstantConstraint)


def _to_constraint(c: ParameterConstraint | bool | int | float):
    if isinstance(c, ParameterConstraint):
        return c
    elif isinstance(c, bool | int | float):
        return ConstantConstraint(c)
    else:
        raise TypeError(f"Can't interpret {c!r} as a parameter constraint")


@dataclass(frozen=True, init=False)
class KernelSignature:
    """
    Signature of a compiled kernel.

    Args:
        parameters (Sequence[ParameterConstraint | bool | int | float]):
            For each parameter of the kernel's Python function, a corresponding
            :py:class:`ParameterConstraint` instance. If a parameter is marked with
            :py:class:`ct.Constant <cuda.tile.Constant>`, the corresponding constraint must be
            a :py:class:`ConstantConstraint` or a ``bool``, ``int`` or ``float`` value
            that specifies the compile-time constant.
            Otherwise, it must be either a :py:class:`ScalarConstraint`,
            :py:class:`ArrayConstraint` or :py:class:`ListConstraint`.
            Passing a ``bool``, ``int`` or ``float`` value as a constraint is convenience notation
            that is equivalent to passing an instance of :py:class:`ConstantConstraint`
            that wraps said value.
        calling_convention (CallingConvention):
            |Calling convention| to use.
        symbol (str | None):
            Symbol name to use for the exported kernel. Set to `None` to automatically
            generate it from the Python function name and this signature, using a name mangling
            algorithm defined by the selected calling convention.
    """

    parameters: tuple[ParameterConstraint, ...]
    calling_convention: CallingConvention
    symbol: str | None

    def __init__(self,
                 parameters: Sequence[ParameterConstraint | bool | int | float],
                 calling_convention: CallingConvention,
                 symbol: str | None = None):
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"`symbol` must be a string, got {type(symbol)}")

        if not isinstance(calling_convention, CallingConvention):
            raise TypeError(f"Unsupported calling convention {calling_convention}")

        parameters = tuple(_to_constraint(c) for c in parameters)
        _validate_alias_groups(parameters)

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "calling_convention", calling_convention)
        object.__setattr__(self, "symbol", symbol)

    def with_mangled_symbol(self, function_name: str) -> "KernelSignature":
        """
        Returns a copy of `self` with the `symbol` attribute replaced with a mangled name.

        Args:
            function_name(str):
                Function name to use as the base of the mangled symbol.
        Returns:
            KernelSignature
        """
        from cuda.tile.compilation._name_mangling import mangle_kernel_name
        symbol = mangle_kernel_name(function_name, self)
        return self.with_symbol(symbol)

    def with_symbol(self, symbol: str | None) -> "KernelSignature":
        """
        Returns a copy of `self` with the `symbol` attribute replaced with the given value.

        Args:
            symbol(str | None): The new symbol name.
        Returns:
            KernelSignature
        """
        return dataclasses.replace(self, symbol=symbol)

    @staticmethod
    def from_kernel_args(kernel: kernel,
                         kernel_args: Sequence[Any],
                         calling_convention: CallingConvention,
                         *,
                         symbol: str | None = None) -> "KernelSignature":
        """
        Returns the signature that would be used if the kernel was compiled just-in-time
        for the given arguments.o

        .. warning::

            It is recommended to limit the use of this function to testing or prototyping.
            Deriving a kernel signature from example arguments may create unexpected assumptions
            on kernel parameters.

            For example, if the base address of an example array argument happens to be
            divisible by 16, an assumption may be made that it will always be so.
            Launching the exported kernel with an array that doesn't satisfy this assumption
            would result in undefined behavior.

        Args:
            kernel (cuda.tile.kernel):
                A kernel function decorated with :py:class:`@ct.kernel <cuda.tile.kernel>`.
            kernel_args (Sequence[Any]):
                Tuple of kernel arguments, as if it were be passed to
                :py:func:`ct.launch() <cuda.tile.launch>`.
            calling_convention (CallingConvention):
                |Calling convention| to use.
            symbol (str | None):
                Specifies the `symbol` attribute of the returned signature.
                If set to `None`, the returned symbol will be automatically filled
                using a name mangling algorithm defined by the selected |calling convention|.
        Returns:
            KernelSignature
        """

        constraints = get_parameter_constraints_from_pyargs(kernel, tuple(kernel_args),
                                                            calling_convention)
        sig = KernelSignature(constraints, calling_convention, symbol)
        if symbol is None:
            sig = sig.with_mangled_symbol(kernel._annotated_function.pyfunc.__name__)
        return sig


def _validate_alias_groups(parameters: Sequence[ParameterConstraint]):
    use_count = defaultdict(int)
    constraint_type_by_group = dict()
    for constraint, groups in _collect_alias_groups(parameters):
        seen = set()
        for ag in groups:
            if ag in seen:
                raise ValueError(f"Alias group `{ag}` occurs more than once"
                                 f" in the group list {list(groups)}")
            seen.add(ag)
            use_count[ag] += 1

            ty = type(constraint)
            existing_ty = constraint_type_by_group.get(ag)
            if existing_ty is not None and ty is not existing_ty:
                raise ValueError(f"Alias group `{ag}` is used in two constraints of different"
                                 f" types {existing_ty.__name__} and {ty.__name__}."
                                 f" This is currently unsupported"
                                 f" (e.g., list storage is not allowed to alias an array).")
            constraint_type_by_group[ag] = ty

    for ag, count in use_count.items():
        if count == 1:
            raise ValueError(f"Alias group `{ag}` is mentioned only once, and is therefore"
                             f" redundant. To specify that an array or a list may not alias"
                             f" any other object, use an empty tuple of alias groups.")


def _collect_alias_groups(parameters: Sequence[ParameterConstraint]
                          ) -> Iterator[tuple[ParameterConstraint, Sequence[str]]]:
    for p in parameters:
        if isinstance(p, ArrayConstraint):
            yield p, p.alias_groups
        elif isinstance(p, ListConstraint):
            yield p, p.alias_groups
            yield from _collect_alias_groups([p.element])


def _check_optional_int(i: int, val, param_name: str):
    if val is not None and not isinstance(val, int):
        raise TypeError(f"Element #{i} of `{param_name}` must be None or int, got {val}")


def _check_divisibility(i: int, val, param_name: str):
    if not isinstance(val, int):
        raise TypeError(f"Element #{i} of `{param_name}` must be int, got {val}")
    if val <= 0:
        raise TypeError(f"Element #{i} of `{param_name}` must be strictly positive, got {val}")


def _parse_assumption_tuple(seq: Sequence[int] | int | None, ndim: int, param_name: str,
                            default_value, predicate):
    if seq is None:
        return (default_value,) * ndim
    elif isinstance(seq, int):
        return (seq,) * ndim
    else:
        seq = tuple(seq)
        if len(seq) != ndim:
            raise ValueError(f"Length of `{param_name}` {seq} doesn't match ndim={ndim}")
        for i, x in enumerate(seq):
            predicate(i, x, param_name)

        return seq


def _remove_redundant_lower_bounds(static_values: tuple[int, ...],
                                   lower_bounds: tuple[int, ...],
                                   param_name: str) -> tuple[int, ...]:
    ret = list(lower_bounds)
    for i, val in enumerate(static_values):
        if val is not None and ret[i] is not None:
            if val < ret[i]:
                raise ValueError(f"{param_name}[{i}] is set to {val},"
                                 f" which is less than the lower bound {ret[i]}")
            ret[i] = None
    return tuple(ret)


def _remove_redundant_divisibility_constraints(static_values: tuple[int, ...],
                                               div_by: tuple[int, ...],
                                               param_name: str) -> tuple[int, ...]:
    ret = list(div_by)
    for i, val in enumerate(static_values):
        if val is not None:
            if val % ret[i] != 0:
                raise ValueError(f"{param_name}[{i}] is set to {val},"
                                 f" which is not divisible by {ret[i]}")
            ret[i] = 1
    return tuple(ret)
