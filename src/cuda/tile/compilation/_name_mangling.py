# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import struct
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Sequence

from ._signature import ArrayConstraint, ParameterConstraint, ListConstraint, ScalarConstraint, \
    KernelSignature, _collect_alias_groups, ConstantConstraint
from cuda.tile._datatype import DType, bool_, uint8, uint16, uint32, uint64, int64, int32, int16, \
    int8, float16, float32, float64, bfloat16, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, tfloat32
from .._cext import CallingConvention


def mangle_kernel_name(function_name: str,
                       kernel_signature: KernelSignature) -> str:
    alias_group_map, alias_group_names = _map_alias_groups(kernel_signature.parameters)
    ret = (function_name + f"_K{kernel_signature.calling_convention.code}"
           + "".join("_" + _mangle_constraint(p, alias_group_map)
                     for p in kernel_signature.parameters))
    parsed_function_name, parsed_sig = _demangle_kernel_name(ret, alias_group_names)
    assert function_name == parsed_function_name
    assert kernel_signature.parameters == parsed_sig.parameters, \
        f"Failed to round-trip mangled name {ret}"
    return ret


def demangle_kernel_name(symbol: str) -> tuple[str, KernelSignature]:
    return _demangle_kernel_name(symbol, None)


def _demangle_kernel_name(symbol: str,
                          alias_group_names: Sequence[str] | None) -> tuple[str, KernelSignature]:
    pos = symbol.rfind("_K")
    if pos < 0:
        raise ValueError(f"`{symbol}` is not a mangled kernel name")
    function_name = symbol[:pos]
    cursor = _Cursor(symbol, symbol[pos + 2:], pos + 2)

    cconv = _demangle_calling_convention(cursor)

    alias_group_demangler = _AliasGroupDemangler(alias_group_names)
    parameters = []
    while len(cursor.remaining) > 0:
        cursor.expect("_", "Expected an underscore")
        constraint = _demangle_constraint(cursor, alias_group_demangler)
        parameters.append(constraint)
    sig = KernelSignature(parameters, cconv, symbol)
    return function_name, sig


@dataclass
class _Cursor:
    original: str
    remaining: str
    pos: int = 0

    def clone(self) -> "_Cursor":
        return _Cursor(self.original, self.remaining, self.pos)

    def make_error(self, msg: str) -> ValueError:
        context = self.remaining[:20]
        return ValueError(f"Invalid mangled name '{self.original}'."
                          f" At offset #{self.pos}, near '{context}': {msg}")

    def peek(self, regex) -> re.Match | None:
        return re.match(regex, self.remaining)

    def read(self, regex) -> str | None:
        m = self.peek(regex)
        if m is None:
            return None
        g = m.group(0)
        n = len(g)
        ret = self.remaining[:n]
        assert ret == g
        self.remaining = self.remaining[n:]
        self.pos += n
        return ret

    def expect(self, regex, msg: str) -> str:
        ret = self.read(regex)
        if ret is None:
            raise self.make_error(msg)
        return ret


class _AliasGroupDemangler:
    def __init__(self, alias_group_names: Sequence[str] | None):
        self._group_names = alias_group_names
        self._last_seen_id = -1

    def demangle_group_ids(self, cursor: _Cursor) -> list[str]:
        prev_group_id = None
        ret = []
        while cursor.read("g") is not None:
            old_cursor = cursor.clone()
            group_id_str = cursor.expect("[0-9a-f]+", "Expected a hex alias group ID")
            if len(group_id_str) > 1 and group_id_str[0] == "0":
                raise old_cursor.make_error("Leading zero in alias group ID")
            group_id = int(group_id_str, base=16)
            if group_id > self._last_seen_id + 1:
                raise old_cursor.make_error("Invalid alias group ID")
            if prev_group_id is not None and group_id <= prev_group_id:
                raise old_cursor.make_error("Alias group IDs are not strictly increasing")

            self._last_seen_id = prev_group_id = group_id

            if self._group_names is None:
                group_name = f"group{group_id}"
            else:
                group_name = self._group_names[group_id]

            ret.append(group_name)
        return ret


def _map_alias_groups(parameters: Sequence[ParameterConstraint]
                      ) -> tuple[dict[str, int], list[str]]:
    name2idx, idx2name = dict(), list()
    for _, groups in _collect_alias_groups(parameters):
        for ag in groups:
            if ag not in name2idx:
                name2idx[ag] = len(name2idx)
                idx2name.append(ag)
    return name2idx, idx2name


def _demangle_calling_convention(cursor: _Cursor) -> CallingConvention:
    cconv_code = cursor.expect("[^_]+", "Expected a calling convention code after _K")
    return CallingConvention.from_code(cconv_code)


def _mangle_constraint(p: ParameterConstraint, alias_group_map: dict[str, int]) -> str:
    if isinstance(p, ArrayConstraint):
        return "A" + _mangle_array_constraint(p, alias_group_map)
    elif isinstance(p, ListConstraint):
        assert isinstance(p.element, ArrayConstraint)
        return "L" + _mangle_list_constraint(p, alias_group_map)
    elif isinstance(p, ScalarConstraint):
        return "S" + _mangle_dtype(p.dtype)
    elif isinstance(p, ConstantConstraint):
        if isinstance(p.value, bool):
            return "B" + str(int(p.value))
        elif isinstance(p.value, int):
            return "I" + _mangle_signed_int(p.value)
        elif isinstance(p.value, float):
            [i] = struct.unpack("<Q", struct.pack("<d", p.value))
            return f"F{i:016x}"
        else:
            raise TypeError(f"Unexpected constant value type: {type(p.value)}")
    else:
        raise TypeError(f"Unexpected constraint type: {type(p)}")


def _demangle_constraint(cursor: _Cursor,
                         alias_group_demangler: _AliasGroupDemangler,
                         ) -> ParameterConstraint:
    orig_cursor = cursor.clone()
    c = cursor.expect("[A-Z]", "Expected a constraint starting with a capital letter")
    if c == "A":
        return _demangle_array_constraint(cursor, alias_group_demangler)
    elif c == "L":
        return _demangle_list_constraint(cursor, alias_group_demangler)
    elif c == "S":
        dtype = _demangle_dtype(cursor)
        return ScalarConstraint(dtype)
    elif c == "B":
        return ConstantConstraint(bool(int(cursor.expect("[01]", "Expected 0 or 1"))))
    elif c == "I":
        return ConstantConstraint(_demangle_signed_int(cursor))
    elif c == "F":
        i = int(cursor.expect("[0-9a-f]{16}", "Expected 16 hex digits"), base=16)
        [f] = struct.unpack("<d", struct.pack("<Q", i))
        return ConstantConstraint(f)
    else:
        raise orig_cursor.make_error(f"Unexpected constraint code '{c}'")


def _mangle_array_constraint(a: ArrayConstraint,
                             alias_group_map: dict[str, int]) -> str:
    ret = f"{a.ndim}{_mangle_dtype(a.dtype)}"

    # NOTE: since we encode axis masks as hex, letters a-f can't be used for predicates

    axis_predicates = OrderedDict()
    _collect_axis_predicate(a.shape_divisible_by, "i", 1, axis_predicates)
    _collect_axis_predicate(a.stride_constant, "t", None, axis_predicates)
    _collect_axis_predicate(a.stride_divisible_by, "v", 1, axis_predicates)
    _collect_axis_predicate(a.stride_lower_bound_incl, "l", None, axis_predicates)

    by_mask = defaultdict(str)
    for pred, axis_mask in axis_predicates.items():
        by_mask[axis_mask] += pred

    for mask in sorted(by_mask.keys()):
        ret += f"_{mask:x}{by_mask[mask]}"

    extras = ""
    if a.base_addr_divisible_by != 1:
        extras += f"p{_mangle_signed_int(a.base_addr_divisible_by)}"
    for group_id in sorted((alias_group_map[ag] for ag in a.alias_groups)):
        extras += f"g{group_id:x}"
    if a.may_alias_internally:
        extras += "i"

    if len(extras) > 0:
        ret += "_" + extras

    return ret


def _demangle_array_constraint(cursor: _Cursor,
                               alias_group_demangler: _AliasGroupDemangler) -> ArrayConstraint:
    orig_cursor = cursor.clone()
    ndim = int(cursor.expect("[0-9]+", "Expected ndim integer"))
    dtype = _demangle_dtype(cursor)

    # Read axis predicates
    shape_divisible_by = [1] * ndim
    stride_constant = [None] * ndim
    stride_divisible_by = [1] * ndim
    stride_lower_bound_incl = [None] * ndim
    while True:
        mask_cursor = cursor.clone()
        axis_mask = cursor.read("_[0-9a-f]+")
        if axis_mask is None:
            break

        axis_mask = int(axis_mask.removeprefix("_"), base=16)
        if axis_mask == 0:
            raise mask_cursor.make_error("Zero axis mask")
        if axis_mask.bit_length() > ndim:
            raise mask_cursor.make_error(f"Axis mask {axis_mask:x} has more bits"
                                         f" ({axis_mask.bit_length()}) than array ndim ({ndim})")

        axis_shape_div_by = 1
        if cursor.read("i") is not None:
            axis_shape_div_by = _demangle_divisibility(cursor)

        axis_stride_constant = None
        if cursor.read("t") is not None:
            axis_stride_constant = _demangle_signed_int(cursor)

        axis_stride_div_by = 1
        if cursor.read("v") is not None:
            axis_stride_div_by = _demangle_divisibility(cursor)

        axis_stride_lb = None
        if cursor.read("l") is not None:
            axis_stride_lb = _demangle_signed_int(cursor)

        while axis_mask > 0:
            i = axis_mask.bit_length() - 1

            if axis_shape_div_by != 1:
                if shape_divisible_by[i] != 1:
                    raise mask_cursor.make_error(
                        f"Shape divisibility specified more than once for axis #{i}")
                shape_divisible_by[i] = axis_shape_div_by

            if axis_stride_constant is not None:
                if stride_constant[i] is not None:
                    raise mask_cursor.make_error(
                        f"Static stride specified more than once for axis #{i}")
                stride_constant[i] = axis_stride_constant

            if axis_stride_div_by != 1:
                if stride_divisible_by[i] != 1:
                    raise mask_cursor.make_error(
                        f"Stride divisibility specified more than once for axis #{i}")
                stride_divisible_by[i] = axis_stride_div_by

            if axis_stride_lb is not None:
                if stride_lower_bound_incl[i] is not None:
                    raise mask_cursor.make_error(
                        f"Stride lower bound specified more than once for axis #{i}")
                stride_lower_bound_incl[i] = axis_stride_lb

            axis_mask &= ~(1 << i)

    for i in range(ndim):
        if stride_constant[i] is not None:
            if stride_divisible_by[i] != 1:
                raise orig_cursor.make_error(f"Stride divisibility specified together"
                                             f" with static stride for axis {i}")
            if stride_lower_bound_incl[i] is not None:
                raise orig_cursor.make_error(f"Stride lower bound specified together"
                                             f" with static stride for axis {i}")

    base_addr_div_by = 1
    alias_groups = []
    may_alias_internally = False
    if cursor.peek("_[a-z]") is not None:
        cursor.expect("_", "Expected an underscore")
        if cursor.read("p"):
            base_addr_div_by = _demangle_divisibility(cursor)

        alias_groups = alias_group_demangler.demangle_group_ids(cursor)

        if cursor.read("i"):
            may_alias_internally = True

    return ArrayConstraint(dtype,
                           ndim,
                           stride_lower_bound_incl=stride_lower_bound_incl,
                           alias_groups=alias_groups,
                           may_alias_internally=may_alias_internally,
                           stride_constant=stride_constant,
                           stride_divisible_by=stride_divisible_by,
                           shape_divisible_by=shape_divisible_by,
                           base_addr_divisible_by=base_addr_div_by)


def _mangle_list_constraint(constraint: ListConstraint, alias_group_map: dict[str, int]):
    ret = ""
    for group_id in sorted((alias_group_map[ag] for ag in constraint.alias_groups)):
        ret += f"g{group_id:x}"
    if constraint.elements_may_alias:
        ret += "i"
    return ret + _mangle_constraint(constraint.element, alias_group_map)


def _demangle_list_constraint(cursor: _Cursor,
                              alias_group_demangler: _AliasGroupDemangler) -> ListConstraint:
    alias_groups = alias_group_demangler.demangle_group_ids(cursor)
    elements_may_alias = cursor.read("i") is not None
    old_cursor = cursor.clone()
    element = _demangle_constraint(cursor, alias_group_demangler)
    if not isinstance(element, ArrayConstraint):
        raise old_cursor.make_error("Expected an ArrayConstraint")
    return ListConstraint(element, alias_groups=alias_groups, elements_may_alias=elements_may_alias)


def _mangle_dtype(dtype: DType):
    try:
        return _mangled_dtype[dtype]
    except KeyError:
        raise ValueError(f"Unexpected dtype {dtype}")


def _demangle_dtype(cursor: _Cursor) -> DType:
    old_cursor = cursor.clone()
    dtype_str = cursor.expect("[^_]+", "Expected dtype name")
    for d, n in _mangled_dtype.items():
        if n == dtype_str:
            return d
    raise old_cursor.make_error(f"Unknown dtype name `{dtype_str}`")


_mangled_dtype = {
    bool_: "b8",
    uint8: "u8",
    uint16: "u16",
    uint32: "u32",
    uint64: "u64",
    int8: "i8",
    int16: "i16",
    int32: "i32",
    int64: "i64",
    float16: "f16",
    float32: "f32",
    float64: "f64",
    bfloat16: "bf16",
    tfloat32: "tf32",
    float8_e4m3fn: "f8m3fn",
    float8_e5m2: "f8m2",
    float8_e8m0fnu: "f8m0fnu",
}


def _collect_axis_predicate(values: Sequence[int | None],
                            letter: str,
                            default: int | None,
                            axis_predicates: OrderedDict[str, int]):
    for i, v in enumerate(values):
        if v != default:
            pred = f"{letter}{_mangle_signed_int(v)}"
            old = axis_predicates.get(pred, 0)
            axis_predicates[pred] = old | (1 << i)


def _mangle_signed_int(val: int) -> str:
    return f"_{-val}" if val < 0 else str(val)


def _demangle_divisibility(cursor: _Cursor) -> int:
    old_cursor = cursor.clone()
    ret = _demangle_signed_int(cursor)
    if ret <= 1:
        raise old_cursor.make_error("Divisibility must be greater than 1")
    return ret


def _demangle_signed_int(cursor: _Cursor) -> int:
    sign = 1 if cursor.read("_") is None else -1
    return sign * int(cursor.expect("[0-9]+", "Expected a decimal integer"))
