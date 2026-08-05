# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import re
import struct
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Protocol, Iterable, Any, TypeVar

from ._signature import ArrayConstraint, ParameterConstraint, ListConstraint, TupleConstraint, \
    ScalarConstraint, KernelSignature, _collect_alias_groups, ConstantConstraint, \
    DataclassConstraint
from cuda.tile._datatype import DType, bool_, uint8, uint16, uint32, uint64, int64, int32, int16, \
    int8, float16, float32, float64, bfloat16, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, \
    tfloat32
from .._cext import CallingConvention, classify_constant, ConstantKind, cconv_v3_enabled


def mangle_kernel_name(function_name: str,
                       kernel_signature: KernelSignature) -> str:
    alias_group_map, alias_group_names = _map_alias_groups(kernel_signature.parameters)
    cconv = kernel_signature.calling_convention
    collected_globals = _CollectedGlobals(dataclasses=set(), enums=set())
    ret = (function_name + f"_K{cconv.code}"
           + "".join("_" + _mangle_constraint(p, alias_group_map, collected_globals)
                     for p in kernel_signature.parameters))
    parsed_function_name, parsed_sig = _demangle_kernel_name(
            ret, alias_group_names,
            allowed_dataclasses=collected_globals.dataclasses,
            allowed_enums=collected_globals.enums)
    assert function_name == parsed_function_name
    assert kernel_signature.parameters == parsed_sig.parameters, \
        f"Failed to round-trip mangled name {ret}"
    return ret


T = TypeVar("T")


class GlobalProvider(Protocol[T]):
    def __call__(self, module_name: str, qualname: str) -> T:
        ...


def demangle_kernel_name(symbol: str) -> tuple[str, KernelSignature]:
    return _demangle_kernel_name(symbol, None, allowed_dataclasses=[], allowed_enums=[])


@dataclass(frozen=True)
class _CollectedGlobals:
    dataclasses: set[type]
    enums: set[type[Enum]]


@dataclass(frozen=True)
class _AllowedGlobals:
    dataclasses: GlobalProvider[type]
    enums: GlobalProvider[type[Enum]]


def _to_global_provider(iterable_or_provider: Iterable[T] | GlobalProvider[T],
                        list_name: str) -> GlobalProvider[T]:
    if callable(iterable_or_provider):
        return iterable_or_provider

    allowed = tuple(iterable_or_provider)

    def provider(module_name: str, class_qualname: str) -> type:
        for c in allowed:
            if c.__module__ == module_name and c.__qualname__ == class_qualname:
                return c
        raise ValueError(f"Global '{module_name}.{class_qualname}' not found in"
                         f" the '{list_name}' list; refusing to demangle the name.")

    return provider


def _demangle_kernel_name(symbol: str,
                          alias_group_names: Sequence[str] | None,
                          allowed_dataclasses: Iterable[type] | GlobalProvider[type],
                          allowed_enums: Iterable[type[Enum]] | GlobalProvider[type[Enum]],
                          ) -> tuple[str, KernelSignature]:
    allowed_globals = _AllowedGlobals(
        dataclasses=_to_global_provider(allowed_dataclasses, "allowed_dataclasses"),
        enums=_to_global_provider(allowed_enums, "allowed_enums")
    )

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
        constraint = _demangle_constraint(cursor, alias_group_demangler, allowed_globals)
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


def _mangle_constraint(p: ParameterConstraint, alias_group_map: dict[str, int],
                       collected_globals: _CollectedGlobals) -> str:
    if isinstance(p, ArrayConstraint):
        return "A" + _mangle_array_constraint(p, alias_group_map)
    elif isinstance(p, ListConstraint):
        assert isinstance(p.element, ArrayConstraint)
        return "L" + _mangle_list_constraint(p, alias_group_map, collected_globals)
    elif isinstance(p, TupleConstraint):
        return "T" + _mangle_tuple_constraint(p, alias_group_map, collected_globals)
    elif isinstance(p, DataclassConstraint):
        return "D" + _mangle_dataclass_constraint(p, alias_group_map, collected_globals)
    elif isinstance(p, ScalarConstraint):
        return "S" + _mangle_dtype(p.dtype)
    elif isinstance(p, ConstantConstraint):
        kind = classify_constant(p.value, True)
        assert kind is not None  # validated in ConstantConstraint.__post_init__()
        match kind:
            case ConstantKind.Bool:
                ret = "B" + str(int(p.value))
            case ConstantKind.Int:
                ret = "I" + _mangle_signed_int(p.value)
            case ConstantKind.Float:
                [i] = struct.unpack("<Q", struct.pack("<d", p.value))
                ret = "F" + f"{i:016x}"
            case ConstantKind.Enum:
                assert cconv_v3_enabled()
                ret = "Ce_" + _mangle_enum_constant(p.value, collected_globals)
            case _: assert False
        assert ret.startswith(kind._value_)
        return ret
    else:
        raise TypeError(f"Unexpected constraint type: {type(p)}")


def _demangle_constraint(cursor: _Cursor,
                         alias_group_demangler: _AliasGroupDemangler,
                         allowed_globals: _AllowedGlobals) -> ParameterConstraint:
    orig_cursor = cursor.clone()
    c = cursor.expect("[A-BD-Z]|C[a-z0-9]+_",
                      "Expected a constraint starting with a capital letter")
    if c == "A":
        return _demangle_array_constraint(cursor, alias_group_demangler)
    elif c == "L":
        return _demangle_list_constraint(cursor, alias_group_demangler, allowed_globals)
    elif c == "T":
        return _demangle_tuple_constraint(cursor, alias_group_demangler, allowed_globals)
    elif c == "D" and cconv_v3_enabled():
        return _demangle_dataclass_constraint(cursor, alias_group_demangler, allowed_globals)
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
    elif c == "Ce_" and cconv_v3_enabled():
        return ConstantConstraint(_demangle_enum_constant(cursor, allowed_globals))
    else:
        raise orig_cursor.make_error(f"Unexpected constraint code '{c}'")


def _mangle_array_constraint(a: ArrayConstraint,
                             alias_group_map: dict[str, int]) -> str:
    ret = f"{a.ndim}{_mangle_dtype(a.dtype)}"

    # NOTE: since we encode axis masks as hex, letters a-f can't be used for predicates

    axis_predicates = OrderedDict()
    _collect_axis_predicate(a.shape_constant, "s", None, axis_predicates)
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

    if a.index_dtype == uint32:
        extras += "u"
    elif a.index_dtype == int64:
        extras += "w"
    else:
        assert a.index_dtype == int32

    if len(extras) > 0:
        ret += "_" + extras

    return ret


def _demangle_array_constraint(cursor: _Cursor,
                               alias_group_demangler: _AliasGroupDemangler) -> ArrayConstraint:
    orig_cursor = cursor.clone()
    ndim = int(cursor.expect("[0-9]+", "Expected ndim integer"))
    dtype = _demangle_dtype(cursor)

    # Read axis predicates
    shape_constant = [None] * ndim
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

        axis_shape_constant = None
        if cursor.read("s") is not None:
            axis_shape_constant = _demangle_signed_int(cursor)

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

            if axis_shape_constant is not None:
                if shape_constant[i] is not None:
                    raise mask_cursor.make_error(
                        f"Static shape specified more than once for axis #{i}")
                shape_constant[i] = axis_shape_constant

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
    index_dtype = int32
    if cursor.peek("_[a-z]") is not None:
        cursor.expect("_", "Expected an underscore")
        if cursor.read("p"):
            base_addr_div_by = _demangle_divisibility(cursor)

        alias_groups = alias_group_demangler.demangle_group_ids(cursor)

        if cursor.read("i"):
            may_alias_internally = True

        if cursor.read("u"):
            index_dtype = uint32
        elif cursor.read("w"):
            index_dtype = int64

    return ArrayConstraint(dtype,
                           ndim,
                           index_dtype=index_dtype,
                           stride_lower_bound_incl=stride_lower_bound_incl,
                           alias_groups=alias_groups,
                           may_alias_internally=may_alias_internally,
                           stride_constant=stride_constant,
                           shape_constant=shape_constant,
                           stride_divisible_by=stride_divisible_by,
                           shape_divisible_by=shape_divisible_by,
                           base_addr_divisible_by=base_addr_div_by)


def _mangle_list_constraint(constraint: ListConstraint, alias_group_map: dict[str, int],
                            collected_globals: _CollectedGlobals) -> str:
    ret = ""
    for group_id in sorted((alias_group_map[ag] for ag in constraint.alias_groups)):
        ret += f"g{group_id:x}"
    if constraint.elements_may_alias:
        ret += "i"
    return ret + _mangle_constraint(constraint.element, alias_group_map, collected_globals)


def _demangle_list_constraint(cursor: _Cursor,
                              alias_group_demangler: _AliasGroupDemangler,
                              allowed_globals: _AllowedGlobals) -> ListConstraint:
    alias_groups = alias_group_demangler.demangle_group_ids(cursor)
    elements_may_alias = cursor.read("i") is not None
    old_cursor = cursor.clone()
    element = _demangle_constraint(cursor, alias_group_demangler, allowed_globals)
    if not isinstance(element, ArrayConstraint):
        raise old_cursor.make_error("Expected an ArrayConstraint")
    return ListConstraint(element, alias_groups=alias_groups, elements_may_alias=elements_may_alias)


def _mangle_tuple_constraint(constraint: TupleConstraint, alias_group_map: dict[str, int],
                             collected_globals: _CollectedGlobals) -> str:
    # Format: {count}{item0_mangling}{item1_mangling}...
    return f"{len(constraint.items)}" + "".join(
        _mangle_constraint(e, alias_group_map, collected_globals) for e in constraint.items)


def _demangle_tuple_constraint(cursor: _Cursor,
                               alias_group_demangler: _AliasGroupDemangler,
                               allowed_globals: _AllowedGlobals) -> TupleConstraint:
    count = int(cursor.expect("[0-9]+", "Expected element count"))
    items = [_demangle_constraint(cursor, alias_group_demangler, allowed_globals)
             for _ in range(count)]
    return TupleConstraint(items)


def _mangle_dataclass_constraint(constraint: DataclassConstraint,
                                 alias_group_map: dict[str, int],
                                 collected_globals: _CollectedGlobals) -> str:
    # Format: {module_name}{qualname}{field count}{field0_mangling}{field1_mangling}...
    return (
        _mangle_global(constraint.cls, collected_globals.dataclasses)
        + str(len(constraint.fields))
        + "".join(_mangle_constraint(f, alias_group_map, collected_globals)
                  for f in constraint.fields)
    )


def _demangle_dataclass_constraint(cursor: _Cursor,
                                   alias_group_demangler: _AliasGroupDemangler,
                                   allowed_globals: _AllowedGlobals) -> DataclassConstraint:
    cls = _demangle_global(cursor, allowed_globals.dataclasses)
    field_count = int(cursor.expect("[1-9][0-9]*", "Expected field count"))
    items = [_demangle_constraint(cursor, alias_group_demangler, allowed_globals)
             for _ in range(field_count)]
    return DataclassConstraint(cls, items)


def _mangle_enum_constant(value, collected_globals: _CollectedGlobals):
    assert isinstance(value, Enum)
    enum_cls = type(value)
    return _mangle_global(enum_cls, collected_globals.enums) + _mangle_string(value._name_)


def _demangle_enum_constant(cursor: _Cursor,
                            allowed_globals: _AllowedGlobals) -> Any:
    enum_cls = _demangle_global(cursor, allowed_globals.enums)
    member_name = _demangle_string(cursor)
    return enum_cls.__members__[member_name]


def _mangle_global(obj: T, collected: set[T]) -> str:
    collected.add(obj)
    return _mangle_string(obj.__module__) + _mangle_string(obj.__qualname__)


def _demangle_global(cursor: _Cursor, provider: GlobalProvider[T]) -> T:
    module_name = _demangle_string(cursor)
    qualname = _demangle_string(cursor)
    return provider(module_name, qualname)


def _mangle_string(s: str) -> str:
    chunks = []
    start = 0
    # Escape all characters that are not alphanumeric ASCII. "K" is escaped as well, so that a
    # mangled string can never produce the "_K" sequence that separates the function name from
    # the signature.
    for m in re.finditer("[^a-zA-JL-Z0-9]|$", s):
        match_start, match_end = m.span()
        if match_start > start:
            chunks.append(s[start:match_start])
        start = match_end
        match m.group(0):
            case "": break
            case "_": chunks.append("__")
            case "K": chunks.append("_k")
            case c if ord(c) <= 0xff: chunks.append(f"_{ord(c):02x}")
            case c if ord(c) <= 0xff_ff: chunks.append(f"_u{ord(c):04x}")
            case c: chunks.append(f"_w{ord(c):08x}")
    assert start == len(s)
    chunks.append("_z")  # end marker
    return "".join(chunks)


def _demangle_string(cursor: _Cursor) -> str:
    chunks = []
    while True:
        # Read any unescaped ASCII chars in bulk ("K" is always escaped, see _mangle_string)
        alphanumeric_chunk = cursor.read("[a-zA-JL-Z0-9]+")
        if alphanumeric_chunk is not None:
            chunks.append(alphanumeric_chunk)

        cursor.expect("_", "Expected an underscore")
        match cursor.expect("[_0-9a-fkzuw]", "Expected a valid character escape"):
            case "_": chunks.append("_")
            case "k": chunks.append("K")
            case "z": break
            case "u":
                digits = cursor.expect("[0-9a-f]{4}", "Expected 4 hex digits after '_u' escape")
                chunks.append(chr(int(digits, base=16)))
            case "w":
                digits = cursor.expect("[0-9a-f]{8}", "Expected 8 hex digits after '_u' escape")
                chunks.append(chr(int(digits, base=16)))
            case c:
                digits = c + cursor.expect("[0-9a-f]", "Expected 2 hex digits after '_' escape")
                chunks.append(chr(int(digits, base=16)))
    return "".join(chunks)


def _mangle_dtype(dtype: DType):
    try:
        return _mangled_dtype[dtype]
    except KeyError:
        raise ValueError(f"Unexpected dtype {dtype}")


def _demangle_dtype(cursor: _Cursor) -> DType:
    old_cursor = cursor.clone()
    dtype_str = cursor.expect("[a-z0-9]+", "Expected dtype name")
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
