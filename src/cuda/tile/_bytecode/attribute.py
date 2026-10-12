# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Optional, Sequence, Mapping

from .basic import encode_varint, StringTable, encode_signed_varint
from .type import TypeId, encode_typeid, I32_TYPE_ID, SimpleType, TypeTable, float_bit_size


class AttributeTag(enum.Enum):
    Integer = b"\x01"
    Float = b"\x02"
    Bool = b"\x03"
    Type = b"\x04"
    String = b"\x05"
    Array = b"\x06"
    DenseElements = b"\x07"
    DivBy = b"\x08"
    SameElements = b"\x09"
    Dictionary = b"\x0a"
    OptimizationHints = b"\x0b"
    Bounded = b"\x0c"


class TaggedAttribute:
    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        raise NotImplementedError(f"encode_tagged() must be implemented for {type(self)}")


@dataclass
class AttributeArray(TaggedAttribute):
    items: Sequence[TaggedAttribute]

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Array._value_)
        encode_varint(len(self.items), buf)
        for item in self.items:
            item.encode_tagged(string_table, buf)


@dataclass
class Dictionary(TaggedAttribute):
    items: Sequence[tuple[str, TaggedAttribute]]

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Dictionary._value_)
        encode_dictionary_untagged(self.items, string_table, buf)


@dataclass
class OptimizationHints(TaggedAttribute):
    hints_by_arch: Sequence[tuple[str, Dictionary]]

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.OptimizationHints._value_)
        encode_dictionary_untagged(self.hints_by_arch, string_table, buf)


def encode_dictionary_untagged(items: Sequence[tuple[str, TaggedAttribute]],
                               string_table: StringTable,
                               buf: bytearray):
    encode_varint(len(items), buf)
    for k, v in items:
        encode_varint(string_table[k.encode()].string_id, buf)
        v.encode_tagged(string_table, buf)


@dataclass
class EntryHints:
    num_cta_in_cga: Optional[int] = None
    occupancy: Optional[int] = None
    num_worker_warps_per_cta: Optional[int] = None

    def as_dictionary(self) -> Dictionary:
        items = []
        if self.num_cta_in_cga is not None:
            items.append(("num_cta_in_cga", Integer.create_i32(self.num_cta_in_cga)))
        if self.occupancy is not None:
            items.append(("occupancy", Integer.create_i32(self.occupancy)))
        if self.num_worker_warps_per_cta is not None:
            items.append(("num_worker_warps_per_cta",
                         Integer.create_i32(self.num_worker_warps_per_cta)))
        return Dictionary(items)


def make_entry_hints(hints_by_arch: Mapping[str, EntryHints]) -> OptimizationHints:
    return OptimizationHints([(arch, hints.as_dictionary())
                              for arch, hints in hints_by_arch.items()])


@dataclass
class LoadStoreHints:
    latency: Optional[int] = None
    allow_tma: bool = True

    def as_dictionary(self) -> Dictionary:
        items = []
        if not self.allow_tma:
            items.append(("allow_tma", Bool(False)))
        if self.latency is not None:
            items.append(("latency", Integer.create_i32(self.latency)))
        return Dictionary(items)


def make_load_store_hints(hints_by_arch: Mapping[str, LoadStoreHints]) -> OptimizationHints:
    return OptimizationHints([(arch, hints.as_dictionary())
                              for arch, hints in hints_by_arch.items()])


class AssumePredicate(TaggedAttribute):
    pass


@dataclass
class DivBy(AssumePredicate):
    divisor: int
    every: Optional[int] = None
    along: Optional[int] = None

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.DivBy._value_)
        encode_varint(self.divisor, buf)
        buf.append((self.every is not None) | ((self.along is not None) << 1))
        if self.every is not None:
            encode_signed_varint(self.every, buf)
        if self.along is not None:
            encode_signed_varint(self.along, buf)


@dataclass
class SameElements(AssumePredicate):
    values: Sequence[int]


@dataclass
class Bounded(AssumePredicate):
    lb: Optional[int]
    ub: Optional[int]

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Bounded._value_)
        buf.append((self.lb is not None) | ((self.ub is not None) << 1))
        if self.lb is not None:
            encode_signed_varint(self.lb, buf)
        if self.ub is not None:
            encode_signed_varint(self.ub, buf)


class Float(TaggedAttribute):
    def __init__(self, value_bits: int, ty: SimpleType, tt: TypeTable):
        self.type_id = tt.simple(ty)
        self.value_bits = value_bits
        self.bit_size = float_bit_size(ty)

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Float._value_)
        encode_typeid(self.type_id, buf)
        encode_ap_int(self.value_bits, self.bit_size, buf)


def encode_ap_int(val: int, bit_width: int, buf: bytearray):
    assert val >= 0
    if bit_width <= 8:
        buf.append(val)
    elif bit_width <= 64:
        encode_signed_varint(val, buf)
    else:
        raise NotImplementedError()


@dataclass
class Integer(TaggedAttribute):
    type_id: TypeId
    bitwidth: int
    value: int

    @staticmethod
    def create_i32(value: int) -> "Integer":
        return Integer(I32_TYPE_ID, 32, value)

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Integer._value_)
        encode_typeid(self.type_id, buf)
        encode_varint(self.value & ((1 << self.bitwidth) - 1), buf)


@dataclass
class Bool(TaggedAttribute):
    value: bool

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        buf.extend(AttributeTag.Bool._value_)
        buf.append(bool(self.value))
