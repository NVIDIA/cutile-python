# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass

from cuda.tile._bytecode.attribute import TaggedAttribute
from cuda.tile._bytecode.basic import (
    StringTable,
    encode_signed_varint,
    encode_varint,
)
from cuda.tile._bytecode.type import TypeId, encode_typeid


class _AttributeTag(enum.Enum):
    PreviewExprInt = 0xD1
    PreviewConcreteType = 0xDB
    PreviewExprConstStr = 0xE4


@dataclass(frozen=True)
class PreviewExprInt(TaggedAttribute):
    value: int

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        encode_varint(_AttributeTag.PreviewExprInt.value, buf)
        encode_signed_varint(self.value, buf)


@dataclass(frozen=True)
class PreviewConcreteType(TaggedAttribute):
    type_id: TypeId

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        encode_varint(_AttributeTag.PreviewConcreteType.value, buf)
        encode_typeid(self.type_id, buf)


@dataclass(frozen=True)
class PreviewExprConstStr(TaggedAttribute):
    value: str

    def encode_tagged(self, string_table: StringTable, buf: bytearray):
        encode_varint(_AttributeTag.PreviewExprConstStr.value, buf)
        encode_varint(string_table[self.value.encode()].string_id, buf)
