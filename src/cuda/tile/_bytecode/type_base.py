# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Sequence

from .basic import Table, encode_varint


@dataclass(frozen=True)
class TypeId:
    type_id: int


class PaddingValue(enum.Enum):
    Missing = b""
    Zero = b"\x00"
    NegZero = b"\x01"
    Nan = b"\x02"
    PosInf = b"\x03"
    NegInf = b"\x04"


class PtrAttr(enum.Enum):
    Missing = b""
    Default = b"\x00"


def encode_typeid(type_id: TypeId, buf: bytearray):
    encode_varint(type_id.type_id, buf)


def encode_sized_typeid_seq(type_ids: Sequence[TypeId], buf: bytearray):
    encode_varint(len(type_ids), buf)
    for i in type_ids:
        encode_varint(i.type_id, buf)


class _TypeTableBase(Table[bytes, TypeId]):
    _wrapper_type = TypeId

    def _predefine(self, tag: bytes, expected_id: TypeId):
        if self[tag].type_id != expected_id.type_id:
            raise RuntimeError("Wrong type registration order")

    def _unwrap_id(self, id: TypeId) -> int:
        return id.type_id
