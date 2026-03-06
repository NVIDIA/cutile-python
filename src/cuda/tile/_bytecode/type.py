# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Sequence

from .basic import encode_varint, encode_int_list, Table
from .version import BytecodeVersion


@dataclass(frozen=True)
class TypeId:
    type_id: int


def encode_typeid(type_id: TypeId, buf: bytearray):
    encode_varint(type_id.type_id, buf)


def encode_sized_typeid_seq(type_ids: Sequence[TypeId], buf: bytearray):
    encode_varint(len(type_ids), buf)
    for i in type_ids:
        encode_varint(i.type_id, buf)


# For simplicity, we always add these to the type table
I1_TYPE_ID = TypeId(0)
I32_TYPE_ID = TypeId(1)


class SimpleType(enum.Enum):
    I1 = b"\x00"
    I8 = b"\x01"
    I16 = b"\x02"
    I32 = b"\x03"
    I64 = b"\x04"
    F16 = b"\x05"
    BF16 = b"\x06"
    F32 = b"\x07"
    TF32 = b"\x08"
    F64 = b"\x09"
    F8E4M3FN = b"\x0a"
    F8E5M2 = b"\x0b"
    Token = b"\x11"
    F8E8M0FNU = b"\x12"  # since 13.2
    F4E2M1FN = b"\x13"  # since 13.3


class _CompositeType(enum.Enum):
    Pointer = b"\x0c"
    Tile = b"\x0d"
    TensorView = b"\x0e"
    PartitionView = b"\x0f"
    Func = b"\x10"


class PaddingValue(enum.Enum):
    Missing = b""
    Zero = b"\x00"
    NegZero = b"\x01"
    Nan = b"\x02"
    PosInf = b"\x03"
    NegInf = b"\x04"


class TypeTable(Table[bytes, TypeId]):
    _wrapper_type = TypeId

    def __init__(self, version: BytecodeVersion):
        super().__init__()
        self.version = version
        self._predefine(SimpleType.I1._value_, I1_TYPE_ID)
        self._predefine(SimpleType.I32._value_, I32_TYPE_ID)

    def simple(self, t: SimpleType) -> TypeId:
        return self[t._value_]

    @property
    def I1(self) -> TypeId:
        return self.simple(SimpleType.I1)

    @property
    def I32(self) -> TypeId:
        return self.simple(SimpleType.I32)

    @property
    def I64(self) -> TypeId:
        return self.simple(SimpleType.I64)

    @property
    def F32(self) -> TypeId:
        return self.simple(SimpleType.F32)

    @property
    def Token(self) -> TypeId:
        return self.simple(SimpleType.Token)

    def tile(self, dtype: TypeId, shape: Sequence[int]) -> TypeId:
        buf = bytearray(_CompositeType.Tile._value_)
        encode_varint(dtype.type_id, buf)
        encode_int_list(shape, 8, buf)
        return self[bytes(buf)]

    def pointer(self, pointee: TypeId) -> TypeId:
        buf = bytearray(_CompositeType.Pointer._value_)
        encode_varint(pointee.type_id, buf)
        return self[bytes(buf)]

    def tensor_view(self,
                    dtype: TypeId,
                    shape: Sequence[int],
                    strides: Sequence[int]) -> TypeId:
        buf = bytearray(_CompositeType.TensorView._value_)
        encode_varint(dtype.type_id, buf)
        encode_int_list(shape, 8, buf)
        encode_int_list(strides, 8, buf)
        return self[bytes(buf)]

    def partition_view(self,
                       tile_shape: Sequence[int],
                       tensor_view: TypeId,
                       dim_map: Sequence[int],
                       padding_value: PaddingValue) -> TypeId:
        buf = bytearray(_CompositeType.PartitionView._value_)
        if self.version >= BytecodeVersion.V_13_3:
            # Unified bitfield: encode optional flags before parameters
            optional_flags = 0
            if padding_value != PaddingValue.Missing:
                optional_flags |= (1 << 0)
            encode_varint(optional_flags, buf)

        encode_int_list(tile_shape, 4, buf)
        encode_varint(tensor_view.type_id, buf)
        encode_int_list(dim_map, 4, buf)
        if self.version >= BytecodeVersion.V_13_3:
            buf.extend(padding_value._value_)
        else:
            encode_varint(int(padding_value != PaddingValue.Missing), buf)
            buf.extend(padding_value._value_)
        return self[bytes(buf)]

    def function(self, parameter_types: Sequence[TypeId], result_types: Sequence[TypeId]) -> TypeId:
        buf = bytearray(_CompositeType.Func._value_)
        encode_sized_typeid_seq(parameter_types, buf)
        encode_sized_typeid_seq(result_types, buf)
        return self[bytes(buf)]

    def _predefine(self, tag: bytes, expected_id: TypeId):
        if self[tag].type_id != expected_id.type_id:
            raise RuntimeError("Wrong type registration order")

    def _unwrap_id(self, id: TypeId) -> int:
        return id.type_id
