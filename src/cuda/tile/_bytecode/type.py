# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Sequence

from .basic import encode_varint, encode_int_list
from .type_base import TypeId, _TypeTableBase, encode_sized_typeid_seq, PaddingValue, PtrAttr
from .type_base import encode_typeid as encode_typeid  # noqa: F401
from .version import BytecodeVersion


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
    I4 = b"\x16"  # since 13.3
    F8E5M3FNU = b"\x82\x01"  # since 13.4


class _CompositeType(enum.Enum):
    Pointer = b"\x0c"
    Tile = b"\x0d"
    TensorView = b"\x0e"
    PartitionView = b"\x0f"
    Function = b"\x10"
    GatherScatterView = b"\x14"  # since 13.3
    StridedView = b"\x15"  # since 13.3


# Predefined type IDs
I1_TYPE_ID = TypeId(0)
I32_TYPE_ID = TypeId(1)


class TypeTable(_TypeTableBase):
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

    def pointer(self, pointeeType: TypeId, ptrAttr: PtrAttr) -> TypeId:
        buf = bytearray(_CompositeType.Pointer._value_)
        use_unified_bitfield = self.version >= BytecodeVersion.V_13_3
        optional_flags = 0
        if ptrAttr != PtrAttr.Missing:
            optional_flags |= (1 << 0)
        if self.version >= BytecodeVersion.V_13_4 and use_unified_bitfield:
            encode_varint(optional_flags, buf)
        encode_varint(pointeeType.type_id, buf)
        if self.version >= BytecodeVersion.V_13_4:
            buf.extend(ptrAttr._value_)
        else:
            if ptrAttr != PtrAttr.Missing:
                raise ValueError(
                    "parameter 'ptrAttr' requires bytecode version 13.4+, "
                    "but targeting " + self.version.as_string())
        return self[bytes(buf)]

    def tile(self, elementType: TypeId, shape: Sequence[int]) -> TypeId:
        buf = bytearray(_CompositeType.Tile._value_)
        encode_varint(elementType.type_id, buf)
        encode_int_list(shape, 8, buf)
        return self[bytes(buf)]

    def tensor_view(self,
                    elementType: TypeId,
                    shape: Sequence[int],
                    strides: Sequence[int],
                    ptrAttr: PtrAttr) -> TypeId:
        buf = bytearray(_CompositeType.TensorView._value_)
        use_unified_bitfield = self.version >= BytecodeVersion.V_13_3
        optional_flags = 0
        if ptrAttr != PtrAttr.Missing:
            optional_flags |= (1 << 0)
        if self.version >= BytecodeVersion.V_13_4 and use_unified_bitfield:
            encode_varint(optional_flags, buf)
        encode_varint(elementType.type_id, buf)
        encode_int_list(shape, 8, buf)
        encode_int_list(strides, 8, buf)
        if self.version >= BytecodeVersion.V_13_4:
            buf.extend(ptrAttr._value_)
        else:
            if ptrAttr != PtrAttr.Missing:
                raise ValueError(
                    "parameter 'ptrAttr' requires bytecode version 13.4+, "
                    "but targeting " + self.version.as_string())
        return self[bytes(buf)]

    def partition_view(self,
                       tile_shape: Sequence[int],
                       tensor_view: TypeId,
                       dim_map: Sequence[int],
                       padding_value: PaddingValue) -> TypeId:
        buf = bytearray(_CompositeType.PartitionView._value_)
        use_unified_bitfield = self.version >= BytecodeVersion.V_13_3
        optional_flags = 0
        if padding_value != PaddingValue.Missing:
            optional_flags |= (1 << 0)
        if use_unified_bitfield:
            encode_varint(optional_flags, buf)
        encode_int_list(tile_shape, 4, buf)
        encode_varint(tensor_view.type_id, buf)
        encode_int_list(dim_map, 4, buf)
        if use_unified_bitfield:
            buf.extend(padding_value._value_)
        else:
            encode_varint(int(padding_value != PaddingValue.Missing), buf)
            buf.extend(padding_value._value_)
        return self[bytes(buf)]

    def gather_scatter_view(self,
                            tile_shape: Sequence[int],
                            tensor_view: TypeId,
                            sparse_dim: int,
                            padding_value: PaddingValue) -> TypeId:
        buf = bytearray(_CompositeType.GatherScatterView._value_)
        optional_flags = 0
        if padding_value != PaddingValue.Missing:
            optional_flags |= (1 << 0)
        encode_varint(optional_flags, buf)
        encode_int_list(tile_shape, 4, buf)
        encode_varint(tensor_view.type_id, buf)
        encode_varint(sparse_dim, buf)
        buf.extend(padding_value._value_)
        return self[bytes(buf)]

    def strided_view(self,
                     tile_shape: Sequence[int],
                     traversal_strides: Sequence[int],
                     tensor_view: TypeId,
                     dim_map: Sequence[int],
                     padding_value: PaddingValue) -> TypeId:
        buf = bytearray(_CompositeType.StridedView._value_)
        optional_flags = 0
        if padding_value != PaddingValue.Missing:
            optional_flags |= (1 << 0)
        encode_varint(optional_flags, buf)
        encode_int_list(tile_shape, 4, buf)
        encode_int_list(traversal_strides, 4, buf)
        encode_varint(tensor_view.type_id, buf)
        encode_int_list(dim_map, 4, buf)
        buf.extend(padding_value._value_)
        return self[bytes(buf)]

    def function(self, parameter_types: Sequence[TypeId], result_types: Sequence[TypeId]) -> TypeId:
        buf = bytearray(_CompositeType.Function._value_)
        encode_sized_typeid_seq(parameter_types, buf)
        encode_sized_typeid_seq(result_types, buf)
        return self[bytes(buf)]
