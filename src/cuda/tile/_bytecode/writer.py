# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from contextlib import contextmanager
from typing import Sequence, Mapping, Any, Iterator, Tuple, NamedTuple

from .attribute import AttributeArray, EntryHints, make_entry_hints
from .basic import encode_varint, StringTable, Table
from .code_builder import CodeBuilder, Value
from .constant import ConstantTable
from .debug_info import DebugAttrId, DebugAttrTable
from .encodings import SymbolVisibility
from .type import TypeTable, TypeId, encode_typeid
from .version import BytecodeVersion


class FunctionBuilder(NamedTuple):
    code_builder: CodeBuilder
    parameters: Tuple[Value, ...]


class GlobalSection:
    def __init__(self, string_table: StringTable, constant_table: ConstantTable,
                 version: BytecodeVersion):
        self._string_table = string_table
        self._constant_table = constant_table
        self._version = version
        self._defined_names = set()
        self._contents = bytearray()

    def define_global(self, name: str, type: TypeId, data: bytes, alignment: int = 0,
                      constant: bool = False,
                      symbol_visibility: SymbolVisibility = SymbolVisibility.Public):
        if name in self._defined_names:
            raise ValueError(f"Global `{name}` has already been defined")
        name_id = self._string_table[name.encode()]
        encode_varint(name_id.string_id, self._contents)
        encode_typeid(type, self._contents)
        encode_varint(self._constant_table.dense_constant(data).constant_id, self._contents)
        encode_varint(alignment, self._contents)
        if self._version >= BytecodeVersion.V_13_3:
            self._contents.extend(symbol_visibility._value_)
            encode_varint(int(constant), self._contents)
        self._defined_names.add(name)

    def is_defined(self, global_name: str):
        return global_name in self._defined_names


class BytecodeWriter:
    def __init__(self, buf: bytearray, version: BytecodeVersion):
        self._num_functions = 0
        self.debug_info = []
        self._buf = buf
        self._string_table = StringTable()
        self._debug_attr_table = DebugAttrTable(self._string_table, version)
        self._constant_table = ConstantTable()
        self._type_table = TypeTable(version)
        self._global_section = GlobalSection(self._string_table, self._constant_table, version)
        self.version = version

    @property
    def debug_attr_table(self) -> DebugAttrTable:
        return self._debug_attr_table

    @property
    def constant_table(self) -> ConstantTable:
        return self._constant_table

    @property
    def type_table(self) -> TypeTable:
        return self._type_table

    @property
    def global_section(self) -> GlobalSection:
        return self._global_section

    @contextmanager
    def function(self,
                 name: str,
                 parameter_types: Sequence[TypeId],
                 result_types: Sequence[TypeId],
                 entry_point: bool,
                 hints: Mapping[str, EntryHints],
                 debug_attr: DebugAttrId) -> Iterator[FunctionBuilder]:
        self._num_functions += 1
        encode_varint(self._string_table[name.encode()].string_id, self._buf)
        sig_ty = self._type_table.function(parameter_types, result_types)
        encode_typeid(sig_ty, self._buf)
        if self.version >= BytecodeVersion.V_13_5:
            self._buf.append(0x02 if entry_point else 0)
        else:
            self._buf.append((0x02 | (0x04 if hints else 0)) if entry_point else 0)
        self.debug_info.append([debug_attr])
        encode_varint(len(self.debug_info), self._buf)

        # optimization hints is required for entry point at v13.5+.
        if entry_point and (hints or self.version >= BytecodeVersion.V_13_5):
            make_entry_hints(hints).encode_tagged(self._string_table, self._buf)

        # arg attrs
        if self.version >= BytecodeVersion.V_13_5:
            AttributeArray(()).encode_tagged(self._string_table, self._buf)

        builder = CodeBuilder(buf=bytearray(),
                              version=self.version,
                              string_table=self._string_table,
                              constant_table=self._constant_table,
                              debug_attr_per_op=self.debug_info[-1])
        yield FunctionBuilder(builder, builder._make_value_tuple(len(parameter_types)))
        encode_varint(len(builder.buf), self._buf)
        self._buf.extend(builder.buf)


@contextmanager
def write_bytecode(num_functions: int, buf: bytearray,
                   version: BytecodeVersion) -> Iterator[BytecodeWriter]:
    _write_header(buf, version)

    with _section(_Section.Func, 8, buf) as section_buf:
        encode_varint(num_functions, section_buf)
        w = BytecodeWriter(section_buf, version)
        yield w
        assert w._num_functions == num_functions

    _write_global_section(w.global_section, buf)

    with _section(_Section.Constant, 8, buf) as section_buf:
        _write_table(w.constant_table, 8, section_buf)

    _write_debug_info_section(w.debug_info, w.debug_attr_table, buf)

    with _section(_Section.Type, 4, buf) as section_buf:
        _write_table(w.type_table, 4, section_buf)

    with _section(_Section.String, 4, buf) as section_buf:
        _write_table(w._string_table, 4, section_buf)

    buf.append(_Section.EndOfBytecode._value_)


def _write_header(buf: bytearray, version: BytecodeVersion):
    buf.extend(b"\x7fTileIR\x00")  # magic number
    buf.append(version.major())
    buf.append(version.minor())
    buf.extend(version.tag().to_bytes(2, "little"))


def patch_header_preview_flag(buf: bytearray, version: BytecodeVersion) -> None:
    tag = version.tag() | 1
    buf[10:12] = tag.to_bytes(2, "little")


class _Section(enum.IntEnum):
    EndOfBytecode = 0x00
    String = 0x01
    Func = 0x02
    Debug = 0x03
    Constant = 0x04
    Type = 0x05
    Global = 0x06
    NumSections = 0x07


def _pad_to(alignment: int, buf: bytearray):
    buf.extend(b'\xcb' * (-len(buf) % alignment))


@contextmanager
def _section(section_id: _Section, alignment: int, buf: bytearray):
    section_buf = bytearray()
    yield section_buf
    buf.append(section_id._value_ | (0x80 if alignment > 1 else 0))
    encode_varint(len(section_buf), buf)
    if alignment > 1:
        encode_varint(alignment, buf)
        _pad_to(alignment, buf)
    buf.extend(section_buf)


def _write_global_section(global_section: GlobalSection,
                          buf: bytearray):
    if len(global_section._defined_names) > 0:
        with _section(_Section.Global, 1, buf) as section_buf:
            encode_varint(len(global_section._defined_names), section_buf)
            section_buf.extend(global_section._contents)


def _write_debug_info_section(debug_info: Sequence[Sequence[DebugAttrId]],
                              attr_table: DebugAttrTable,
                              buf: bytearray):
    with _section(_Section.Debug, 8, buf) as section_buf:
        # Number of functions with debug info
        encode_varint(len(debug_info), section_buf)

        # For each function, offset into the index array
        _pad_to(4, section_buf)
        index_offset = 0
        for func_info in debug_info:
            section_buf.extend(index_offset.to_bytes(4, "little"))
            index_offset += len(func_info)

        encode_varint(index_offset, section_buf)

        # For each operation, the index of its debug attribute
        _pad_to(8, section_buf)
        for func_info in debug_info:
            for attr in func_info:
                section_buf.extend(attr.debug_attr_id.to_bytes(8, "little"))

        # Workaround for the decoder failing on empty tables
        if len(attr_table) == 0:
            attr_table = DebugAttrTable(StringTable(), attr_table._version)
            id = attr_table[b"\x00"]
            assert id.debug_attr_id == 1

        # Write the debug attribute table
        _write_table(attr_table, 4, section_buf)


def _write_table(table: Table[bytes, Any], index_size: int, buf: bytearray):
    # Write number of items
    items = sorted(table.items(), key=lambda x: table._unwrap_id(x[1]))
    encode_varint(len(items), buf)

    # For each item, write an offset into the data buffer
    _pad_to(index_size, buf)
    offset = 0
    for expected_id, (encoded_item, id) in enumerate(items, table._starting_id):
        assert expected_id == table._unwrap_id(id)
        buf.extend(offset.to_bytes(index_size, "little"))
        offset += len(encoded_item)

    # Write the data buffer
    for encoded_item, _ in items:
        buf.extend(encoded_item)
