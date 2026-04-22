# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Optional, Sequence, Tuple

from .attribute import TaggedAttribute, OptimizationHints, AssumePredicate
from .basic import encode_varint
from .code_builder import (
    CodeBuilder, NestedBlockBuilder, Value, encode_optional_operand,
    encode_unsized_variadic_operands, encode_sized_variadic_operands, encode_operand
)
from .type import encode_typeid, encode_sized_typeid_seq, TypeId
from .version import BytecodeVersion


class AtomicRMWMode(enum.Enum):
    AND = b"\x00"
    OR = b"\x01"
    XOR = b"\x02"
    ADD = b"\x03"
    ADDF = b"\x04"
    MAX = b"\x05"
    MIN = b"\x06"
    UMAX = b"\x07"
    UMIN = b"\x08"
    XCHG = b"\x09"


class ComparisonOrdering(enum.Enum):
    UNORDERED = b"\x00"
    ORDERED = b"\x01"


class ComparisonPredicate(enum.Enum):
    EQUAL = b"\x00"
    NOT_EQUAL = b"\x01"
    LESS_THAN = b"\x02"
    LESS_THAN_OR_EQUAL = b"\x03"
    GREATER_THAN = b"\x04"
    GREATER_THAN_OR_EQUAL = b"\x05"


class IntegerOverflow(enum.Enum):
    NONE = b"\x00"
    NSW = b"\x01"
    NUW = b"\x02"
    NW = b"\x03"


class MemoryOrderingSemantics(enum.Enum):
    WEAK = b"\x00"
    RELAXED = b"\x01"
    ACQUIRE = b"\x02"
    RELEASE = b"\x03"
    ACQ_REL = b"\x04"


class MemoryScope(enum.Enum):
    TL_BLK = b"\x00"
    DEVICE = b"\x01"
    SYS = b"\x02"


class ProgramIDDim(enum.Enum):
    X = b"\x00"
    Y = b"\x01"
    Z = b"\x02"


class PtrAttr(enum.Enum):
    NONE = b"\x00"
    UNICAST = b"\x01"
    MULTICAST = b"\x02"


class RoundingMode(enum.Enum):
    NEAREST_EVEN = b"\x00"
    ZERO = b"\x01"
    NEGATIVE_INF = b"\x02"
    POSITIVE_INF = b"\x03"
    APPROX = b"\x04"
    FULL = b"\x05"
    NEAREST_INT_TO_ZERO = b"\x06"


class Signedness(enum.Enum):
    Unsigned = b"\x00"
    Signed = b"\x01"


class SymbolVisibility(enum.Enum):
    Public = b"\x00"
    Private = b"\x01"


def encode_AbsFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(0, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_AbsIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(1, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_AddFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(2, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_AddIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    overflow: IntegerOverflow,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(3, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_AllocaOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    num_elem: int,  # since 13.3
    alignment: int,  # since 13.3
    global_: bool,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(113, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(global_), _buf)
    # Attributes
    code_builder.encode_opattr_int(num_elem)
    code_builder.encode_opattr_int(alignment)
    return code_builder.new_op()


def encode_AndIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(4, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_AssertOp(
    code_builder: CodeBuilder,
    condition: Value,
    message: str,
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(5, _buf)
    # Attributes
    code_builder.encode_opattr_str(message)
    # Operands
    encode_operand(condition, _buf)
    return code_builder.new_op(0)


def encode_AssumeOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    value: Value,
    predicate: AssumePredicate,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(6, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_tagged(AssumePredicate, predicate)
    # Operands
    encode_operand(value, _buf)
    return code_builder.new_op()


def encode_Atan2Op(  # since 13.2
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.2
    x: Value,  # since 13.2
    y: Value,  # since 13.2
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(110, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(x, _buf)
    encode_operand(y, _buf)
    return code_builder.new_op()


def encode_AtomicCASTkoOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    result_token_type: TypeId,
    pointers: Value,
    cmp: Value,
    val: Value,
    mask: Optional[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: MemoryScope,
) -> Tuple[Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(7, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    encode_typeid(result_token_type, _buf)
    # Flags
    encode_varint((mask is not None)
                  | ((token is not None) << 1), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    # Operands
    encode_operand(pointers, _buf)
    encode_operand(cmp, _buf)
    encode_operand(val, _buf)
    encode_optional_operand(mask, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op(2)


def encode_AtomicRMWTkoOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    result_token_type: TypeId,
    pointers: Value,
    arg: Value,
    mask: Optional[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: MemoryScope,
    mode: AtomicRMWMode,
) -> Tuple[Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(8, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    encode_typeid(result_token_type, _buf)
    # Flags
    encode_varint((mask is not None)
                  | ((token is not None) << 1), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    code_builder.encode_opattr_enum(AtomicRMWMode, mode)
    # Operands
    encode_operand(pointers, _buf)
    encode_operand(arg, _buf)
    encode_optional_operand(mask, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op(2)


def encode_AtomicRedViewTkoOp(  # since 13.3
    code_builder: CodeBuilder,
    result_token_type: TypeId,  # since 13.3
    view: Value,  # since 13.3
    index: Sequence[Value],  # since 13.3
    value: Value,  # since 13.3
    token: Optional[Value],  # since 13.3
    memory_ordering_semantics: MemoryOrderingSemantics,  # since 13.3
    memory_scope: MemoryScope,  # since 13.3
    mode: AtomicRMWMode,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(117, _buf)
    # Variadic result types
    encode_sized_typeid_seq((result_token_type,), _buf)
    # Flags
    encode_varint((token is not None), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    code_builder.encode_opattr_enum(AtomicRMWMode, mode)
    # Operands
    encode_operand(view, _buf)
    encode_sized_variadic_operands(index, _buf)
    encode_operand(value, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op()


def encode_BitcastOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(9, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_BreakOp(
    code_builder: CodeBuilder,
    operands: Sequence[Value],
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(10, _buf)
    # Variadic result types
    encode_sized_typeid_seq((), _buf)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op(0)


def encode_BroadcastOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(11, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_CatOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    dim: int,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(12, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_int(dim)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_CeilOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(13, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_CmpFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    comparison_predicate: ComparisonPredicate,
    comparison_ordering: ComparisonOrdering,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(14, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(ComparisonPredicate, comparison_predicate)
    code_builder.encode_opattr_enum(ComparisonOrdering, comparison_ordering)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_CmpIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    comparison_predicate: ComparisonPredicate,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(15, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(ComparisonPredicate, comparison_predicate)
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_ConstantOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    value: bytes,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(16, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_dense_int_or_fp_elements(value)
    return code_builder.new_op()


def encode_ContinueOp(
    code_builder: CodeBuilder,
    operands: Sequence[Value],
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(17, _buf)
    # Variadic result types
    encode_sized_typeid_seq((), _buf)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op(0)


def encode_CosHOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(19, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_CosOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(18, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_DivFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(20, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_DivIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    signedness: Signedness,
    rounding: RoundingMode,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(21, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    code_builder.encode_opattr_enum(RoundingMode, rounding)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_EntryOp(
    code_builder: CodeBuilder,
    sym_name: str,
    function_type: TypeId,
    arg_attrs: Optional[Sequence[TaggedAttribute]],
    res_attrs: Optional[Sequence[TaggedAttribute]],
    optimization_hints: Optional[OptimizationHints],
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(22, _buf)
    # Flags
    encode_varint((arg_attrs is not None)
                  | ((res_attrs is not None) << 1)
                  | ((optimization_hints is not None) << 2), _buf)
    # Attributes
    code_builder.encode_opattr_str(sym_name)
    code_builder.encode_opattr_typeid(function_type)
    if arg_attrs is not None:
        code_builder.encode_opattr_array(arg_attrs)
    if res_attrs is not None:
        code_builder.encode_opattr_array(res_attrs)
    if optimization_hints is not None:
        code_builder.encode_opattr_optimization_hints(optimization_hints)
    return code_builder.new_op_with_nested_blocks(0, 1)


def encode_Exp2Op(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(24, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_ExpOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    rounding_mode: RoundingMode,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(23, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    if code_builder.version >= BytecodeVersion.V_13_3:
        code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    else:
        assert rounding_mode == RoundingMode.FULL
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_ExtIOp(
    code_builder: CodeBuilder,
    to_type: TypeId,
    from_: Value,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(37, _buf)
    # Result types
    encode_typeid(to_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(from_, _buf)
    return code_builder.new_op()


def encode_ExtractOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    indices: Sequence[Value],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(38, _buf)
    # Variadic result types
    encode_sized_typeid_seq((result_type,), _buf)
    # Operands
    encode_varint(1 + len(indices), _buf)
    encode_operand(source, _buf)
    encode_unsized_variadic_operands(indices, _buf)
    return code_builder.new_op()


def encode_FToFOp(
    code_builder: CodeBuilder,
    to_type: TypeId,
    from_: Value,
    rounding_mode: RoundingMode,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(42, _buf)
    # Result types
    encode_typeid(to_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(from_, _buf)
    return code_builder.new_op()


def encode_FToIOp(
    code_builder: CodeBuilder,
    to_type: TypeId,
    from_: Value,
    signedness: Signedness,
    rounding_mode: RoundingMode,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(43, _buf)
    # Result types
    encode_typeid(to_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(from_, _buf)
    return code_builder.new_op()


def encode_FloorOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(39, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_FmaOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    acc: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(40, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    encode_operand(acc, _buf)
    return code_builder.new_op()


def encode_ForOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    lowerBound: Value,
    upperBound: Value,
    step: Value,
    initValues: Sequence[Value],
    unsignedCmp: bool,  # since 13.2
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(41, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Flags
    _flag_bits = bool(unsignedCmp)
    assert _flag_bits < 1 or code_builder.version >= BytecodeVersion.V_13_2
    if code_builder.version >= BytecodeVersion.V_13_2:
        encode_varint(_flag_bits, _buf)
    # Operands
    encode_varint(3 + len(initValues), _buf)
    encode_operand(lowerBound, _buf)
    encode_operand(upperBound, _buf)
    encode_operand(step, _buf)
    encode_unsized_variadic_operands(initValues, _buf)
    return code_builder.new_op_with_nested_blocks(len(result_types), 1)


def encode_GetGlobalOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    name: str,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(44, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_str(name)
    return code_builder.new_op()


def encode_GetIndexSpaceShapeOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    src: Value,
) -> Sequence[Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(45, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Operands
    encode_operand(src, _buf)
    return code_builder.new_op(len(result_types))


def encode_GetNumTileBlocksOp(
    code_builder: CodeBuilder,
    gridSize_x_type: TypeId,
    gridSize_y_type: TypeId,
    gridSize_z_type: TypeId,
) -> Tuple[Value, Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(46, _buf)
    # Result types
    encode_typeid(gridSize_x_type, _buf)
    encode_typeid(gridSize_y_type, _buf)
    encode_typeid(gridSize_z_type, _buf)
    return code_builder.new_op(3)


def encode_GetTensorShapeOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    src: Value,
) -> Sequence[Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(47, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Operands
    encode_operand(src, _buf)
    return code_builder.new_op(len(result_types))


def encode_GetTileBlockIdOp(
    code_builder: CodeBuilder,
    blockId_x_type: TypeId,
    blockId_y_type: TypeId,
    blockId_z_type: TypeId,
) -> Tuple[Value, Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(48, _buf)
    # Result types
    encode_typeid(blockId_x_type, _buf)
    encode_typeid(blockId_y_type, _buf)
    encode_typeid(blockId_z_type, _buf)
    return code_builder.new_op(3)


def encode_GlobalOp(
    code_builder: CodeBuilder,
    sym_name: str,
    value: bytes,
    alignment: int,
    constant: bool,  # since 13.3
    symbol_visibility: SymbolVisibility,  # since 13.3
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(49, _buf)
    # Flags
    _flag_bits = bool(constant)
    assert _flag_bits < 1 or code_builder.version >= BytecodeVersion.V_13_3
    if code_builder.version >= BytecodeVersion.V_13_3:
        encode_varint(_flag_bits, _buf)
    # Attributes
    code_builder.encode_opattr_str(sym_name)
    code_builder.encode_opattr_dense_int_or_fp_elements(value)
    code_builder.encode_opattr_int(alignment)
    if code_builder.version >= BytecodeVersion.V_13_3:
        code_builder.encode_opattr_enum(SymbolVisibility, symbol_visibility)
    else:
        assert symbol_visibility == SymbolVisibility.Public
    return code_builder.new_op(0)


def encode_IToFOp(
    code_builder: CodeBuilder,
    to_type: TypeId,
    from_: Value,
    signedness: Signedness,
    rounding_mode: RoundingMode,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(59, _buf)
    # Result types
    encode_typeid(to_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(from_, _buf)
    return code_builder.new_op()


def encode_IfOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    condition: Value,
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(50, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Operands
    encode_operand(condition, _buf)
    return code_builder.new_op_with_nested_blocks(len(result_types), 2)


def encode_IntToPtrOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(51, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_IotaOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(58, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    return code_builder.new_op()


def encode_JoinTokensOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    tokens: Sequence[Value],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(60, _buf)
    # Variadic result types
    encode_sized_typeid_seq((result_type,), _buf)
    # Operands
    encode_varint(len(tokens), _buf)
    encode_unsized_variadic_operands(tokens, _buf)
    return code_builder.new_op()


def encode_LoadPtrTkoOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    result_token_type: TypeId,
    source: Value,
    mask: Optional[Value],
    paddingValue: Optional[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: Optional[MemoryScope],
    optimization_hints: Optional[OptimizationHints],
) -> Tuple[Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(61, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    encode_typeid(result_token_type, _buf)
    # Flags
    encode_varint((memory_scope is not None)
                  | ((optimization_hints is not None) << 1)
                  | ((mask is not None) << 2)
                  | ((paddingValue is not None) << 3)
                  | ((token is not None) << 4), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    if memory_scope is not None:
        code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    if optimization_hints is not None:
        code_builder.encode_opattr_optimization_hints(optimization_hints)
    # Operands
    encode_operand(source, _buf)
    encode_optional_operand(mask, _buf)
    encode_optional_operand(paddingValue, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op(2)


def encode_LoadViewTkoOp(
    code_builder: CodeBuilder,
    tile_type: TypeId,
    result_token_type: TypeId,
    view: Value,
    index: Sequence[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: Optional[MemoryScope],
    optimization_hints: Optional[OptimizationHints],
) -> Tuple[Value, Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(62, _buf)
    # Variadic result types
    encode_sized_typeid_seq((tile_type, result_token_type,), _buf)
    # Flags
    encode_varint((memory_scope is not None)
                  | ((optimization_hints is not None) << 1)
                  | ((token is not None) << 2), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    if memory_scope is not None:
        code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    if optimization_hints is not None:
        code_builder.encode_opattr_optimization_hints(optimization_hints)
    # Operands
    encode_operand(view, _buf)
    encode_sized_variadic_operands(index, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op(2)


def encode_Log2Op(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(64, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_LogOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(63, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_LoopOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    initValues: Sequence[Value],
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(65, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Operands
    encode_varint(len(initValues), _buf)
    encode_unsized_variadic_operands(initValues, _buf)
    return code_builder.new_op_with_nested_blocks(len(result_types), 1)


def encode_MakeGatherScatterViewOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    tensor_view: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(115, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(tensor_view, _buf)
    return code_builder.new_op()


def encode_MakePartitionViewOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    tensor_view: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(66, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(tensor_view, _buf)
    return code_builder.new_op()


def encode_MakeStridedViewOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    tensor_view: Value,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(116, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(tensor_view, _buf)
    return code_builder.new_op()


def encode_MakeTensorViewOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    base: Value,
    dynamicShape: Sequence[Value],
    dynamicStrides: Sequence[Value],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(67, _buf)
    # Variadic result types
    encode_sized_typeid_seq((result_type,), _buf)
    # Operands
    encode_operand(base, _buf)
    encode_sized_variadic_operands(dynamicShape, _buf)
    encode_sized_variadic_operands(dynamicStrides, _buf)
    return code_builder.new_op()


def encode_MakeTokenOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(68, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    return code_builder.new_op()


def encode_MaxFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    propagate_nan: bool,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(69, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(propagate_nan)
                  | (bool(flush_to_zero) << 1), _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MaxIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(70, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MinFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    propagate_nan: bool,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(71, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(propagate_nan)
                  | (bool(flush_to_zero) << 1), _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MinIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(72, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MmaFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    acc: Value,
    fast_acc: bool,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(73, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    _flag_bits = bool(fast_acc)
    assert _flag_bits < 1 or code_builder.version >= BytecodeVersion.V_13_3
    if code_builder.version >= BytecodeVersion.V_13_3:
        encode_varint(_flag_bits, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    encode_operand(acc, _buf)
    return code_builder.new_op()


def encode_MmaFScaledOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    lhs: Value,  # since 13.3
    rhs: Value,  # since 13.3
    acc: Value,  # since 13.3
    lhs_scale: Value,  # since 13.3
    rhs_scale: Value,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(114, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    encode_operand(acc, _buf)
    encode_operand(lhs_scale, _buf)
    encode_operand(rhs_scale, _buf)
    return code_builder.new_op()


def encode_MmaIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    acc: Value,
    signedness_lhs: Signedness,
    signedness_rhs: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(74, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness_lhs)
    code_builder.encode_opattr_enum(Signedness, signedness_rhs)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    encode_operand(acc, _buf)
    return code_builder.new_op()


def encode_ModuleOp(
    code_builder: CodeBuilder,
    sym_name: str,
    producer: Optional[str],  # since 13.3
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(75, _buf)
    # Flags
    _flag_bits = (producer is not None)
    assert _flag_bits < 1 or code_builder.version >= BytecodeVersion.V_13_3
    if code_builder.version >= BytecodeVersion.V_13_3:
        encode_varint(_flag_bits, _buf)
    # Attributes
    code_builder.encode_opattr_str(sym_name)
    if producer is not None:
        code_builder.encode_opattr_str(producer)
    return code_builder.new_op_with_nested_blocks(0, 1)


def encode_MulFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(76, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MulIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    overflow: IntegerOverflow,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(78, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_MulhiIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    x: Value,
    y: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(77, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(x, _buf)
    encode_operand(y, _buf)
    return code_builder.new_op()


def encode_NegFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(79, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_NegIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    overflow: IntegerOverflow,  # since 13.2
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(80, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    if code_builder.version >= BytecodeVersion.V_13_2:
        code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    else:
        assert overflow == IntegerOverflow.NONE
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_OffsetOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    ptr: Value,
    offset: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(81, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(ptr, _buf)
    encode_operand(offset, _buf)
    return code_builder.new_op()


def encode_OrIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(82, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_PackOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    source: Value,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(111, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_PermuteOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    permutation: Sequence[int],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(83, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_dense_int32_array(permutation)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_PowOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    exponent: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(84, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    encode_operand(exponent, _buf)
    return code_builder.new_op()


def encode_PrintTkoOp(
    code_builder: CodeBuilder,
    result_token_type: Optional[TypeId],  # since 13.2
    args: Sequence[Value],
    token: Optional[Value],  # since 13.2
    str: str,
) -> Optional[Value]:
    _buf = code_builder.buf
    # Opcode
    encode_varint(85, _buf)
    # Variadic result types
    result_types = []
    if code_builder.version >= BytecodeVersion.V_13_2:
        result_token_idx = len(result_types)
        result_types.append(result_token_type)
    else:
        assert result_token_type is None
        result_token_idx = None
    encode_sized_typeid_seq(result_types, _buf)
    # Flags
    _flag_bits = (token is not None)
    assert _flag_bits < 1 or code_builder.version >= BytecodeVersion.V_13_2
    if code_builder.version >= BytecodeVersion.V_13_2:
        encode_varint(_flag_bits, _buf)
    # Attributes
    code_builder.encode_opattr_str(str)
    # Operands
    encode_sized_variadic_operands(args, _buf)
    encode_optional_operand(token, _buf)
    results = code_builder.new_op(len(result_types))
    return None if result_token_idx is None else results[result_token_idx]


def encode_PtrToIntOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(86, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_PtrToPtrOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(87, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_ReduceOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    operands: Sequence[Value],
    dim: int,
    identities: Sequence[TaggedAttribute],
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(88, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Attributes
    code_builder.encode_opattr_int(dim)
    code_builder.encode_opattr_array(identities)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op_with_nested_blocks(len(result_types), 1)


def encode_RemFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(89, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_RemIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(90, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_ReshapeOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(91, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_ReturnOp(
    code_builder: CodeBuilder,
    operands: Sequence[Value],
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(92, _buf)
    # Variadic result types
    encode_sized_typeid_seq((), _buf)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op(0)


def encode_RsqrtOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(93, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_ScanOp(
    code_builder: CodeBuilder,
    result_types: Sequence[TypeId],
    operands: Sequence[Value],
    dim: int,
    reverse: bool,
    identities: Sequence[TaggedAttribute],
) -> NestedBlockBuilder:
    _buf = code_builder.buf
    # Opcode
    encode_varint(94, _buf)
    # Variadic result types
    encode_sized_typeid_seq(result_types, _buf)
    # Attributes
    code_builder.encode_opattr_int(dim)
    code_builder.encode_opattr_bool(reverse)
    code_builder.encode_opattr_array(identities)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op_with_nested_blocks(len(result_types), 1)


def encode_SelectOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    cond: Value,
    val_if_true: Value,
    val_if_false: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(95, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(cond, _buf)
    encode_operand(val_if_true, _buf)
    encode_operand(val_if_false, _buf)
    return code_builder.new_op()


def encode_ShLIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    overflow: IntegerOverflow,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(96, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_ShRIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    signedness: Signedness,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(97, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(Signedness, signedness)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_SinHOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(99, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_SinOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(98, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_SqrtOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(100, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_StorePtrTkoOp(
    code_builder: CodeBuilder,
    result_token_type: TypeId,
    destination: Value,
    value: Value,
    mask: Optional[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: Optional[MemoryScope],
    optimization_hints: Optional[OptimizationHints],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(101, _buf)
    # Result types
    encode_typeid(result_token_type, _buf)
    # Flags
    encode_varint((memory_scope is not None)
                  | ((optimization_hints is not None) << 1)
                  | ((mask is not None) << 2)
                  | ((token is not None) << 3), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    if memory_scope is not None:
        code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    if optimization_hints is not None:
        code_builder.encode_opattr_optimization_hints(optimization_hints)
    # Operands
    encode_operand(destination, _buf)
    encode_operand(value, _buf)
    encode_optional_operand(mask, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op()


def encode_StoreViewTkoOp(
    code_builder: CodeBuilder,
    result_token_type: TypeId,
    tile: Value,
    view: Value,
    index: Sequence[Value],
    token: Optional[Value],
    memory_ordering_semantics: MemoryOrderingSemantics,
    memory_scope: Optional[MemoryScope],
    optimization_hints: Optional[OptimizationHints],
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(102, _buf)
    # Variadic result types
    encode_sized_typeid_seq((result_token_type,), _buf)
    # Flags
    encode_varint((memory_scope is not None)
                  | ((optimization_hints is not None) << 1)
                  | ((token is not None) << 2), _buf)
    # Attributes
    code_builder.encode_opattr_enum(MemoryOrderingSemantics, memory_ordering_semantics)
    if memory_scope is not None:
        code_builder.encode_opattr_enum(MemoryScope, memory_scope)
    if optimization_hints is not None:
        code_builder.encode_opattr_optimization_hints(optimization_hints)
    # Operands
    encode_operand(tile, _buf)
    encode_operand(view, _buf)
    encode_sized_variadic_operands(index, _buf)
    encode_optional_operand(token, _buf)
    return code_builder.new_op()


def encode_SubFOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    rounding_mode: RoundingMode,
    flush_to_zero: bool,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(103, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Flags
    encode_varint(bool(flush_to_zero), _buf)
    # Attributes
    code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_SubIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
    overflow: IntegerOverflow,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(104, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_TanHOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
    rounding_mode: RoundingMode,  # since 13.2
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(106, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Attributes
    if code_builder.version >= BytecodeVersion.V_13_2:
        code_builder.encode_opattr_enum(RoundingMode, rounding_mode)
    else:
        assert rounding_mode == RoundingMode.FULL
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_TanOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    source: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(105, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_TruncIOp(
    code_builder: CodeBuilder,
    to_type: TypeId,
    from_: Value,
    overflow: IntegerOverflow,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(107, _buf)
    # Result types
    encode_typeid(to_type, _buf)
    # Attributes
    code_builder.encode_opattr_enum(IntegerOverflow, overflow)
    # Operands
    encode_operand(from_, _buf)
    return code_builder.new_op()


def encode_UnpackOp(  # since 13.3
    code_builder: CodeBuilder,
    result_type: TypeId,  # since 13.3
    source: Value,  # since 13.3
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(112, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(source, _buf)
    return code_builder.new_op()


def encode_XOrIOp(
    code_builder: CodeBuilder,
    result_type: TypeId,
    lhs: Value,
    rhs: Value,
) -> Value:
    _buf = code_builder.buf
    # Opcode
    encode_varint(108, _buf)
    # Result types
    encode_typeid(result_type, _buf)
    # Operands
    encode_operand(lhs, _buf)
    encode_operand(rhs, _buf)
    return code_builder.new_op()


def encode_YieldOp(
    code_builder: CodeBuilder,
    operands: Sequence[Value],
) -> None:
    _buf = code_builder.buf
    # Opcode
    encode_varint(109, _buf)
    # Variadic result types
    encode_sized_typeid_seq((), _buf)
    # Operands
    encode_varint(len(operands), _buf)
    encode_unsized_variadic_operands(operands, _buf)
    return code_builder.new_op(0)


__all__ = [
    'AtomicRMWMode',
    'ComparisonOrdering',
    'ComparisonPredicate',
    'IntegerOverflow',
    'MemoryOrderingSemantics',
    'MemoryScope',
    'ProgramIDDim',
    'PtrAttr',
    'RoundingMode',
    'Signedness',
    'SymbolVisibility',
    'encode_AbsFOp',
    'encode_AbsIOp',
    'encode_AddFOp',
    'encode_AddIOp',
    'encode_AllocaOp',
    'encode_AndIOp',
    'encode_AssertOp',
    'encode_AssumeOp',
    'encode_Atan2Op',
    'encode_AtomicCASTkoOp',
    'encode_AtomicRMWTkoOp',
    'encode_AtomicRedViewTkoOp',
    'encode_BitcastOp',
    'encode_BreakOp',
    'encode_BroadcastOp',
    'encode_CatOp',
    'encode_CeilOp',
    'encode_CmpFOp',
    'encode_CmpIOp',
    'encode_ConstantOp',
    'encode_ContinueOp',
    'encode_CosHOp',
    'encode_CosOp',
    'encode_DivFOp',
    'encode_DivIOp',
    'encode_EntryOp',
    'encode_Exp2Op',
    'encode_ExpOp',
    'encode_ExtIOp',
    'encode_ExtractOp',
    'encode_FToFOp',
    'encode_FToIOp',
    'encode_FloorOp',
    'encode_FmaOp',
    'encode_ForOp',
    'encode_GetGlobalOp',
    'encode_GetIndexSpaceShapeOp',
    'encode_GetNumTileBlocksOp',
    'encode_GetTensorShapeOp',
    'encode_GetTileBlockIdOp',
    'encode_GlobalOp',
    'encode_IToFOp',
    'encode_IfOp',
    'encode_IntToPtrOp',
    'encode_IotaOp',
    'encode_JoinTokensOp',
    'encode_LoadPtrTkoOp',
    'encode_LoadViewTkoOp',
    'encode_Log2Op',
    'encode_LogOp',
    'encode_LoopOp',
    'encode_MakeGatherScatterViewOp',
    'encode_MakePartitionViewOp',
    'encode_MakeStridedViewOp',
    'encode_MakeTensorViewOp',
    'encode_MakeTokenOp',
    'encode_MaxFOp',
    'encode_MaxIOp',
    'encode_MinFOp',
    'encode_MinIOp',
    'encode_MmaFOp',
    'encode_MmaFScaledOp',
    'encode_MmaIOp',
    'encode_ModuleOp',
    'encode_MulFOp',
    'encode_MulIOp',
    'encode_MulhiIOp',
    'encode_NegFOp',
    'encode_NegIOp',
    'encode_OffsetOp',
    'encode_OrIOp',
    'encode_PackOp',
    'encode_PermuteOp',
    'encode_PowOp',
    'encode_PrintTkoOp',
    'encode_PtrToIntOp',
    'encode_PtrToPtrOp',
    'encode_ReduceOp',
    'encode_RemFOp',
    'encode_RemIOp',
    'encode_ReshapeOp',
    'encode_ReturnOp',
    'encode_RsqrtOp',
    'encode_ScanOp',
    'encode_SelectOp',
    'encode_ShLIOp',
    'encode_ShRIOp',
    'encode_SinHOp',
    'encode_SinOp',
    'encode_SqrtOp',
    'encode_StorePtrTkoOp',
    'encode_StoreViewTkoOp',
    'encode_SubFOp',
    'encode_SubIOp',
    'encode_TanHOp',
    'encode_TanOp',
    'encode_TruncIOp',
    'encode_UnpackOp',
    'encode_XOrIOp',
    'encode_YieldOp',
]
