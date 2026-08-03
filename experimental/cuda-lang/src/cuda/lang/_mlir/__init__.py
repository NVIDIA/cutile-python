# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._builtins import APFloat
from ._builtins import APInt
from ._builtins import AffineMap
from ._builtins import Attribute
from ._builtins import Block  # noqa: F401
from ._builtins import BlockLabel  # noqa: F401
from ._builtins import DenseResourceElementsHandle
from ._builtins import IntegerSet
from ._builtins import NamedAttribute
from ._builtins import Operation  # noqa: F401
from ._builtins import Region
from ._builtins import SignednessSemantics
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from ._builtins import use_location  # noqa: F401
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import dataclasses
import struct


# ========= 'builtin' dialect of MLIR ==========


# ---- Interfaces ----


class AlignmentAttrOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ArgAndResultAttrsOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class BlobAttr:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class BranchOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class CallOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class CallableOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class CastOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ConditionallySpeculatable:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DLTIQueryInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DataLayoutEntryInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DataLayoutOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DataLayoutSpecInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DataLayoutTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DestructurableAccessorOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DestructurableAllocationOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DestructurableTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DeviceMappingAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class DeviceMaskingAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ElementsAttr:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class FloatType:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class FunctionOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class InferIntRangeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class InferShapedTypeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class InferStridedMetadataOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class InferTypeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class MemRefElementTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class MemRefLayoutAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class MemoryEffectOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class MemorySpaceCastConsumerOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class MemorySpaceCastOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OffsetSizeAndStrideOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OpAsmAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OpAsmOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class OpAsmTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class PromotableAllocationOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class PromotableMemOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class PromotableOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class PtrLikeTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class RegionBranchOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class RegionBranchTerminatorOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class RegionKindInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ReifyRankedShapedTypeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class SafeMemorySlotAccessOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class SelectLikeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ShapedDimOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ShapedType:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')

    DYNAMIC = -1 << 63

    def get_element_type(self) -> Type:
        raise NotImplementedError("get_element_type() must be implemented"
                                  " by subclasses of ShapedType")

    def clone_with(self, shape: Optional[Sequence[int]], element_type: Type) -> Type:
        raise NotImplementedError("clone_with() must be implemented"
                                  " by subclasses of ShapedType")


class SymbolOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class SymbolUserAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class SymbolUserOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TargetDeviceSpecInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TargetSystemSpecInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class TypedAttr:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')

    def get_type(self) -> Type:
        raise NotImplementedError("get_type() must be implemented by subclasses of TypedAttr")


class VectorElementTypeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class VectorTransferOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class VectorUnrollOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ViewLikeOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class WeightedBranchOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class WeightedRegionBranchOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Attributes ----


@dataclass(kw_only=True)
class AffineMapAttr(Attribute, MemRefLayoutAttrInterface, OpAsmAttrInterface):
    value: "AffineMap"

    def _print_mlir_unqualified(self, p):
        p("affine_map<")
        self.value.print_mlir(p)
        p(">")


@dataclass(kw_only=True)
class ArrayAttr(Attribute):
    value: "Sequence[Attribute]"

    def _print_mlir_unqualified(self, p):
        p("[")
        comma = ""
        for x in self.value:
            p(comma)
            x.print_mlir(p)
            comma = ", "
        p("]")


@dataclass(kw_only=True)
class BoolAttr(Attribute):
    value: bool

    def __bool__(self):
        return self.value

    def _print_mlir_unqualified(self, p):
        p("true" if self.value else "false")


class LocationAttr(Attribute):
    pass


@dataclass(kw_only=True)
class CallSiteLoc(LocationAttr):
    callee: "Location"
    caller: "Location"

    def _print_mlir_unqualified(self, p):
        p("callsite(")
        self.callee._print_mlir_unqualified(p)
        p(" at ")
        self.caller._print_mlir_unqualified(p)
        p(")")


@dataclass(kw_only=True)
class DenseArrayAttr(Attribute, BlobAttr):
    elementType: "Type"
    size: "int"
    rawData: "bytes"


class DenseElementsAttr(Attribute):
    pass


class _DenseArrayAttrImpl(DenseArrayAttr):
    def __init__(self, items, format: str, type: Type):
        self._format = format
        size = len(items)
        data = struct.pack("=" + format * size, *items)
        super().__init__(elementType=type, size=size, rawData=data)

    def _print_mlir_unqualified(self, p):
        p("array<")
        self.elementType.print_mlir(p)
        sep = ": "
        for x in struct.unpack("=" + self._format * self.size, self.rawData):
            p(sep)
            p(x)
            sep = ", "
        p(">")


class DenseI16ArrayAttr(_DenseArrayAttrImpl):
    def __init__(self, items: Sequence[int]):
        super().__init__(items, 'h', IntegerType.signless(16))


class DenseI32ArrayAttr(_DenseArrayAttrImpl):
    def __init__(self, items: Sequence[int]):
        super().__init__(items, 'i', IntegerType.signless(32))


class DenseI64ArrayAttr(_DenseArrayAttrImpl):
    def __init__(self, items: Sequence[int]):
        super().__init__(items, 'q', IntegerType.signless(64))


class DenseI8ArrayAttr(_DenseArrayAttrImpl):
    def __init__(self, items: Sequence[int]):
        super().__init__(items, 'b', IntegerType.signless(8))


@dataclass(kw_only=True)
class DenseIntOrFPElementsAttr(DenseElementsAttr, TypedAttr, ElementsAttr):
    type: "ShapedType"
    rawData: "bytes"

    def get_type(self):
        return self.type

    def _print_mlir_unqualified(self, p):
        p("dense<")
        p(self.rawData)
        p(">")


DenseTypedElementsAttr = DenseIntOrFPElementsAttr


class DenseIntElementsAttr(DenseIntOrFPElementsAttr):
    pass


@dataclass(kw_only=True)
class DenseResourceElementsAttr(Attribute, TypedAttr, ElementsAttr, BlobAttr):
    type: "ShapedType"
    rawHandle: "DenseResourceElementsHandle"

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class DenseStringElementsAttr(DenseElementsAttr, TypedAttr, ElementsAttr):
    type: "ShapedType"
    value: "Sequence[str]"

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class DictionaryAttr(Attribute):
    value: "Sequence[NamedAttribute]"

    def _print_mlir_unqualified(self, p):
        p("{")
        comma = ""
        for x in self.value:
            p(comma)
            p.print_escaped_string(x.name.value)
            if not isinstance(x.value, UnitAttr):
                p(" = ")
                x.value.print_mlir(p)
            comma = ", "
        p("}")


class DistinctAttr(Attribute):
    def __repr__(self):
        return f"DistinctAttr(id={id(self)})"


@dataclass(kw_only=True)
class FileLineColRange(LocationAttr):
    filename: "StringAttr"
    start_line: "int"
    start_column: "int"
    end_line: "int"
    end_column: "int"

    def _print_mlir_unqualified(self, p):
        self.filename._print_mlir_unqualified(p)
        p(f":{self.start_line}:{self.start_column}")
        if (self.end_line, self.end_column) == (
            self.start_line,
            self.start_column,
        ):
            return
        p(" to ")
        if self.end_line != self.start_line:
            p(self.end_line)
        p(f":{self.end_column}")


@dataclass(kw_only=True)
class SymbolRefAttr(Attribute):
    rootReference: "StringAttr"
    nestedReferences: "Sequence[FlatSymbolRefAttr]"

    def _print_mlir_unqualified(self, p):
        p(f"@{self.rootReference.value}")
        for ref in self.nestedReferences:
            p(f"::@{ref.rootReference.value}")


class FlatSymbolRefAttr(SymbolRefAttr):
    def __init__(self, value: "str | StringAttr"):
        if isinstance(value, str):
            value = StringAttr(value=value)
        super().__init__(rootReference=value, nestedReferences=())


@dataclass(kw_only=True)
class FloatAttr(Attribute, TypedAttr):
    type: "Type" = dataclasses.field(default_factory=lambda: NoneType())
    value: "APFloat"

    def get_type(self):
        return self.type

    def _print_mlir_unqualified(self, p):
        import math
        import struct
        from .._fp_utils import isnormal

        value = float(self.value)
        match self.type:
            case Float16Type():
                if isnormal(value, 16):
                    return p(self.value)
                bits = struct.unpack(">H", struct.pack(">e", value))[0]
                p(f"0x{bits:04X}")
            case BFloat16Type():
                if isnormal(value, 32):
                    return p(self.value)
                bits = struct.unpack(">I", struct.pack(">f", value))[0] >> 16
                p(f"0x{bits:04X}")
            case Float32Type():
                if isnormal(value, 32):
                    return p(self.value)
                bits = struct.unpack(">I", struct.pack(">f", value))[0]
                p(f"0x{bits:08X}")
            case Float64Type():
                if isnormal(value, 64):
                    return p(self.value)
                bits = struct.unpack(">Q", struct.pack(">d", value))[0]
                p(f"0x{bits:016X}")
            case _:
                if math.isfinite(value):
                    return p(self.value)
                raise TypeError(f"Cannot print non-normal FloatAttr {value} for type {self.type}")


@dataclass(kw_only=True)
class FusedLoc(LocationAttr):
    locations: "Sequence[Location]"
    metadata: "Attribute"


@dataclass(kw_only=True)
class IntegerAttr(Attribute, TypedAttr):
    type: "Type" = dataclasses.field(default_factory=lambda: NoneType())
    value: "APInt"

    def get_type(self):
        return self.type

    @staticmethod
    def make(type: "IntegerType | IndexType", value: int) -> "IntegerAttr":
        match type:
            case IntegerType():
                width = type.width
                signedness = type.signedness
            case IndexType():
                width = 64
                signedness = SignednessSemantics.SIGNLESS
            case _:
                raise TypeError(f"Invalid type for IntegerAttr: {type}")

        shifted = value >> (width - 1)
        match signedness:
            case SignednessSemantics.SIGNLESS: valid = (-1, 0, 1)
            case SignednessSemantics.UNSIGNED: valid = (0, 1)
            case SignednessSemantics.SIGNED: valid = (-1, 0)
        assert shifted in valid

        return IntegerAttr(type=type, value=value)

    def _print_mlir_unqualified(self, p):
        if (isinstance(self.type, IntegerType)
                and self.type.width == 1 and self.type.signedness == SignednessSemantics.SIGNLESS):
            p("true" if self.value else "false")
        else:
            p(self.value)


@dataclass(kw_only=True)
class IntegerSetAttr(Attribute, OpAsmAttrInterface):
    value: "IntegerSet"


Location = LocationAttr


@dataclass(kw_only=True)
class NameLoc(LocationAttr):
    name: "StringAttr"
    childLoc: "Location"


@dataclass(kw_only=True)
class OpaqueAttr(Attribute, TypedAttr):
    dialectNamespace: "StringAttr"
    attrData: "str"
    type: "Type" = dataclasses.field(default_factory=lambda: NoneType())

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class SparseElementsAttr(Attribute, TypedAttr, ElementsAttr):
    type: "ShapedType"
    indices: "DenseIntElementsAttr"
    values: "DenseElementsAttr"

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class StridedLayoutAttr(Attribute, MemRefLayoutAttrInterface):
    offset: "int"
    strides: "Sequence[int]"

    def _print_mlir_unqualified(self, p):
        p("strided<[")
        comma = ""
        for s in self.strides:
            p(comma)
            p("?" if s == ShapedType.DYNAMIC else s)
            comma = ", "
        p("]")
        if self.offset != 0:
            p(", offset: ")
            p("?" if self.offset == ShapedType.DYNAMIC else self.offset)
        p(">")


@dataclass(kw_only=True)
class StringAttr(Attribute, TypedAttr):
    value: "str"
    type: "Type" = dataclasses.field(default_factory=lambda: NoneType())

    def get_type(self):
        return self.type

    def _print_mlir_unqualified(self, p):
        p.print_escaped_string(self.value)


@dataclass(kw_only=True)
class TypeAttr(Attribute):
    value: "Type"

    def _print_mlir_unqualified(self, p):
        self.value.print_mlir(p)


@dataclass(kw_only=True)
class UnitAttr(Attribute):
    pass


@dataclass(kw_only=True)
class UnknownLoc(LocationAttr):

    def _print_mlir_unqualified(self, p):
        p("?")


# ---- Types ----


@dataclass(kw_only=True)
class BFloat16Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("bf16")


class BaseMemRefType(Type, PtrLikeTypeInterface, ShapedType):
    pass


@dataclass(kw_only=True)
class ComplexType(Type):
    elementType: "Type"


@dataclass(kw_only=True)
class Float128Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f128")


@dataclass(kw_only=True)
class Float16Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f16")


@dataclass(kw_only=True)
class Float32Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f32")


@dataclass(kw_only=True)
class Float4E2M1FNType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f4E2M1FN")


@dataclass(kw_only=True)
class Float64Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f64")


@dataclass(kw_only=True)
class Float6E2M3FNType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f6E2M3FN")


@dataclass(kw_only=True)
class Float6E3M2FNType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f6E3M2FN")


@dataclass(kw_only=True)
class Float80Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f80")


@dataclass(kw_only=True)
class Float8E3M4Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E3M4")


@dataclass(kw_only=True)
class Float8E4M3B11FNUZType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E4M3B11FNUZ")


@dataclass(kw_only=True)
class Float8E4M3FNType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E4M3FN")


@dataclass(kw_only=True)
class Float8E4M3FNUZType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E4M3FNUZ")


@dataclass(kw_only=True)
class Float8E4M3Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E4M3")


@dataclass(kw_only=True)
class Float8E5M2FNUZType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E5M2FNUZ")


@dataclass(kw_only=True)
class Float8E5M2Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E5M2")


@dataclass(kw_only=True)
class Float8E8M0FNUType(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("f8E8M0FNU")


@dataclass(kw_only=True)
class FloatTF32Type(Type, VectorElementTypeInterface, FloatType):

    def _print_mlir_unqualified(self, p):
        p("tf32")


@dataclass(kw_only=True)
class FunctionType(Type):
    inputs: "Sequence[Type]"
    results: "Sequence[Type]"

    def _print_mlir_unqualified(self, p):
        p("(")
        comma = ""
        for x in self.inputs:
            p(comma)
            x.print_mlir(p)
            comma = ", "
        p(") -> ")
        use_parens = len(self.results) != 1 or isinstance(self.results[0], FunctionType)
        if use_parens:
            p("(")
        comma = ""
        for x in self.results:
            p(comma)
            x.print_mlir(p)
            comma = ", "
        if use_parens:
            p(")")


@dataclass(kw_only=True)
class GraphType(Type):
    inputs: "Sequence[Type]"
    results: "Sequence[Type]"


@dataclass(kw_only=True)
class IndexType(Type, VectorElementTypeInterface):

    def _print_mlir_unqualified(self, p):
        p("index")


@dataclass(kw_only=True)
class IntegerType(Type, VectorElementTypeInterface):
    width: "int"
    signedness: "SignednessSemantics"

    @staticmethod
    def signless(width: int) -> "IntegerType":
        return IntegerType(width=width, signedness=SignednessSemantics.SIGNLESS)

    @staticmethod
    def unsigned(width: int) -> "IntegerType":
        return IntegerType(width=width, signedness=SignednessSemantics.UNSIGNED)

    @staticmethod
    def signed(width: int) -> "IntegerType":
        return IntegerType(width=width, signedness=SignednessSemantics.SIGNED)

    def _print_mlir_unqualified(self, p):
        match self.signedness:
            case SignednessSemantics.SIGNLESS: p("i")
            case SignednessSemantics.UNSIGNED: p("ui")
            case SignednessSemantics.SIGNED: p("si")
        p(self.width)


@dataclass(kw_only=True)
class MemRefType(BaseMemRefType, PtrLikeTypeInterface, ShapedType):
    shape: "Sequence[int]"
    elementType: "Type"
    layout: "MemRefLayoutAttrInterface"
    memorySpace: "Attribute"

    def _print_mlir_unqualified(self, p):
        p("memref<")
        for s in self.shape:
            p("?" if s == ShapedType.DYNAMIC else s)
            p("x")
        self.elementType.print_mlir(p)
        if self.layout is not None and (
            not isinstance(self.layout, AffineMapAttr)
            or not self.layout.value.is_identity()
        ):
            p(", ")
            self.layout.print_mlir(p)
        if self.memorySpace is not None:
            p(", ")
            self.memorySpace.print_mlir(p)
        p(">")

    def get_element_type(self) -> Type:
        return self.elementType


@dataclass(kw_only=True)
class NoneType(Type):
    pass


@dataclass(kw_only=True)
class OpaqueType(Type):
    dialectNamespace: "StringAttr"
    typeData: "str"


class TensorType(Type, ShapedType):
    pass


@dataclass(kw_only=True)
class RankedTensorType(TensorType, ShapedType):
    shape: "Sequence[int]"
    elementType: "Type"
    encoding: "Attribute"


@dataclass(kw_only=True)
class TupleType(Type):
    types: "Sequence[Type]"


@dataclass(kw_only=True)
class UnrankedMemRefType(BaseMemRefType, PtrLikeTypeInterface, ShapedType):
    elementType: "Type"
    memorySpace: "Attribute"


@dataclass(kw_only=True)
class UnrankedTensorType(TensorType, ShapedType):
    elementType: "Type"


@dataclass(kw_only=True)
class VectorType(Type, ShapedType):
    shape: "Sequence[int]"
    elementType: "Type"
    scalableDims: "Sequence[bool]"

    def _print_mlir_unqualified(self, p):
        p("vector<")
        for is_scalable, s in zip(self.scalableDims, self.shape, strict=True):
            if is_scalable:
                p("[")
            p("?" if s == ShapedType.DYNAMIC else s)
            if is_scalable:
                p("]")
            p("x")
        self.elementType.print_mlir(p)
        p(">")

    def get_element_type(self) -> Type:
        return self.elementType

    def clone_with(self, shape: Optional[Sequence[int]], element_type: Type) -> Type:
        return VectorType(
            shape=self.shape if shape is None else shape,
            elementType=element_type,
            scalableDims=self.scalableDims,
        )


# ---- Operators ----


def add_ModuleOp(
    *,
    sym_name: Optional[str] = None,
    sym_visibility: Optional[str] = None,
    bodyRegion: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if sym_name is not None:
        all_props.append(('sym_name', StringAttr(value=sym_name)))
    if sym_visibility is not None:
        all_props.append(('sym_visibility', StringAttr(value=sym_visibility)))
    return add_operation(
        name="builtin.module",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[bodyRegion],
    )


def add_UnrealizedConversionCastOp(
    *,
    outputs_types: Sequence[Type],
    inputs: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, ...]:
    all_props = []
    return add_operation(
        name="builtin.unrealized_conversion_cast",
        result_type=outputs_types,
        operands=list(inputs),
        properties=all_props,
        attributes=extra_attributes,
    )


# ---- Dialects ----
from . import arith  # noqa: F401, E402
from . import cf  # noqa: F401, E402
from . import llvm  # noqa: F401, E402
from . import nvvm  # noqa: F401, E402
from . import ptr  # noqa: F401, E402
from . import gpu  # noqa: F401, E402
from . import memref  # noqa: F401, E402
