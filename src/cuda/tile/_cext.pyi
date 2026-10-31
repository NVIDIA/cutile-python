# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
from typing import Any, Optional, Sequence, TypeAlias

from cuda.lang._enums import MemoryOrder, VectorReduction
from cuda.tile._context import TileContextConfig
from cuda.tile._memory_model import MemoryScope, MemorySpace
from cuda.tile._numeric_semantics import RoundingMode

Dim3: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


class DType:
    """A *data type* (or *dtype*) describes the type of the objects of an |array|, |tile|, or
    operation.

    |Dtypes| determine how values are stored in memory and how operations on those values are
    performed.
    """

    @property
    def bitwidth(self):
        """The number of bits in an element of the |data type|."""

    @property
    def name(self):
        """The name of the |data type|."""

    def __call__(self, value, /):
        """Construct a Scalar of this |data type| from a value."""


def is_numeric(t: DType, /) -> bool: ...
def is_boolean(t: DType, /) -> bool: ...
def is_integral(t: DType, /) -> bool: ...
def is_signed(t: DType, /) -> bool: ...
def is_float(t: DType, /) -> bool: ...
def is_unrestricted_float(t: DType, /) -> bool: ...
def is_restricted_float(t: DType, /) -> bool: ...
def is_arithmetic(t: DType, /) -> bool: ...
def _is_pointer_dtype(t: DType, /) -> bool: ...
def _is_foreign_pointer_dtype(t: DType, /) -> bool: ...

def _get_pointer_dtype(pointee_dtype: DType | None, memory_space: MemorySpace, /) -> DType: ...
def _get_foreign_pointer_dtype(pointee_dtype: DType, /) -> DType: ...
def _pointer_pointee_dtype(pointer_dtype: DType, /) -> DType: ...
def _pointer_memory_space(pointer_dtype: DType, /) -> MemorySpace: ...
def _foreign_pointer_pointee_dtype(foreign_pointer_dtype: DType, /) -> DType: ...

def integer_dtype_min(dtype: DType) -> int: ...
def integer_dtype_max(dtype: DType) -> int: ...
def _promote_dtypes(t1: DType, t2: DType, /) -> DType | None: ...
def encode_float(value: float, dtype: DType, /) -> int: ...
def decode_float(bits: int, dtype: DType, /) -> float: ...


bool_: DType
uint8: DType
uint16: DType
uint32: DType
uint64: DType
_int128: DType
int8: DType
int16: DType
int32: DType
int64: DType
float16: DType
float32: DType
float64: DType
bfloat16: DType
tfloat32: DType
float8_e4m3fn: DType
float8_e5m2: DType
float8_e8m0fnu: DType
float8_e5m3fnu: DType
float4_e2m1fn: DType
mbarrier: DType
cluster_launch_control_token: DType
tensor_map_descriptor: DType
float6_e2m3fn: DType
float6_e3m2fn: DType


def launch(stream,
           grid: Dim3,
           kernel,
           kernel_args: tuple[Any, ...],
           /, *,
           programmatic_dependent_launch: bool = False):
    ...


def launch_extended(stream,
                    block_count: Dim3,
                    thread_count: Dim3,
                    kernel,
                    kernel_args: tuple[Any, ...],
                    /, *,
                    cooperative: bool = False,
                    block_in_cluster_count: Dim3 | None = None,
                    preferred_block_in_cluster_count: Dim3 | None = None,
                    programmatic_dependent_launch: bool = False
                    ):
    ...


def get_compute_capability(device_id: int = 0) -> tuple[int, int]:
    ...


def get_driver_version():
    ...


def _get_max_grid_size(device_id, /):
    ...


def get_parameter_constraints_from_pyargs(dispatcher, pyargs, calling_convention, /):
    ...


class ConstantKind(enum.Enum):
    ...


def classify_constant(value, kernel_arg, /) -> ConstantKind | None:
    ...


def foreign_dtype_object_register(foreign, native):
    ...


def foreign_dtype_object_to_native(foreign, /):
    ...


def dev_features_enabled():
    ...


def cconv_v3_enabled():
    ...


class TileDispatcher:
    def __init__(self, parameter_annotations: Sequence):
        ...


class HostDispatcher:
    def __init__(self, parameter_annotations: Sequence):
        ...


class TileContext:
    def __init__(self, config: TileContextConfig):
        ...

    @property
    def config(self) -> TileContextConfig:
        ...

    @property
    def autotune_cache(self) -> Any | None:
        ...

    @autotune_cache.setter
    def autotune_cache(self, value: Any | None):
        ...


class CallingConvention:
    @staticmethod
    def cutile_python_v1() -> "CallingConvention":
        ...

    @staticmethod
    def cutile_python_v2() -> "CallingConvention":
        ...

    @staticmethod
    def from_code(code: str, /) -> "CallingConvention":
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def code(self) -> str:
        ...

    @property
    def version(self) -> int:
        ...


default_tile_context: TileContext


def _synchronize_context() -> None: ...
def _create_stream() -> int: ...
def _destroy_stream(stream: int) -> None: ...
def _benchmark(stream: int,
               grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
               kernel,
               pyargs_tuples: tuple[tuple[Any, ...], ...],
               flush_l2: bool = True,
               /) -> float: ...


def run_coroutine(coro):
    """
    Run a coroutine using a software stack to bypass the Python's recursion limit.
    Use resume_after() to break the call chain and push a new frame to the software stack.
    """


def _export_ipc_benchmark_payload(stream: int,
                                  grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
                                  kernel,
                                  pyargs_tuples: tuple[Any, ...],
                                  flush_l2: bool = True,
                                  /) -> bytes | None: ...


def _benchmark_with_ipc_payload(payload: bytes, /) -> float: ...


class TensorMapDataType(enum.Enum):
    UINT8 = ...
    UINT16 = ...
    UINT32 = ...
    INT32 = ...
    UINT64 = ...
    INT64 = ...
    FLOAT16 = ...
    FLOAT32 = ...
    FLOAT64 = ...
    BFLOAT16 = ...
    FLOAT32_FTZ = ...
    TFLOAT32 = ...
    TFLOAT32_FTZ = ...
    _16U4_ALIGN8B = ...
    _16U4_ALIGN16B = ...
    _16U6_ALIGN16B = ...


class TensorMapInterleave(enum.Enum):
    NONE = ...
    INTERLEAVE_16B = ...
    INTERLEAVE_32B = ...


class SwizzleMode(enum.Enum):
    SWIZZLE_NONE = ...
    SWIZZLE_32B = ...
    SWIZZLE_64B = ...
    SWIZZLE_128B = ...
    SWIZZLE_128B_ATOM_32B = ...
    SWIZZLE_128B_ATOM_32B_FLIP_8B = ...
    SWIZZLE_128B_ATOM_64B = ...


class TensorMapL2Promotion(enum.Enum):
    NONE = ...
    L2_64B = ...
    L2_128B = ...
    L2_256B = ...


class TensorMapFloatOOBFill(enum.Enum):
    NONE = ...
    NAN_REQUEST_ZERO_FMA = ...


class TMALoadMode(enum.Enum):
    TILE = ...
    IM2COL = ...
    IM2COL_W = ...
    IM2COL_W_128 = ...
    TILE_GATHER4 = ...


class TMAStoreMode(enum.Enum):
    TILE = ...
    IM2COL = ...
    TILE_SCATTER4 = ...


_TENSOR_MAP_DESCRIPTOR_BYTES: int
_TENSOR_MAP_DESCRIPTOR_ALIGNMENT: int
_TENSOR_MAP_MAX_RANK: int


def _tensor_map_data_type_bitwidth(data_type: int, /) -> int: ...


def _tensor_map_tiled(
    array: Any,
    tile_shape: tuple[int, ...],
    order: tuple[int, ...],
    interleave: int,
    swizzle: int,
    l2_promotion: int,
    oob_fill: int,
    /,
) -> bytes: ...


class BitstreamWriter:
    def fixed1(self, value: int) -> None: ...
    def fixed2(self, value: int) -> None: ...
    def fixed3(self, value: int) -> None: ...
    def fixed4(self, value: int) -> None: ...
    def fixed8(self, value: int) -> None: ...
    def fixed32(self, value: int) -> None: ...

    def vbr4(self, value: int) -> None: ...
    def vbr5(self, value: int) -> None: ...
    def vbr6(self, value: int) -> None: ...
    def vbr8(self, value: int) -> None: ...

    def align_to_word(self) -> int: ...
    def aligned_word(self, value: int) -> int: ...
    def patch_word(self, offset: int, value: int) -> None: ...

    def raw_blob(self, data: bytearray) -> None: ...

    def to_bytes(self) -> bytes: ...


class NVVMProgram:
    def add_module(self, contents: bytes | bytearray, name: str, /) -> None: ...
    def compile(self, options: Sequence[str], /) -> bytes: ...


class NVVM:
    def __init__(self, dll_path: str, /): ...
    def ir_version(self) -> tuple[int, int, int, int]: ...
    def version(self) -> tuple[int, int]: ...
    def create_program(self) -> NVVMProgram: ...


# ----- Tensors (cuda.tile.Tile, cuda.lang.Scalar / Vector / Pointer) -----

# A single value, e.g. an element of a vector or a value stored through a pointer
_ScalarLike: TypeAlias = "Scalar | Pointer | int | float | bool"


class Tile:
    """Class for |tile| objects."""

    @property
    def dtype(self) -> "DType":
        """The |data type| of the |tile|'s elements.

        Returns:
            DType (constant):
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of elements in each of the |tile|'s dimensions.

        Returns:
            tuple[const int,...]:
        """

    @property
    def ndim(self) -> int:
        """The number of dimensions in the |tile|.

        Returns:
            int (constant):
        """

    def item(self) -> "Tile":
        """Equivalent to self.reshape(()).

        Returns:
            Tile: A scalar tile.

        Examples:

            .. testcode::
                :template: setup_only.py

                @ct.kernel
                def kernel(x):
                    idx = ct.full((1,), 2, dtype=ct.int32).item()
                    tile = ct.load(x, (idx,), shape=(4,))
                    print(tile)

                x = torch.arange(16, device='cuda:0')
                ct.launch(stream, (1,), kernel, (x,))

            .. testoutput::

                [8, 9, 10, 11]
        """

    def extract(self, index, shape):
        """See :py:func:`extract`."""

    def insert(self, index, value) -> "Tile":
        """See :py:func:`insert`."""

    def reshape(self, shape) -> "Tile":
        """See :py:func:`reshape`."""

    def permute(self, axes) -> "Tile":
        """See :py:func:`permute`."""

    def transpose(self, axis0=None, axis1=None) -> "Tile":
        """See :py:func:`transpose`."""

    def astype(self, dtype, *, rounding_mode: Optional[RoundingMode] = None) -> "Tile":
        """See :py:func:`astype`."""

    def __index__(self) -> int:
        """0D Tile can be used as index in range"""

    def __getitem__(self, index) -> "Tile":
        """Syntax sugar for expand_dim"""

    def __add__(self, other) -> "Tile":
        ...

    def __sub__(self, other) -> "Tile":
        ...

    def __mul__(self, other) -> "Tile":
        ...

    def __truediv__(self, other) -> "Tile":
        ...

    def __floordiv__(self, other) -> "Tile":
        ...

    def __mod__(self, other) -> "Tile":
        ...

    def __pow__(self, other) -> "Tile":
        ...

    def __and__(self, other) -> "Tile":
        ...

    def __or__(self, other) -> "Tile":
        ...

    def __xor__(self, other) -> "Tile":
        ...

    def __radd__(self, other) -> "Tile":
        ...

    def __rsub__(self, other) -> "Tile":
        ...

    def __rmul__(self, other) -> "Tile":
        ...

    def __rtruediv__(self, other) -> "Tile":
        ...

    def __rfloordiv__(self, other) -> "Tile":
        ...

    def __rmod__(self, other) -> "Tile":
        ...

    def __rpow__(self, other) -> "Tile":
        ...

    def __rand__(self, other) -> "Tile":
        ...

    def __ror__(self, other) -> "Tile":
        ...

    def __rxor__(self, other) -> "Tile":
        ...

    def __ge__(self, other) -> "Tile":
        ...

    def __gt__(self, other) -> "Tile":
        ...

    def __le__(self, other) -> "Tile":
        ...

    def __lt__(self, other) -> "Tile":
        ...

    def __eq__(self, other) -> "Tile":
        ...

    def __ne__(self, other) -> "Tile":
        ...

    def __neg__(self) -> "Tile":
        ...

    def __invert__(self) -> "Tile":
        ...

    def __matmul__(self, other) -> "Tile":
        ...

    def __rmatmul__(self, other) -> "Tile":
        ...

    def __divmod__(self, other):
        ...

    def __rdivmod__(self, other):
        ...

    def __lshift__(self, other):
        ...

    def __rlshift__(self, other):
        ...

    def __rshift__(self, other):
        ...

    def __rrshift__(self, other):
        ...


class Scalar:

    @property
    def dtype(self) -> "DType": ...

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __floordiv__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __and__(self, other):
        ...

    def __or__(self, other):
        ...

    def __xor__(self, other):
        ...

    def __lshift__(self, other):
        ...

    def __rshift__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __rfloordiv__(self, other):
        ...

    def __rmod__(self, other):
        ...

    def __rpow__(self, other):
        ...

    def __rand__(self, other):
        ...

    def __ror__(self, other):
        ...

    def __rxor__(self, other):
        ...

    def __rlshift__(self, other):
        ...

    def __rrshift__(self, other):
        ...

    def __ge__(self, other):
        ...

    def __gt__(self, other):
        ...

    def __le__(self, other):
        ...

    def __lt__(self, other):
        ...

    def __eq__(self, other):
        ...

    def __ne__(self, other):
        ...

    def __neg__(self):
        ...

    def __invert__(self):
        ...

    def __divmod__(self, other):
        ...

    def __rdivmod__(self, other):
        ...


class Vector:
    """Fixed-size collection returned by vectorized pointer operations."""

    def __init__(self, *elements: _ScalarLike, dtype: DType | None = None) -> None:
        """Constructs a vector from scalar elements, optionally with an explicit dtype.

        Args:
            elements: Variable number of elements used to construct the vector.
            dtype: Data type of the vector.
        """

    @property
    def dtype(self) -> "DType": ...

    @property
    def element_count(self) -> int: ...

    def __getitem__(self, item): ...

    def __setitem__(self, key, value): ...

    def with_item(self, index: int, value: _ScalarLike) -> "Vector":
        """Return a new vector with one element replaced.

        Vectors have value semantics, so this operation does not modify the
        original vector. ``index`` must select an element of the vector.

        Args:
            index: Index in vector to replace.
            value: New value.
        """

    def astype(self, dtype: "DType") -> "Vector":
        """Convert each element to ``dtype``.

        Returns a new vector of the same length with the given dtype.

        Args:
            dtype: Target data type of the result vector.
        """

    def bitcast(self, dtype: "DType") -> "Vector":
        """Reinterpret each element's bits as ``dtype``.

        The element and target dtype must have the same bitwidth, producing a
        ``Vector`` of the same length.

        Args:
            dtype: Target element data type (same bitwidth as the current one).
        """

    def reinterpret_as_scalar(self, dtype: "DType"):
        """Reinterpret the whole vector's bits as a single scalar of ``dtype``.

        ``dtype``'s bitwidth must equal the vector's total bitwidth (element
        bitwidth times length).

        The bytes are packed **little-endian**, as if the vector were stored to
        memory and loaded back as ``dtype``.

        Args:
            dtype: Target scalar data type.
        """

    def reinterpret_as_vector(self, dtype: "DType", length: int) -> "Vector":
        """Reinterpret the whole vector's bits as ``Vector[dtype, length]``.

        ``length * dtype.bitwidth`` must equal the source vector's total
        bitwidth.

        The raw bytes are preserved and re-split **little-endian**, as if the
        vector were stored to memory and loaded back as the new vector type.

        Args:
            dtype: Target element data type of the result vector.
            length: Number of elements in the result vector.
        """

    def reduce(
        self,
        op: VectorReduction,
        /,
        *,
        propagate_nan: bool = False,
        reassociate: bool = False,
    ) -> Scalar:
        """Reduce vector ``self`` to a scalar.

        Args:
            op: Operation to apply to the vector elements.
            propagate_nan: For floating-point min and max, return NaN if any
                element is NaN.
            reassociate: Permit the compiler to change the operation order.
        """

    def __len__(self): ...

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __floordiv__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __and__(self, other):
        ...

    def __or__(self, other):
        ...

    def __xor__(self, other):
        ...

    def __lshift__(self, other):
        ...

    def __rshift__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __rfloordiv__(self, other):
        ...

    def __rmod__(self, other):
        ...

    def __rpow__(self, other):
        ...

    def __rand__(self, other):
        ...

    def __ror__(self, other):
        ...

    def __rxor__(self, other):
        ...

    def __rlshift__(self, other):
        ...

    def __rrshift__(self, other):
        ...

    def __ge__(self, other):
        ...

    def __gt__(self, other):
        ...

    def __le__(self, other):
        ...

    def __lt__(self, other):
        ...

    def __eq__(self, other):
        ...

    def __ne__(self, other):
        ...

    def __neg__(self):
        ...

    def __invert__(self):
        ...

    def __divmod__(self, other):
        ...

    def __rdivmod__(self, other):
        ...


class Pointer:
    """Address in host or CUDA device memory.

    A typed pointer identifies the data type at its address. An opaque pointer
    does not identify a data type. Pointer arithmetic and memory access require
    a typed pointer.
    Pointer offsets are given in element counts, not in bytes.
    """

    def __add__(self, other):
        """Return a pointer that is ``other`` elements after this pointer.

        Args:
            other: Integral scalar that gives the element offset.
        """

    def __radd__(self, other):
        """Return a pointer that is ``other`` elements after this pointer.

        Args:
            other: Integral scalar that gives the element offset.
        """

    def __sub__(self, other):
        """Return a pointer that is ``other`` elements before this pointer.

        Args:
            other: Integral scalar that gives the element offset.
        """

    def __getitem__(self, index):
        """Load one value at an element offset from this pointer.

        ``self[index]`` is equivalent to ``(self + index).load()``.

        Args:
            index: Integral scalar that gives the element offset.
        """

    def __setitem__(self, index, value):
        """Store one value at an element offset from this pointer.

        ``self[index] = value`` is equivalent to
        ``(self + index).store(value)``.

        Args:
            index: Integral scalar that gives the element offset.
            value: Value to store.
        """

    def load(
        self,
        *,
        count: int = 1,
        alignment: int | None = None,
    ) -> Scalar | Pointer | Vector:
        """Load one or more consecutive values from this address.

        This operation is valid only for a typed pointer.

        Args:
            count: Compile-time number of values to load. ``1``
                returns a scalar. A value greater than ``1`` returns a vector.
                For best performance, align a vector load to the total size of
                the vector in bytes.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have this alignment. If the value is ``None``, the compiler
                does not get an alignment hint.
        """

    def store(
        self,
        value: _ScalarLike | Vector,
        *,
        alignment: int | None = None,
    ) -> None:
        """Store one or more consecutive values at this address.

        This operation is valid only for a typed pointer. A scalar value stores
        one value. A vector stores all its elements in consecutive locations.

        Args:
            value: Scalar or vector to store. The value must be compatible with
                the pointee data type.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have this alignment. If the value is ``None``, the compiler
                does not get an alignment hint.
        """

    def atomic_load(
        self,
        *,
        count: int = 1,
        memory_order: MemoryOrder = MemoryOrder.ACQUIRE,
        memory_scope: MemoryScope = MemoryScope.DEVICE,
        mmio: bool = False,
        alignment: int | None = None,
    ) -> Scalar | Pointer | Vector:
        """Atomically load one or more consecutive values from this address.

        This operation is valid only for a typed pointer. The total access size
        must be a power-of-two number of bytes. A vector is loaded as one atomic
        value instead of atomic loads for each vector element.

        Args:
            count: Compile-time number of values to load. ``1``
                returns a scalar. A value greater than ``1`` returns a vector.
            memory_order: Memory order for the load. Supported values are
                ``MemoryOrder.RELAXED`` and ``MemoryOrder.ACQUIRE``.
            memory_scope: Scope of threads that participate in memory ordering.
            mmio: Whether to use a memory-mapped I/O access. MMIO requires
                ``MemoryOrder.RELAXED`` or ``MemoryOrder.ACQUIRE`` and
                ``MemoryScope.SYS``. The pointer must refer to global memory.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have at least this alignment. If the value is ``None``, the
                total access size is used as the alignment.
        """

    def atomic_store(
        self,
        value: _ScalarLike | Vector,
        *,
        memory_order: MemoryOrder = MemoryOrder.RELEASE,
        memory_scope: MemoryScope = MemoryScope.DEVICE,
        mmio: bool = False,
        alignment: int | None = None,
    ) -> None:
        """Atomically store one or more consecutive values to this address.

        This operation is valid only for a typed pointer. The total access size
        must be a power-of-two number of bytes. A vector is stored as one atomic
        value instead of atomic stores for each vector element.

        Args:
            value: Scalar or vector to store. The element data type must be
                compatible with the pointee data type.
            memory_order: Memory order for the store. Supported values are
                ``MemoryOrder.RELAXED`` and ``MemoryOrder.RELEASE``.
            memory_scope: Scope of threads that participate in memory ordering.
            mmio: Whether to use a memory-mapped I/O access. MMIO requires
                ``MemoryOrder.RELAXED`` or ``MemoryOrder.RELEASE`` and
                ``MemoryScope.SYS``. The pointer must refer to global memory.
            alignment: Minimum byte alignment that the compiler can assume.
                The value must be a positive power of two. The address must
                have at least this alignment. If the value is ``None``, the
                total access size is used as the alignment.
        """

    @property
    def pointer_dtype(self) -> DType:
        """Pointer data type of this pointer.

        This value is a compile-time constant.
        """

    @property
    def opaque(self) -> bool:
        """Whether the pointer has no pointee data type.

        This value is a compile-time constant.
        """

    @property
    def pointee_dtype(self) -> DType:
        """Data type of the value at this address.

        Access to this property causes a compilation error if the pointer is
        opaque.
        """

    @property
    def memory_space(self) -> MemorySpace:
        """CUDA memory space of this pointer.

        ``GENERIC`` may also refer to host address space for compiled host code.
        """


class OpaqueValue:
    """Value of an opaque data type, such as ``mbarrier`` or ``tensor_map_descriptor``."""

    @property
    def dtype(self) -> DType:
        """Data type of this value.

        This value is a compile-time constant.
        """


def _tensor_to_ir_var(tensor: Tile | Scalar | Vector | Pointer | OpaqueValue, /) -> Any:
    ...


def _make_symbolic_tensor(var: Any, use_tile: bool, /) \
        -> Tile | Scalar | Vector | Pointer | OpaqueValue:
    ...


def _make_constant_0d_tensor(value: int | float | bool, dtype: DType, use_tile: bool,
                             /) -> Tile | Scalar:
    ...


def _binary_arithmetic_tensorlike(fn: str, x: Any, y: Any, rounding_mode: Any,
                                  flush_to_zero: bool, propagate_nan: bool, /) -> Any:
    ...


def _dtype_constructor(x: Any, dtype: DType, use_tile: bool, /) -> Any:
    ...


def _tensorlike_astype(x: Any, dtype: DType, rounding_mode: Optional[RoundingMode],
                       use_tile: bool, /) -> Any:
    ...


def _tile_reshape(x: Any, shape: Any, /) -> Tile:
    ...


def _tile_item(x: Any, /) -> Tile:
    ...


def _tile_getitem(x: Any, key: Any, /) -> Tile:
    ...


def _vector_getitem(x: Any, key: Any, /) -> Any:
    ...


def _vector_with_item(x: Any, index: Any, value: Any, /) -> Any:
    ...


def _tile_matmul(x: Any, y: Any, /) -> Any:
    ...


def _divmod_tensorlike(x: Any, y: Any, /) -> tuple[Any, Any]:
    ...


def _tensorlike_negative(x: Any, compile_time_errors: bool, /) -> Any:
    ...


def _tensorlike_positive(x: Any, compile_time_errors: bool, /) -> Any:
    ...


def _tensorlike_invert(x: Any, compile_time_errors: bool, /) -> Any:
    ...


def _binary_wrapping_arithmetic_tensorlike(fn: str, x: Any, y: Any,
                                           compile_time_errors: bool, /) -> Any:
    ...


def _comparison_tensorlike(fn: str, x: Any, y: Any, /) -> Any:
    ...


def _binary_bitwise_tensorlike(fn: str, x: Any, y: Any, /) -> Any:
    ...


def _bitwise_shift_tensorlike(fn: str, x: Any, y: Any, /) -> Any:
    ...
