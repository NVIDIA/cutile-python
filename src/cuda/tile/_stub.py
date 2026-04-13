# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import functools
import inspect
import textwrap
from typing import Annotated, TypeVar, Union, Literal, Optional, Protocol

from cuda.tile._memory_model import MemoryOrder, MemoryScope
from cuda.tile._execution import function
from cuda.tile._datatype import DType
from cuda.tile._numeric_semantics import RoundingMode, PaddingMode


###############################################################################
# Types


class ScalarProtocol(Protocol):
    @function
    def __index__(self) -> int:
        """Scalar can be used as index in range"""

    @function
    def __add__(self, other) -> "TileOrScalar":
        ...

    @function
    def __sub__(self, other) -> "TileOrScalar":
        ...

    @function
    def __mul__(self, other) -> "TileOrScalar":
        ...

    @function
    def __truediv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __floordiv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __mod__(self, other) -> "TileOrScalar":
        ...

    @function
    def __pow__(self, other) -> "TileOrScalar":
        ...

    @function
    def __and__(self, other) -> "TileOrScalar":
        ...

    @function
    def __or__(self, other) -> "TileOrScalar":
        ...

    @function
    def __xor__(self, other) -> "TileOrScalar":
        ...

    @function
    def __radd__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rsub__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rmul__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rtruediv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rfloordiv__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rmod__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rpow__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rand__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ror__(self, other) -> "TileOrScalar":
        ...

    @function
    def __rxor__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ge__(self, other) -> "TileOrScalar":
        ...

    @function
    def __gt__(self, other) -> "TileOrScalar":
        ...

    @function
    def __le__(self, other) -> "TileOrScalar":
        ...

    @function
    def __lt__(self, other) -> "TileOrScalar":
        ...

    @function
    def __eq__(self, other) -> "TileOrScalar":
        ...

    @function
    def __ne__(self, other) -> "TileOrScalar":
        ...


Scalar = int | float | ScalarProtocol

Shape = Union[int, tuple[int, ...]]
Order = Union[tuple[int, ...], Literal['C'], Literal['F']]


class Array:
    """Class for |global array| objects."""

    @property
    @function
    def dtype(self) -> "DType":
        """The |data type| of the |array|'s elements.

        Returns:
            DType (constant):
        """

    @property
    @function
    def shape(self) -> tuple[int, ...]:
        """The number of elements in each of the |array|'s dimensions.

        Returns:
            tuple[int32,...]:
        """

    @property
    @function
    def strides(self) -> tuple[int, ...]:
        """The number of elements to step in each dimension while traversing the |array|.

        Returns:
            tuple[int32,...]:
        """

    @property
    @function
    def ndim(self) -> int:
        """The number of dimensions in the |array|.

        Returns:
            int (constant):
        """

    def slice(self, axis, start, stop) -> "Array":
        """Creates a view of the |array| sliced along a single `axis`.

        The returned array references the same underlying memory as |array|,
        but with a restricted range from index `start` (inclusive) to `stop` (exclusive)
        along the specified axis. No data is copied.

        `axis` must be a constant integer. Negative values are supported and count
        from the last dimension (e.g., ``axis=-1`` refers to the last axis).

        `start` and `stop` must be integers (scalars or 0D tiles).
        They must satisfy ``0 <= start < N`` and ``start <= stop <= N``, where ``N``
        is the size of `array` along the sliced axis.

        For example, consider a 2-dimensional array ``A`` of shape ``(M, N)``.
        Slicing along axis 0 from `start` to `stop` produces an array of
        shape ``(stop - start, N)``:

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                sub = x.slice(axis=0, start=1, stop=3)
                print(ct.load(sub, (0, 0), shape=(2, 4)))

            x = torch.arange(16, device='cuda').reshape(4, 4)
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::

            [[4, 5, 6, 7], [8, 9, 10, 11]]

        Using NumPy slice notation for illustration, this is equivalent to::

            sub = A[start:stop, :]  # NumPy notation for reference only

        The slice bounds can be dynamic (runtime values):

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x, offset, length):
                sub = x.slice(axis=0, start=offset, stop=offset+length)
                print(ct.load(sub, (0,), shape=(4,)))
                print(ct.load(sub, (1,), shape=(4,)))

            x = torch.arange(16, device='cuda')
            ct.launch(stream, (1,), kernel, (x, 8, 8))

        .. testoutput::

            [8, 9, 10, 11]
            [12, 13, 14, 15]

        """
        return _m_array_slice(self, axis, start, stop)

    def tiled_view(self, tile_shape: Constant[Shape], *,
                   padding_mode: PaddingMode = PaddingMode.UNDETERMINED) -> "TiledView":
        """Creates a |tiled view| of this array with a fixed `tile_shape`.

        The resulting :class:`TiledView` partitions this array into a grid of
        equally sized tiles.

        Args:
            tile_shape (tuple[const int,...]): The shape of each tile in the view.
                Must have the same rank as this array.
            padding_mode (PaddingMode): The value used to pad tiles that extend
                beyond the array boundaries. By default, the padding value is
                undetermined.

        Returns:
            TiledView:

        Examples:

            .. testcode::
                :template: setup_only.py

                @ct.kernel
                def kernel(x):
                    tv = x.tiled_view((2, 4))
                    print(tv.load((0, 0)))
                    print(tv.load((1, 0)))

                x = torch.arange(16, device='cuda').reshape(4, 4)
                ct.launch(stream, (1,), kernel, (x,))

            .. testoutput::

                [[0, 1, 2, 3], [4, 5, 6, 7]]
                [[8, 9, 10, 11], [12, 13, 14, 15]]

        .. seealso::
            :ref:`Tiled Views <data-tiled-views>`
        """
        return _m_array_tiled_view(self, tile_shape, padding_mode=padding_mode)

    def get_raw_memory(self) -> "RawArrayMemory":
        """Returns an object that allows loading and storing by element offset.

        The returned object holds the array's base pointer. Use
        :py:meth:`RawArrayMemory.load_offset`
        and :py:meth:`RawArrayMemory.store_offset` with an offset in **elements** (no shape/stride
        index calculation). Useful when you already have memory offsets.

        Returns:
            RawArrayMemory:
        """
        return _m_array_get_raw_memory(self)


class RawArrayMemory:
    """Type stub for RawArrayMemory objects returned by :py:meth:`Array.get_raw_memory`."""

    @property
    @function
    def dtype(self) -> "DType":
        """The data type of the elements in the |RawArrayMemory|.

        Returns:
            DType (constant):
        """

    def load_offset(self, offset: "TileOrScalar", /, *,
                    mask: Optional["Tile"] = None,
                    padding_value: "TileOrScalar" = 0,
                    latency: Optional[int] = None) -> "Tile":
        """Loads from memory at base_ptr + offset (offset in elements).

        Args:
            offset: Element offset(s); scalar or tile of integer type.
            mask: Optional boolean mask; where False, padding_value is used instead of load.
            padding_value: Value used when mask is False; default 0.
            latency: Optional latency hint (1--10).

        Returns:
            Tile: Loaded tile; shape matches broadcast(offset).
        """
        return _m_raw_array_memory_load_offset(
            self, offset, mask=mask, padding_value=padding_value, latency=latency)

    def store_offset(self, offset: "TileOrScalar", value: "TileOrScalar", /, *,
                     mask: Optional["Tile"] = None,
                     latency: Optional[int] = None) -> None:
        """Stores to memory at base_ptr + offset (offset in elements).

        Args:
            offset: Element offset(s); scalar or tile of integer type.
            value: Value(s) to store; broadcast to offset shape.
            mask: Optional boolean mask; where False, no store occurs.
            latency: Optional latency hint (1--10).
        """
        return _m_raw_array_memory_store_offset(
            self, offset, value, mask=mask, latency=latency)


class Tile:
    """Class for |tile| objects."""

    @property
    @function
    def dtype(self) -> "DType":
        """The |data type| of the |tile|'s elements.

        Returns:
            DType (constant):
        """

    @property
    @function
    def shape(self) -> tuple[int, ...]:
        """The number of elements in each of the |tile|'s dimensions.

        Returns:
            tuple[const int,...]:
        """

    @property
    @function
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

                x = torch.arange(16, device='cuda')
                ct.launch(stream, (1,), kernel, (x,))

            .. testoutput::

                [8, 9, 10, 11]
        """
        return _m_tile_item(self)

    def extract(self, index, shape):
        """See :py:func:`extract`."""
        return extract(self, index, shape)

    def reshape(self, shape) -> "Tile":
        """See :py:func:`reshape`."""
        return reshape(self, shape)

    def permute(self, axes) -> "Tile":
        """See :py:func:`permute`."""
        return permute(self, axes)

    def transpose(self, axis0=None, axis1=None) -> "Tile":
        """See :py:func:`transpose`."""
        return transpose(self, axis0, axis1)

    def astype(self, dtype) -> "Tile":
        """See :py:func:`astype`."""
        return astype(self, dtype)

    @function
    def __index__(self) -> int:
        """0D Tile can be used as index in range"""

    def __getitem__(self, index) -> "Tile":
        """Syntax sugar for expand_dim"""
        return expand_dims(self, index)

    def __add__(self, other) -> "Tile":
        return add(self, other)

    def __sub__(self, other) -> "Tile":
        return sub(self, other)

    def __mul__(self, other) -> "Tile":
        return mul(self, other)

    def __truediv__(self, other) -> "Tile":
        return truediv(self, other)

    def __floordiv__(self, other) -> "Tile":
        return floordiv(self, other)

    def __mod__(self, other) -> "Tile":
        return mod(self, other)

    def __pow__(self, other) -> "Tile":
        return pow(self, other)

    def __and__(self, other) -> "Tile":
        return bitwise_and(self, other)

    def __or__(self, other) -> "Tile":
        return bitwise_or(self, other)

    def __xor__(self, other) -> "Tile":
        return bitwise_xor(self, other)

    def __radd__(self, other) -> "Tile":
        return add(other, self)

    def __rsub__(self, other) -> "Tile":
        return sub(other, self)

    def __rmul__(self, other) -> "Tile":
        return mul(other, self)

    def __rtruediv__(self, other) -> "Tile":
        return truediv(other, self)

    def __rfloordiv__(self, other) -> "Tile":
        return floordiv(other, self)

    def __rmod__(self, other) -> "Tile":
        return mod(other, self)

    def __rpow__(self, other) -> "Tile":
        return pow(other, self)

    def __rand__(self, other) -> "Tile":
        return bitwise_and(other, self)

    def __ror__(self, other) -> "Tile":
        return bitwise_or(other, self)

    def __rxor__(self, other) -> "Tile":
        return bitwise_xor(other, self)

    def __ge__(self, other) -> "Tile":
        return greater_equal(self, other)

    def __gt__(self, other) -> "Tile":
        return greater(self, other)

    def __le__(self, other) -> "Tile":
        return less_equal(self, other)

    def __lt__(self, other) -> "Tile":
        return less(self, other)

    def __eq__(self, other) -> "Tile":
        return equal(self, other)

    def __ne__(self, other) -> "Tile":
        return not_equal(self, other)

    def __neg__(self) -> "Tile":
        return negative(self)

    def __invert__(self) -> "Tile":
        return bitwise_not(self)

    def __matmul__(self, other) -> "Tile":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tile":
        return matmul(other, self)


TileOrScalar = Union[Tile, Scalar]


class TiledView:
    """Class for |tiled view| objects."""

    @property
    @function
    def dtype(self) -> "DType":
        """The data type of the elements in the |tiled view|.

        Returns:
            DType (constant):
        """

    @property
    @function
    def num_tiles(self) -> tuple[int, ...]:
        """The number of tiles in each of the |tiled view|'s dimensions.

        Returns:
            tuple[int32,...]:
        """

    @property
    @function
    def tile_shape(self) -> tuple[int, ...]:
        """The shape of tiles produced by each indexed access.

        Returns:
            tuple[const int,...]:
        """

    def load(self, index: Shape, *,
             latency: Optional[int] = None,
             allow_tma: Optional[bool] = None) -> Tile:
        """Loads a tile from the |tiled view| at the given tile `index`.

        The returned tile has shape :attr:`tile_shape`.

        For a tile that partially extends beyond the tiled view boundaries, out-of-bound elements
        are filled according to the view's padding mode.
        If the tile lies entirely outside the tiled view, the behavior is undefined.

        Args:
            index (tuple[int,...]): An index in the |tiled view|'s tile space.
            latency (const int): A hint indicating how heavy DRAM traffic will be. It shall be an
                integer between 1 (low) and 10 (high). By default, the compiler will infer the
                latency.
            allow_tma (const bool): If False, the load will not use TMA. By default, TMA is
                allowed.

        Returns:
            Tile:

        Examples:

            .. testcode::
                :template: setup_only.py

                @ct.kernel
                def kernel(x):
                    tv = x.tiled_view(4)
                    tile = tv.load(0)
                    print(tile)

                x = torch.arange(8, device='cuda')
                ct.launch(stream, (1,), kernel, (x,))

            .. testoutput::

                [0, 1, 2, 3]
        """
        return _m_tiled_view_load(self, index, latency=latency, allow_tma=allow_tma)

    def store(self, index: Shape, tile: Tile, *,
              latency: Optional[int] = None,
              allow_tma: Optional[bool] = None) -> None:
        """Stores a `tile` into the |tiled view| at the given tile `index`.

        The `tile`'s shape must be broadcastable to :attr:`tile_shape`.
        If the `tile`'s dtype differs from the view's dtype, an implicit cast is performed.

        For a tile that partially extends beyond the tiled view boundaries, out-of-bound elements
        are ignored.
        If the tile lies entirely outside the tiled view, the behavior is undefined.

        Args:
            index (tuple[int,...]): An index in the |tiled view|'s tile space.
            tile (Tile): The tile to store.
            latency (const int): A hint indicating how heavy DRAM traffic will be. It shall be an
                integer between 1 (low) and 10 (high). By default, the compiler will infer the
                latency.
            allow_tma (const bool): If False, the store will not use TMA. By default, TMA is
                allowed.

        Examples:

            .. testcode::
                :template: setup_only.py

                @ct.kernel
                def kernel(x):
                    tv = x.tiled_view(4)
                    tile = ct.full((4,), 99, dtype=ct.int32)
                    tv.store(0, tile)

                x = torch.zeros(8, dtype=torch.int32, device='cuda')
                ct.launch(stream, (1,), kernel, (x,))
                print(x.tolist())

            .. testoutput::

                [99, 99, 99, 99, 0, 0, 0, 0]
        """
        _m_tiled_view_store(self, index, tile, latency=latency, allow_tma=allow_tma)


###############################################################################
# Constantness Hints


class ConstantAnnotation:
    """
    A ``typing.Annotated`` metadata class indicating that an object shall be |constant embedded|.

    If an object of this class is passed as a metadata argument to a ``typing.Annotated`` type hint
    on a parameter, then the parameter shall be a constant embedded.
    """

    def __repr__(self):
        return "ConstantAnnotation()"


T = TypeVar("T")
Constant = Annotated[T, ConstantAnnotation()]
Constant.__doc__ = """A type hint indicating that a value shall be |constant embedded|.
It can be used either with (``Constant[int]``) or without (``Constant``, meaning a constant of any
type) an underlying type hint.
"""


###############################################################################
# Operations


@function
def bid(axis) -> int:
    """Gets the index of current block.

    Args:
        axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

    Returns:
        int32:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            bidx = ct.bid(0)
            bidy = ct.bid(1)
            bidz = ct.bid(2)
            print(f"Hello from block ({bidx}, {bidy}, {bidz})")

        .. testoutput::

            Hello from block (0, 0, 0)
    """


@function
def num_blocks(axis) -> int:
    """Gets the number of blocks along the axis.

    Args:
        axis (const int): The axis of the block index space. Possible values are 0, 1, 2.

    Returns:
        int32:

    Examples:

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel():
                bidx = ct.bid(0)
                bidy = ct.bid(1)
                bidz = ct.bid(2)
                nx = ct.num_blocks(0)
                ny = ct.num_blocks(1)
                nz = ct.num_blocks(2)
                master_block = (bidx == 0 and bidy == 0 and bidz == 0)
                if master_block:
                    print(f"Number of tile blocks: {(nx, ny, nz)}")

            ct.launch(stream, (2, 3, 4), kernel, ())

        .. testoutput::

            Number of tile blocks: (2, 3, 4)
    """


@function
def num_tiles(array: Array, /,
              axis: int,
              shape: Constant[Shape],
              order: Constant[Order] = "C") -> int:
    """Gets the number of tiles in the |tile space| of the array along the `axis`.

    Args:
        array (Array): An array object on a cuda device.
        axis (const int): The axis of the tile partition space to get the dim size.
        shape (const int...): A sequence of const integers definining the shape of the tile.
        order ("C" or "F", or tuple[const int,...]): Order of axis mapping. See :py:func:`load`.

    Returns:
        int32

    Examples:

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                tile_shape = (4, 8)
                num_tiles_row = ct.num_tiles(x, 0, shape=tile_shape)
                num_tiles_col = ct.num_tiles(x, 1, shape=tile_shape)
                print(f"The tile space has {num_tiles_row} rows and {num_tiles_col} columns.")

            x = torch.empty(42, 64, device='cuda')
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::

            The tile space has 11 rows and 8 columns.
    """


@function
def load(array: Array, /,
         index: Shape,
         shape: Constant[Shape], *,
         order: Constant[Order] = "C",
         padding_mode: PaddingMode = PaddingMode.UNDETERMINED,
         latency: Optional[int] = None,
         allow_tma: Optional[bool] = None,
         memory_order: MemoryOrder = MemoryOrder.WEAK,
         memory_scope: MemoryScope = MemoryScope.NONE) -> Tile:
    """Loads a tile from the `array` which is partitioned into a |tile space|.

    The |tile space| is the result of partitioning the `array` into a grid of equally
    sized tiles specified by `shape`.

    For example, partitoning a 2D `array` of shape ``(M, N)`` using tile shape
    ``(tm, tn)`` results in a 2D tile space of size ``(cdiv(M, tm), cdiv(N, tn))``.
    An index into this tile space using index ``(i, j)`` produces a tile of size ``(tm, tn)``::

        t = ct.load(array, (i, j), (tm, tn))  # `t` has shape (tm, tn)

    The result tile `t` will be computed according to ::

        t[x, y] = array[i * tm + x, j * tn + y]  (for all 0<=x<tm, 0<=y<tn)

    For a tile that partially extends beyond the array boundaries, out-of-bound elements
    are filled according to `padding_mode`.
    If the tile lies entirely outside the array, the behavior is undefined.

    `order` is used to map the tile axis to the array axis. The transposed example of the above call
    to `load` would be::

        ct.load(array, (j, i), shape=(tn, tm), order=(1, 0))

    The result tile `t` will be computed according to ::

        t[y, x] = array[i * tm + x, j * tn + y]



    Args:
        array (Array): The |array| to load from.
        index (tuple[int,...]): An index in the |tile space| of ``shape`` from ``array``.
        shape (tuple[const int,...]): A tuple of const integers definining the shape of the tile.
        order ("C" or "F", or tuple[const int,...]): Permutation applied to array axes before the
            logical |tile space| is constructed. Can be specified either as a tuple of constants,
            or as one of the two special string literal values:

            * "C" is an alias for ``(0, 1, 2, ...)``, i.e. no permutation applied;
            * "F" is an alias for ``(..., 2, 1, 0)``, i.e. axis order is reversed.

        padding_mode (PaddingMode): The value used to pad the tile when it extends beyond the array
            boundaries. By default, the padding value is undetermined.
        latency (const int): A hint indicating how heavy DRAM traffic will be. It shall be an
            integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
        allow_tma (const bool): If False, the load will not use TMA. By default, TMA is allowed.
        memory_order (MemoryOrder): Memory ordering semantics for the load.
            Defaults to ``MemoryOrder.WEAK``. Valid values: ``WEAK``, ``RELAXED``, ``ACQUIRE``.
        memory_scope (MemoryScope): The scope of threads that participate in memory
            ordering. Only meaningful when ``memory_order`` is not ``WEAK``.

    Returns:
        Tile:

    Examples:

        Load from an 1D array.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                zero_pad = ct.PaddingMode.ZERO
                print(ct.load(x, (0,), shape=4))
                print(ct.load(x, (1,), shape=4))
                print(ct.load(x, (2,), shape=4, padding_mode=zero_pad))

            x = torch.arange(10, device='cuda')
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::

            [0, 1, 2, 3]
            [4, 5, 6, 7]
            [8, 9, 0, 0]

        Load from a 2D array in transposed order.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                print(ct.load(x, (0, 0), shape=(1, 4), order='F'))
                print(ct.load(x, (1, 0), shape=(1, 4), order='F'))
                print(ct.load(x, (2, 0), shape=(1, 4), order='F'))
                print(ct.load(x, (3, 0), shape=(1, 4), order='F'))

            x = torch.arange(16, device='cuda').reshape(4, 4)
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::

            [[0, 4, 8, 12]]
            [[1, 5, 9, 13]]
            [[2, 6, 10, 14]]
            [[3, 7, 11, 15]]

        Load from a 3D array with last 2 axes transposed.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                print(ct.load(x, (0, 0, 0), shape=(1, 2, 2), order=(0, 2, 1)))
                print(ct.load(x, (1, 0, 0), shape=(1, 2, 2), order=(0, 2, 1)))

            x = torch.arange(8, device='cuda').reshape(2, 2, 2)
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::

            [[[0, 2], [1, 3]]]
            [[[4, 6], [5, 7]]]

        Load a single scalar.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                for i in range(10):
                    tile = ct.load(x, (i,), shape=())
                    print(tile, end=" ")
                print()

            x = torch.arange(10, device='cuda')
            ct.launch(stream, (1,), kernel, (x,))

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

            0 1 2 3 4 5 6 7 8 9 

    .. seealso::
        - :py:func:`store`
        - :py:func:`gather`
        - |Tile space|
    """


@function
def store(array: Array, /,
          index: Shape,
          tile: TileOrScalar, *,
          order: Constant[Order] = "C",
          latency: Optional[int] = None,
          allow_tma: Optional[bool] = None,
          memory_order: MemoryOrder = MemoryOrder.WEAK,
          memory_scope: MemoryScope = MemoryScope.NONE) -> None:
    """Stores a `tile` value into the `array` at the `index` of its |tile space|.

    The |tile space| is the result of partitioning the `array` into a grid of tiles
    with equal size defined by the shape of the `tile`.

    For example, given a tile `t` of shape ``(tm, tn)`` and array of shape ``(M, N)``::

        # tile `t` has shape (tm, tn)
        ct.store(array, (i, j), t)

    The above call to `store` will store elements according to::

        array[i * tm + x, i * tn + y] = t[x, y]  (for 0<=x<tm, 0<=y<tn)

    For a tile that partially extends beyond the array boundaries, out-of-bound elements
    are ignored.
    If the tile lies entirely outside the array, the behavior is undefined.

    Args:
        array (Array): The |array| to store to.
        index (tuple[int,...]): An index in the |tile space| of ``array``.
            ``shape`` is inferred from the ``tile`` argument.
        tile (Tile): The |tile| to store. The rank of the tile must match rank of the array,
            unless it is a scalar or 0d tile.
        order ("C" or "F", or tuple[const int,...]): Order of axis mapping. See :py:func:`load`.
        latency (int, optional): A hint indicating how heavy DRAM traffic will be. It shall be an
            integer between 1 (low) and 10 (high). By default, the compiler will infer the latency.
        allow_tma (bool, optional): If False, the store will not use TMA.
            By default, TMA is allowed.
        memory_order (MemoryOrder): Memory ordering semantics for the store.
            Defaults to ``MemoryOrder.WEAK``. Valid values: ``WEAK``, ``RELAXED``, ``RELEASE``.
        memory_scope (MemoryScope): The scope of threads that participate in memory
            ordering. Only meaningful when ``memory_order`` is not ``WEAK``.

    Examples:

        Store into 1D array, partial out of bound store is ignored.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                tile = ct.ones((4,), dtype=x.dtype)
                ct.store(x, (0,), tile)
                ct.store(x, (1,), tile * 2)

            x = torch.zeros(6, dtype=torch.int32, device='cuda')
            ct.launch(stream, (1,), kernel, (x,))
            print(x.tolist())

        .. testoutput::

            [1, 1, 1, 1, 2, 2]

        When storing with a scalar (0d tile), it is broadcasted to the rank of the array.

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                ct.store(x, (0, 0), tile=0)
                ct.store(x, (0, 1), tile=1)
                ct.store(x, (1, 0), tile=2)
                ct.store(x, (1, 1), tile=3)

            x = torch.zeros(4, dtype=torch.int32, device='cuda').reshape(2, 2)
            ct.launch(stream, (1,), kernel, (x,))
            print(x.tolist())

        .. testoutput::

            [[0, 1], [2, 3]]

    .. seealso::
        - :py:func:`load`
        - :py:func:`scatter`
        - |Tile space|
    """


@function
def gather(array, indices, /, *, mask=None, padding_value=0, check_bounds=True,
           latency=None) -> Tile:
    """
    Loads a tile from the `array` elements specified by `indices`.

    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.

    The result shape will be the same as the broadcasted shape of indices.

    For example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``.
    Then the result tile will have the broadcasted shape ``(M, N, K)``::

        t = ct.gather(array, (ind0, ind1))   # `t` has shape (M, N, K)

    The result tile `t` will be computed according to ::

        t[i, j, k] = array[ind0[i, j, 0], ind1[i, 0, k]]   (for all 0<=i<M, 0<=j<N, 0<=k<K)

    If the array is 1-dimensional, `indices` can be passed as a tile rather than a tuple.
    This is a convenience notation that is strictly equivalent to passing a tuple of length 1::

        ct.gather(array, ind0)   # equivalent to ct.gather(array, (ind0,))

    A custom boolean `mask` can be provided to control which elements are loaded.
    The mask must be a scalar or a tile whose shape is broadcastable to the common shape
    of indices. Where the mask is ``False``, `padding_value` is returned instead of loading
    from the array.

    `gather()` checks that indices are within the bounds of the array. For indices
    that are out of bounds, `padding_value` will be returned (zero by default).
    It must be a scalar or a tile whose shape is broadcastable to the common shape of indices.

    If both `mask` and `check_bounds=True` are provided, the effective mask is the logical
    AND of both the custom mask and the bounds-checking mask. This means an element is only
    loaded if both the custom mask is ``True`` AND the index is within bounds.

    To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within the bounds
    of the array, and any out-of-bounds access will result in undefined behavior.

    Negative indices are interpreted as out of bounds, i.e. they don't follow the Python's
    negative index convention.
    """


@function
def scatter(array, indices, value, /, *, mask=None, check_bounds=True, latency=None):
    """
    Stores a tile `value` into the `array` elements specified by `indices`.

    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.

    `value` must be a scalar or a tile whose shape is broadcastable to the
    common shape of `indices`.

    For example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``, and ``value`` is a tile of shape of ``(N, K)``::

        # ind0: (M, N, 1),  ind1: (M, 1, K),  value: (N, K)
        ct.scatter(array, (ind0, ind1), value)

    The above call to `scatter` will store elements according to ::

        array[ind0[i, j, 0], ind1[i, 0, k]] = value[j, k]

    If the array is 1-dimensional, `indices` can be passed as a tile rather than a tuple.
    This is a convenience notation that is strictly equivalent to passing a tuple of length 1::

        ct.scatter(array, ind0, value)   # equivalent to ct.scatter(array, (ind0,), value)

    A custom boolean `mask` can be provided to control which elements are stored.
    The mask must be a scalar or a tile whose shape is broadcastable to the common shape
    of indices. Where the mask is ``False``, no store occurs.

    `scatter()` checks that indices are within the bounds of the array. For indices
    that are out of bounds, nothing is stored.

    If both `mask` and `check_bounds=True` are provided, the effective mask is the logical
    AND of both the custom mask and the bounds-checking mask. This means an element is only
    stored if both the custom mask is ``True`` AND the index is within bounds.

    To disable bounds checking, set `check_bounds` to ``False``. In this mode, the caller
    is responsible for ensuring that all indices are within the bounds of the array, and
    any out-of-bounds access will result in undefined behavior.
    """


# =========== Atomic ============


@function
def atomic_cas(array, indices, expected, desired, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic compare-and-swap on array elements with given indices.

    For each specified index, `atomic_cas()` compares the corresponding array element
    to the `expected` value. If it matches, it is then overwritten with the `desired` value;
    otherwise, no update is performed. In either case, the old value of the element is returned.
    For each individual element, the described compare-and-swap operation is performed atomically,
    but the operation as a whole is not atomic, and the order of individual updates is unspecified.

    `atomic_cas()` follows the same convention as :py:func:`gather()` and :py:func:`scatter()`:
    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.
    If the array is 1-dimensional, `indices` can be passed as a single tile
    rather than a tuple of length 1.

    `expected` and `desired` must be scalars or tiles whose shapes are broadcastable
    to the common shape of `indices`.

    By default, `atomic_cas()` checks that indices are within the bounds of the array.
    For indices that are out of bounds, no operation is performed, and a corresponding `expected`
    value is returned. To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within
    the bounds of the array, and any out-of-bounds access will result in undefined behavior.

    As an example, consider a 2-dimensional array. In this case, indices must be a tuple
    of length 2. Suppose that ``ind0`` and ``ind1`` are integer tiles
    of shapes ``(M, N, 1)`` and ``(M, 1, K)``, and both ``expected`` and ``desrired``
    are tiles of shape of ``(N, K)``::

        # ind0: (M, N, 1),  ind1: (M, 1, K),  expected: (N, K),  desired: (N, K)
        ct.atomic_cas(array, (ind0, ind1), expected, desired)

    The above call to `atomic_cas` will behave similarly to the following pseudocode::

        in parallel, for all (i, j, k) such that 0<=i<M, 0<=j<N, i<=k<K:
            if not check_bounds or (0 <= ind0[i, j, 0] < array.shape[0]
                                    and 0 <= ind1[i, 0, k] < array.shape[1]):
                do atomically:
                    actual = array[ind0[i, j, 0], ind1[i, 0, k]]
                    if actual == expected[j, k]:
                        array[ind0[i, j, 0], ind1[i, 0, k]] = desired[j, k]
                result[i, j, k] = actual
            else:
                result[i, j, k] = expected[j, k]

    Examples:

        .. testcode::
            :template: setup_only.py

            @ct.kernel
            def kernel(x):
                indices = ct.arange(4, dtype=ct.int32)
                expected = ct.full((4,), 0, dtype=ct.int32)
                desired = ct.full((4,), 42, dtype=ct.int32)
                old = ct.atomic_cas(x, indices, expected, desired)
                print(old)

            x = torch.tensor([0, 1, 0, 1], device='cuda')
            ct.launch(stream, (1,), kernel, (x,))
            print(x.tolist())

        .. testoutput::

            [0, 1, 0, 1]
            [42, 1, 42, 1]
    """


def _doc_atomic_rmw_op(f):
    op_name = f.__name__
    f.__doc__ += f"""\

    For each individual element, the operation is performed atomically,
    but the operation as a whole is not atomic, and the order of individual writes is unspecified.

    `{op_name}()` follows the same convention as :py:func:`gather()` and :py:func:`scatter()`:
    `indices` must be a tuple whose length equals the `array` rank.
    All elements of this tuple must be integer tiles or scalars of the same shape,
    or different shapes that are broadcastable to a common shape.
    If the array is 1-dimensional, `indices` can be passed as a single tile
    rather than a tuple of length 1.

    `update` must be a scalar or a tile whose shape is broadcastable to the
    common shape of `indices`.

    By default, `{op_name}()` checks that indices are within the bounds of the array.
    For indices that are out of bounds, no operation is performed, and an implementation-defined
    value is returned. To disable bounds checking, set `check_bounds` to ``False``.
    In this mode, the caller is responsible for ensuring that all indices are within
    the bounds of the array, and any out-of-bounds access will result in undefined behavior.

    Examples:

    .. testcode::
        :template: setup_only.py

        @ct.kernel
        def kernel(x):
            indices = ct.arange(4, dtype=ct.int32)
            update = ct.full((4,), 10, dtype=ct.int32)
            old = ct.{op_name}(x, indices, update)
            print(old)

        x = torch.ones(4, dtype=torch.int32, device='cuda')
        ct.launch(stream, (1,), kernel, (x,))

    .. testoutput::

        [1, 1, 1, 1]
    """

    return f


@function
@_doc_atomic_rmw_op
def atomic_xchg(array, indices, update, /, *,
                check_bounds=True,
                memory_order=MemoryOrder.ACQ_REL,
                memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic exchange of array elements at given indices.

    For each specified index, `atomic_xchg()` stores the corresponding `update`
    to the array element at that location, and returns the original value of the element
    before the update.
    """


@function
@_doc_atomic_rmw_op
def atomic_add(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic post-increment of array elements at given indices.

    For each specified index, `atomic_add()` reads the corresponding array element,
    adds `update` to it, and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_max(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic maximum value assignment on array elements at given indices.

    For each specified index, `atomic_max()` reads the corresponding array element,
    computes the maximum between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_min(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic minimum value assignment on array elements at given indices.

    For each specified index, `atomic_min()` reads the corresponding array element,
    computes the minimum between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_and(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> TileOrScalar:
    """Bulk atomic AND operation on array elements at given indices.

    For each specified index, `atomic_and()` reads the corresponding array element,
    computes the bitwise AND between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_or(array, indices, update, /, *,
              check_bounds=True,
              memory_order=MemoryOrder.ACQ_REL,
              memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic OR operation on array elements at given indices.

    For each specified index, `atomic_or()` reads the corresponding array element,
    computes the bitwise OR between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


@function
@_doc_atomic_rmw_op
def atomic_xor(array, indices, update, /, *,
               check_bounds=True,
               memory_order=MemoryOrder.ACQ_REL,
               memory_scope=MemoryScope.DEVICE) -> Tile:
    """Bulk atomic XOR operation on array elements at given indices.

    For each specified index, `atomic_xor()` reads the corresponding array element,
    computes the bitwise XOR between its value and the corresponding value of `update`,
    and writes the modified value back to the same location.
    The original value of the element before the update is returned.
    """


# ======== Factory ==============


@function
def arange(size, /, *, dtype) -> Tile:
    """Creates a tile with value starting from 0 to `size - 1`.

    Args:
        size (const int): Size of the tile.
        dtype (DType): Datatype of the tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.arange(4, dtype=ct.int32)
            print(tile)

        .. testoutput::

            [0, 1, 2, 3]
    """


@function
def full(shape: Shape, fill_value: Scalar, dtype: DType) -> Tile:
    """Creates a tile filled with given value.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        fill_value (int | float | bool]): Value for the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.full((2, 2), 3, dtype=ct.int32)
            print(tile)

        .. testoutput::

            [[3, 3], [3, 3]]
    """


@function
def ones(shape, dtype) -> Tile:
    """Creates a tile filled with ones.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.ones((2, 2), dtype=ct.int32)
            print(tile)

        .. testoutput::

            [[1, 1], [1, 1]]
    """


@function
def zeros(shape, dtype) -> Tile:
    """Creates a tile filled with zeros.

    Args:
        shape (tuple[const int,...]):  The shape of the tile.
        dtype (DType): The |Data type| of the tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.zeros((2, 2), dtype=ct.int32)
            print(tile)

        .. testoutput::

            [[0, 0], [0, 0]]
    """

# =========== Matmul ============


@function
def mma(x, y, /, acc) -> Tile:
    """Matrix multiply-accumulate.

    Computes ``(x @ y) + acc`` as a single operation
    (where ``@`` denotes matrix multiplication).
    Preserves the dtype of `acc`.

    Args:
        x (Tile): LHS of the mma, 2D or 3D.
        y (Tile): RHS of the mma, 2D or 3D.
        acc (Tile): Accumulator of mma.

    Supported datatypes:

    +----------+---------------+
    | Input    |  Acc/Output   |
    +==========+===============+
    | f16      |  f16 or f32   |
    +----------+---------------+
    | bf16     |  f32          |
    +----------+---------------+
    | f32      |  f32          |
    +----------+---------------+
    | f64      |  f64          |
    +----------+---------------+
    | tf32     |  f32          |
    +----------+---------------+
    | f8e4m3fn |  f16 or f32   |
    +----------+---------------+
    | f8e5m2   |  f16 or f32   |
    +----------+---------------+
    | [u|i]8   |  i32          |
    +----------+---------------+

    If `x` and `y` have different dtype, they will NOT be promoted to common dtype.
    Shape of `x` and `y` will be broadcasted to up until the last two axes.

    Returns:
        Tile:

    Examples:

        2D x 2D with accumulation.

        .. testcode::
            :template: kernel_wrapper.py

            x = ct.ones((2, 4), dtype=ct.float32)
            y = ct.ones((4, 2), dtype=ct.float32)
            acc = ct.full((2, 2), 10.0, dtype=ct.float32)
            # (x @ y) + acc: each element = 1*4 + 10 = 14
            print(f"{ct.mma(x, y, acc):.1f}")

        .. testoutput::

            [[14.0, 14.0], [14.0, 14.0]]

        Batched: 3D x 2D with broadcast.

        .. testcode::
            :template: kernel_wrapper.py

            x = ct.ones((2, 2, 4), dtype=ct.float32)
            y = ct.ones((4, 2), dtype=ct.float32)
            acc = ct.zeros((2, 2, 2), dtype=ct.float32)
            print(f"{ct.mma(x, y, acc):.1f}")

        .. testoutput::

            [[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]
    """


@function
def mma_scaled(x, x_scale, y, y_scale, /, acc) -> Tile:
    """Block-scaled matrix multiply-accumulate.

    Computes a matrix multiply-accumulate where inputs are scaled by block scales
    along the K dimension before the mma::

        result[i, j] = sum(x[i, k] * x_scale[i, k // B] * y[k, j] * y_scale[k // B, j]
                           for k in range(K)) + acc[i, j]

    The scaling block size is ``B = K // K_s``, where ``K_s`` is the K dimension of the scale tile.
    ``K`` must be divisible by ``K_s``, and ``B`` must be one of the allowed values listed
    in the table below.

    Args:
        x (Tile): LHS input, 2D or 3D ``[..., M, K]``.
        x_scale (Tile): Scale factors for x, shape ``[..., M, K_s]``.
            All dimensions except K_s must match x exactly.
        y (Tile): RHS input, 2D or 3D ``[..., K, N]``.
        y_scale (Tile): Scale factors for y, shape ``[..., K_s, N]``.
            All dimensions except K_s must match y exactly.
        acc (Tile): Accumulator ``[..., M, N]``.

    Supported datatypes and scaling block sizes:

    +----------------------------+------------+---------+--------+
    | Input (x/y)                | Scale      | Acc/Out | B      |
    +============================+============+=========+========+
    | f8e4m3fn, f8e5m2           | f8e8m0fnu  | f32     | 32     |
    +----------------------------+------------+---------+--------+
    | f4e2m1fn                   | f8e8m0fnu  | f32     | 16, 32 |
    +----------------------------+------------+---------+--------+
    | f4e2m1fn                   | f8e4m3fn   | f32     | 16     |
    +----------------------------+------------+---------+--------+

    Batch dimensions of x and y are broadcast against each other (same as
    :func:`mma`). x_scale's batch dimension must match x's batch exactly,
    and y_scale's batch dimension must match y's batch exactly; both are
    then broadcast to the output batch shape.

    Returns:
        Tile:

    Example:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.ones((16, 64), ct.float8_e4m3fn)
            sx = ct.ones((16, 2), ct.float8_e8m0fnu)
            ty = ct.ones((64, 16), ct.float8_e4m3fn)
            sy = ct.ones((2, 16), ct.float8_e8m0fnu)
            acc = ct.zeros((16, 16), ct.float32)
            tz = ct.mma_scaled(tx, sx, ty, sy, acc)
    """


@function
def matmul(x, y, /) -> Tile:
    """Performs matrix multiply on the given tiles.

    Args:
        x (Tile): LHS of the matmul, 1D, 2D, or 3D.
        y (Tile): RHS of the matmul, 1D, 2D, or 3D.

    Supported input datatypes: [f16, bf16, f32, f64, tf32, f8e4m3fn, f8e5m2, i8, u8]

    If `x` and `y` have different dtype, they will first be promoted to common
    dtype. The result dtype is the same as the promoted input types.
    Shape of `x` and `y` will be broadcasted to up until the last two axes.

    Returns:
        Tile:

    Examples:

        2D x 2D.

        .. testcode::
            :template: kernel_wrapper.py

            x = ct.ones((2, 4), dtype=ct.float32)
            y = ct.ones((4, 2), dtype=ct.float32)
            print(f"{x @ y:.1f}")

        .. testoutput::

            [[4.0, 4.0], [4.0, 4.0]]

        1D x 1D (dot product).

        .. testcode::
            :template: kernel_wrapper.py

            x = ct.ones(4, dtype=ct.float32)
            y = ct.ones(4, dtype=ct.float32)
            print(f"{x @ y:.1f}")

        .. testoutput::

            4.0

        Batched: 3D x 2D with broadcast.

        .. testcode::
            :template: kernel_wrapper.py

            x = ct.ones((2, 2, 4), dtype=ct.float32)
            y = ct.ones((4, 2), dtype=ct.float32)
            print(f"{x @ y:.1f}")

        .. testoutput::

            [[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]
    """


# ======== Shape and Dtype ==============
@function
def expand_dims(x, /, axis) -> Tile:
    """Reshapes the tile by inserting a new axis of size 1 at given position.

    This can also be done via the NumPy-style syntax: `x[:, None]` or `x[np.newaxis, :]`

    Args:
        x (Tile): input tile.
        axis (const int): axis to expand the tile dimension.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(ct.expand_dims(tx, 0))
            print(tx[:, None])

        .. testoutput::

            [[0, 1, 2, 3]]
            [[0], [1], [2], [3]]
    """


@function
def cat(tiles, /, axis) -> Tile:
    """Concatenates two tiles along the `axis`.

    Args:
        tiles (tuple): a pair of tiles to concatenate.
        axis (const int): axis to concatenate the tiles.

    Returns:
        Tile:

    Notes:
        Due to power-of-two assumption on all tile shapes,
        the two input tiles must have the same shape.

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((2, 2), 3, dtype=ct.int32)
            ty = ct.full((2, 2), 7, dtype=ct.int32)
            print(ct.cat((tx, ty), 0))
            print(ct.cat((tx, ty), 1))

        .. testoutput::

            [[3, 3], [3, 3], [7, 7], [7, 7]]
            [[3, 3, 7, 7], [3, 3, 7, 7]]
    """


@function
def broadcast_to(x, /, shape) -> Tile:
    """Broadcasts a tile to the specified shape
    following |Numpy broadcasting rule|.

    Args:
        x (Tile): input tile.
        shape (tuple[const int,...]): target shape.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = ct.broadcast_to(tx, (2, 4))
            print(ty)

        .. testoutput::

            [[0, 1, 2, 3], [0, 1, 2, 3]]
    """


@function
def reshape(x, /, shape) -> Tile:
    """Reshapes a tile to the specified shape.

    One of the shape elements may be specified as -1 to indicate that the
    corresponding dimension is to be inferred automatically.

    For example, reshaping a ``(16, 2)`` tile to ``(8, -1)`` will
    produce a tile of shape ``(8, 4)``: as there are 32 elements in total,
    the second dimension will be computed as 32 divided by 8.

    Args:
        x (Tile): input tile.
        shape (tuple[const int,...]): target shape.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32)
            ty = ct.reshape(tx, (2, 4))
            print(ty)

        .. testoutput::

            [[0, 1, 2, 3], [4, 5, 6, 7]]
    """


@function
def permute(x, /, axes) -> Tile:
    """Permutes the axes of the input tile.

    Args:
        x (Tile): input tile.
        axes (tuple[const int,...]): the desired axes order.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            ty = ct.permute(tx, (2, 0, 1))
            print(ty)

        .. testoutput::

            [[[0, 2], [4, 6]], [[1, 3], [5, 7]]]
    """


@function
def transpose(x, /, axis0=None, axis1=None) -> Tile:
    """Transposes two axes of the input tile with at least 2 dimensions.

    For a 2-dimensional tile, the two axes are transposed if `axis0` and `axis1` are not specified.
    For tiles with more than 2 dimensions, `axis0` and `axis1` must be explicitly specified.

    Args:
        x (Tile): input tile.
        axis0 (const int): the first axis to transpose.
        axis1 (const int): the second axis to transpose.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32)
            tx = ct.reshape(tx, (2, 4))
            ty = ct.transpose(tx)
            print(ty)

        .. testoutput::

            [[0, 4], [1, 5], [2, 6], [3, 7]]
    """


@function
def astype(x, dtype, /) -> Tile:
    """Converts a tile to the specified data type.

    Args:
        x (Tile): input tile.
        dtype (DType): target data type.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = ct.astype(tx, ct.float32)
            print(f"{ty:.1f}")

        .. testoutput::

            [0.0, 1.0, 2.0, 3.0]
    """


@function
def bitcast(x, /, dtype) -> Tile:
    """Reinterpets tile as being of specified data type.

    Args:
        x (Tile): input tile.
        dtype (DType): target data type.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            ty = ct.bitcast(1.0, ct.uint32)
            print(f"{ty:#x}")

        .. testoutput::

            0x3f800000
    """


@function
def pack_to_bytes(x, /) -> Tile:
    """Flattens a tile and reinterprets its raw bytes as uint8 elements.

    The total number of bits of the input tile must be divisible by 8.

    Args:
        x (Tile): input tile.

    Returns:
        Tile: a 1D uint8 tile with ``total_elements * bit width // 8`` elements.

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full(1, 0x04030201, dtype=ct.int32)
            ty = ct.pack_to_bytes(tx)
            print(ty)

        .. testoutput::

            [1, 2, 3, 4]
    """


@function
def unpack_from_bytes(x, /, dtype) -> Tile:
    """Reinterprets a 1D uint8 byte tile as a 1D tile of the target data type.

    The inverse of :py:func:`pack_to_bytes`. The input must be a 1D tile of
    dtype uint8, and the total number of bits must be divisible by the
    target data type bit width.

    Args:
        x (Tile): a 1D tile of dtype uint8.
        dtype (DType): target data type.

    Returns:
        Tile: a 1D tile of ``dtype`` with ``num_bytes * 8 // bit width`` elements.

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.uint8) + 1
            ty = ct.unpack_from_bytes(tx, ct.int32)
            print(f"{ty:#x}")

        .. testoutput::

            [0x4030201]
    """


def _math_op_extra_block(f, indent):
    base = inspect.unwrap(f)
    sig = inspect.signature(base)
    extra = []
    for name in sig.parameters:
        if name == "rounding_mode":
            extra.append(
                f"{name} (RoundingMode): The rounding mode for the operation, only supported "
                "for float types, default is RoundingMode.RN when applicable."
            )
        elif name == "flush_to_zero":
            extra.append(
                f"{name} (const bool): If True, flushes subnormal inputs and results to "
                "sign-preserving zero, default is False."
            )
    return ("\n" + textwrap.indent("\n".join(extra), indent)) if extra else ""


# ======== Reduction ==============
def _doc_reduce_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    orig_doc = f.__doc__ or ""
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""\
Performs {op_name} reduction on tile along the `axis`.

Args:
    x (Tile): input tile.
    axis (None | const int | tuple[const int,...]): the axis for reduction.
        The default, `axis=None`, will reduce all of the elements.
        For `argmin` and `argmax`, tuple of axis is not supported.
    keepdims (const bool): If true, preserves the number of dimension
        from the input tile.{extra_block}

Returns:
    Tile:

""" + orig_doc

    return wrapped


@_doc_reduce_op
@function
def sum(x, /, axis=None, *, keepdims=False, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.sum(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 28

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.sum(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[6], [22]]

        Reduce axes (1, 2).

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            print("input:", tx)
            print("reduced:", ct.sum(tx, (1, 2)))

        .. testoutput::

            input: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            reduced: [6, 22]
    """
    pass


@_doc_reduce_op
@function
def max(x, /, axis=None, *, keepdims=False, flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.max(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 7

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.max(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[3], [7]]

        Reduce axes (1, 2).

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            print("input:", tx)
            print("reduced:", ct.max(tx, (1, 2)))

        .. testoutput::

            input: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            reduced: [3, 7]

    """
    pass


@_doc_reduce_op
@function
def min(x, /, axis=None, *, keepdims=False, flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.min(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 0

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.min(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[0], [4]]

        Reduce axes (1, 2).

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            print("input:", tx)
            print("reduced:", ct.min(tx, (1, 2)))

        .. testoutput::

            input: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            reduced: [0, 4]
    """
    pass


@_doc_reduce_op
@function
def prod(x, /, axis=None, *, keepdims=False, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.prod(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 0

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.prod(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[0], [840]]
    """


@_doc_reduce_op
@function
def argmax(x, /, axis=None, *, keepdims=False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.argmax(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 7

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.argmax(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[3], [3]]

        If there are ties, the smallest index is returned.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.zeros(4, dtype=ct.int32)
            tx = ct.cat((tx, tx + 1), axis=0)
            print("input:", tx)
            print("reduced:", ct.argmax(tx, 0))

        .. testoutput::

            input: [0, 0, 0, 0, 1, 1, 1, 1]
            reduced: 4

        Reduce over tuple of axes is not supported for argmax.
    """
    pass


@_doc_reduce_op
@function
def argmin(x, /, axis=None, *, keepdims=False) -> Tile:
    """
    Examples:

        Reduce all axes.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.argmin(tx, None))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: 0

        Reduce axis 1 and keepdims.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reduced:", ct.argmin(tx, 1, keepdims=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reduced: [[0], [0]]

        If there are ties, the smallest index is returned.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.zeros(4, dtype=ct.int32)
            tx = ct.cat((tx + 1, tx), axis=0)
            print("input:", tx)
            print("reduced:", ct.argmin(tx, 0))

        .. testoutput::

            input: [1, 1, 1, 1, 0, 0, 0, 0]
            reduced: 4

        Reduce over tuple of axes is not supported for argmin.
    """
    pass


@function
def reduce(x, /, axis, func, identity, *, keepdims=False):
    """
    Apply custom reduction function along axis.

    Args:
        x: input tile or a tuple of tiles to be reduced. If a tuple is provided, shapes
            of the tiles in the tuple must be broadcastable to a common shape.
        axis (int): an integer constant that specifies the axis to reduce along.
        func: function for combining two values. If `x` is a single tile, then the function
            must take two 0d tile arguments and return the combined 0d tile. For example,
            `lambda a, b: a + b` or `operator.add` can be used to implement the sum reduction.
            If `x` is a tuple of N tiles, then the function takes 2N tiles and returns a tuple
            of N combined tiles. The first N arguments correspond to one of the groups of values
            being combined, while the rest correspond to the other.
        identity: a constant scalar or a tuple of constant scalars that specifies the identity
            element of the `func`.
        keepdims (bool): True to keep the axis of size 1, False to remove the reduced axis.
            Default: False.

    Returns:
        Reduced tile, or tuple of reduced tiles, depending on the type of `x`.
    """


# ======== Scan ==============
def _doc_scan_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    orig_doc = f.__doc__ or ""
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""Performs {op_name} on tile along the `axis`.

    Args:
        x (Tile): input tile
        axis (const int): the axis for scan, default 0.
        reverse (const bool): if True, the scan is performed in the reverse direction.{extra_block}

    Returns:
        Tile:

    """ + orig_doc

    return wrapped


@_doc_scan_op
@function
def cumsum(x, /, axis=0, *, reverse=False, rounding_mode: Optional[RoundingMode] = None,
           flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        Scan along axis 1.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("cumsum:", ct.cumsum(tx, 1))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            cumsum: [[0, 1, 3, 6], [4, 9, 15, 22]]

        Reverse scan.

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(8, dtype=ct.int32).reshape((2, 4))
            print("input:", tx)
            print("reverse:", ct.cumsum(tx, 1, reverse=True))

        .. testoutput::

            input: [[0, 1, 2, 3], [4, 5, 6, 7]]
            reverse: [[6, 6, 5, 3], [22, 18, 13, 7]]
    """
    pass


@_doc_scan_op
@function
def cumprod(x, /, axis=0, *, reverse=False, rounding_mode: Optional[RoundingMode] = None,
            flush_to_zero: bool = False) -> Tile:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((2, 4), 2, dtype=ct.int32)
            print("input:", tx)
            print("cumprod:", ct.cumprod(tx, 1))

        .. testoutput::

            input: [[2, 2, 2, 2], [2, 2, 2, 2]]
            cumprod: [[2, 4, 8, 16], [2, 4, 8, 16]]
    """
    pass


@function
def scan(x, /, axis, func, identity, *, reverse=False):
    """
    Apply custom scan (inclusive prefix) function along axis.

    Args:
        x: input tile or a tuple of tiles to be scanned. If a tuple is provided, shapes
            of the tiles in the tuple must be broadcastable to a common shape.
        axis (int): an integer constant that specifies the axis to scan along.
        func: function for combining two values. If `x` is a single tile, then the function
            must take two 0d tile arguments and return the combined 0d tile. For example,
            `lambda a, b: a + b` or `operator.add` can be used to implement cumsum.
            If `x` is a tuple of N tiles, then the function takes 2N tiles and returns a tuple
            of N combined tiles. The first N arguments correspond to one of the groups of values
            being combined, while the rest correspond to the other.
        identity: a constant scalar or a tuple of constant scalars that specifies the identity
            element of the `func`.
        reverse (bool): if True, the scan is performed in the reverse direction along the axis.
            Default: False.

    Returns:
        Scanned tile, or tuple of scanned tiles, depending on the type of `x`.
    """
    pass


# ======== Math binary ==============
def _doc_binary_op(builtin_op):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        op_name = f.__name__
        extra_block = _math_op_extra_block(f, indent="            ")

        if builtin_op in ("min", "max"):
            builtin_example = f"{builtin_op}({{}}, {{}})"
        else:
            builtin_example = f"{{}} {builtin_op} {{}}"

        orig_doc = textwrap.dedent(f.__doc__ or "")

        wrapped.__doc__ = f"""\
Elementwise {op_name} on two tiles.

Can also use builtin operation `{builtin_example.format('x', 'y')}`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.{extra_block}

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

""" + orig_doc
        return wrapped
    return decorator


@_doc_binary_op('+')
@function
def add(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx + 10)

        .. testoutput::

            [10, 11, 12, 13]
    """
    pass


@_doc_binary_op('-')
@function
def sub(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx - 1)

        .. testoutput::

            [-1, 0, 1, 2]
    """
    pass


@_doc_binary_op('*')
@function
def mul(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
        flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx * 3)

        .. testoutput::

            [0, 3, 6, 9]
    """
    pass


@_doc_binary_op('/')
@function
def truediv(x, y, /, *, rounding_mode: Optional[RoundingMode] = None,
            flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = ct.full((4,), 2, dtype=ct.int32)
            print(f"{tx / ty:.1f}")

        .. testoutput::

            [0.0, 0.5, 1.0, 1.5]
    """
    pass


@function
def floordiv(x, y, /) -> TileOrScalar:
    """Elementwise floordiv on two tiles.

    Can also use builtin operation ``x // y``.

    Supports both integer and floating-point operands. For float inputs,
    the result is ``floor(x / y)`` as a float (e.g. ``5.5 // 2.2 == 2.0``).

    Args:
        x (Tile): LHS tile.
        y (Tile): RHS tile.

    The ``shape`` of ``x`` and ``y`` will be broadcasted and
    ``dtype`` promoted to common dtype.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 7, dtype=ct.int32)
            ty = ct.full((4,), 3, dtype=ct.int32)
            tz = tx // ty
            print(tz)

        .. testoutput::

            [2, 2, 2, 2]
    """
    pass


@_doc_binary_op('**')
@function
def pow(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 2.0, dtype=ct.float32)
            ty = ct.arange(4, dtype=ct.float32)
            print(f"{tx ** ty:.1f}")

        .. testoutput::

            [1.0, 2.0, 4.0, 8.0]
    """
    pass


@function
def atan2(x1, x2, /) -> TileOrScalar:
    """Elementwise atan2 of two tiles.

    Computes the element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    Args:
        x1 (Tile): Numerator tile (y-coordinate).
        x2 (Tile): Denominator tile (x-coordinate).

    The `shape` of `x1` and `x2` will be broadcasted and
    `dtype` promoted to common dtype.

    Returns:
        Tile: The angles in radians, in the range [-pi, pi].

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            y = ct.full((4,), 1.0, dtype=ct.float32)
            x = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.atan2(y, x):.4f}")

        .. testoutput::

            [1.5708, 1.5708, 1.5708, 1.5708]
    """


@_doc_binary_op('%')
@function
def mod(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx % 3)

        .. testoutput::

            [0, 1, 2, 0]
    """
    pass


@_doc_binary_op('&')
@function
def bitwise_and(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 7, dtype=ct.int32)
            ty = ct.full((4,), 3, dtype=ct.int32)
            print(tx & ty)

        .. testoutput::

            [3, 3, 3, 3]
    """
    pass


@_doc_binary_op('|')
@function
def bitwise_or(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 5, dtype=ct.int32)
            ty = ct.full((4,), 3, dtype=ct.int32)
            print(tx | ty)

        .. testoutput::

            [7, 7, 7, 7]
    """
    pass


@_doc_binary_op('^')
@function
def bitwise_xor(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 7, dtype=ct.int32)
            ty = ct.full((4,), 3, dtype=ct.int32)
            print(tx ^ ty)

        .. testoutput::

            [4, 4, 4, 4]
    """
    pass


@_doc_binary_op('<<')
@function
def bitwise_lshift(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx << 2)

        .. testoutput::

            [0, 4, 8, 12]
    """
    pass


@_doc_binary_op('>>')
@function
def bitwise_rshift(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 8, dtype=ct.int32)
            ty = ct.arange(4, dtype=ct.int32)
            print(tx >> ty)

        .. testoutput::

            [8, 4, 2, 1]
    """
    pass


@function
def bitwise_not(x, /) -> TileOrScalar:
    """Elementwise bitwise not on a tile.

    Can also use builtin operator `~x`.

    Args:
        x (Tile): input tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0, dtype=ct.int32)
            ty = ~tx
            print(ty)

        .. testoutput::

            [-1, -1, -1, -1]
    """

# TODO:  Do we support logical and, or, not?


@_doc_binary_op('min')
@function
def minimum(x, y, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = ct.full((4,), 2, dtype=ct.int32)
            print(min(tx, ty))

        .. testoutput::

            [0, 1, 2, 2]
    """
    pass


@_doc_binary_op('max')
@function
def maximum(x, y, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = ct.full((4,), 2, dtype=ct.int32)
            print(max(tx, ty))

        .. testoutput::

            [2, 2, 2, 3]
    """
    pass


@function(host=True)
def cdiv(x, y, /) -> TileOrScalar:
    """Computes ceil(x / y). Can be used on the host.

    Args:
        x (Tile): int tile.
        y (Tile): int tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: setup_only.py

            print(ct.cdiv(9, 4))
            print(ct.cdiv(8, 4))

        .. testoutput::

            3
            2
    """
    return (x - 1) // y + 1


# ======== Comparison ==============

def _doc_cmp_op(builtin_op):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)

        orig_doc = textwrap.dedent(f.__doc__ or "")

        wrapped.__doc__ = f"""\
Compare two tiles elementwise with `{builtin_op}`.

Can also use builtin operation `x {builtin_op} y`.

Args:
    x (Tile): LHS tile.
    y (Tile): RHS tile.

The `shape` of `x` and `y` will be broadcasted and
`dtype` promoted to common dtype.

Returns:
    Tile:

""" + orig_doc
        return wrapped
    return decorator


@_doc_cmp_op('>')
@function
def greater(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx > 2)

        .. testoutput::

            [0, 0, 0, 1]
    """
    pass


@_doc_cmp_op('>=')
@function
def greater_equal(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx >= 2)

        .. testoutput::

            [0, 0, 1, 1]
    """
    pass


@_doc_cmp_op('<')
@function
def less(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx < 2)

        .. testoutput::

            [1, 1, 0, 0]
    """
    pass


@_doc_cmp_op('<=')
@function
def less_equal(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx <= 2)

        .. testoutput::

            [1, 1, 1, 0]
    """
    pass


@_doc_cmp_op('==')
@function
def equal(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx == 2)

        .. testoutput::

            [0, 0, 1, 0]
    """
    pass


@_doc_cmp_op('!=')
@function
def not_equal(x, y, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            print(tx != 2)

        .. testoutput::

            [1, 1, 0, 1]
    """
    pass


# ======== Math unary ==============
def _doc_unary_op(f):

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)

    op_name = f.__name__
    orig_doc = f.__doc__ or ""
    extra_block = _math_op_extra_block(f, indent="        ")

    wrapped.__doc__ = f"""
    Perform `{op_name}` on a tile.

    Args:
        x (Tile):{extra_block}

    Returns:
        Tile:

    """ + orig_doc
    return wrapped


@_doc_unary_op
@function
def exp(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.exp(tx):.1f}")

        .. testoutput::

            [1.0, 1.0, 1.0, 1.0]
    """
    pass


@_doc_unary_op
@function
def exp2(x, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 3.0, dtype=ct.float32)
            print(f"{ct.exp2(tx):.1f}")

        .. testoutput::

            [8.0, 8.0, 8.0, 8.0]
    """
    pass


@_doc_unary_op
@function
def log(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 1.0, dtype=ct.float32)
            print(f"{ct.log(tx):.1f}")

        .. testoutput::

            [0.0, 0.0, 0.0, 0.0]
    """
    pass


@_doc_unary_op
@function
def log2(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 8.0, dtype=ct.float32)
            print(f"{ct.log2(tx):.1f}")

        .. testoutput::

            [3.0, 3.0, 3.0, 3.0]
    """
    pass


@_doc_unary_op
@function
def sqrt(x, /, *, rounding_mode: Optional[RoundingMode] = None,
         flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 4.0, dtype=ct.float32)
            print(f"{ct.sqrt(tx):.1f}")

        .. testoutput::

            [2.0, 2.0, 2.0, 2.0]
    """
    pass


@_doc_unary_op
@function
def rsqrt(x, /, *, flush_to_zero: bool = False) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 4.0, dtype=ct.float32)
            print(f"{ct.rsqrt(tx):.1f}")

        .. testoutput::

            [0.5, 0.5, 0.5, 0.5]
    """
    pass


@_doc_unary_op
@function
def sin(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.sin(tx):.1f}")

        .. testoutput::

            [0.0, 0.0, 0.0, 0.0]
    """
    pass


@_doc_unary_op
@function
def cos(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.cos(tx):.1f}")

        .. testoutput::

            [1.0, 1.0, 1.0, 1.0]
    """
    pass


@_doc_unary_op
@function
def tan(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.tan(tx):.1f}")

        .. testoutput::

            [0.0, 0.0, 0.0, 0.0]
    """
    pass


@_doc_unary_op
@function
def sinh(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.sinh(tx):.1f}")

        .. testoutput::

            [0.0, 0.0, 0.0, 0.0]
    """
    pass


@_doc_unary_op
@function
def cosh(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.cosh(tx):.1f}")

        .. testoutput::

            [1.0, 1.0, 1.0, 1.0]
    """
    pass


@function
def tanh(x, /, *, rounding_mode: Optional[RoundingMode] = None) -> TileOrScalar:
    """
    Perform `tanh` on a tile.

    Args:
        x (Tile):
        rounding_mode (RoundingMode): Supported values:

            - ``RoundingMode.FULL``
            - ``RoundingMode.APPROX`` (since CTK 13.2)

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 0.0, dtype=ct.float32)
            print(f"{ct.tanh(tx):.1f}")

        .. testoutput::

            [0.0, 0.0, 0.0, 0.0]
    """


@_doc_unary_op
@function
def floor(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 3.7, dtype=ct.float32)
            print(f"{ct.floor(tx):.1f}")

        .. testoutput::

            [3.0, 3.0, 3.0, 3.0]
    """
    pass


@_doc_unary_op
@function
def ceil(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 3.2, dtype=ct.float32)
            print(f"{ct.ceil(tx):.1f}")

        .. testoutput::

            [4.0, 4.0, 4.0, 4.0]
    """
    pass


@_doc_unary_op
@function
def abs(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32) - 2
            print("input:", tx)
            print("abs:", ct.abs(tx))

        .. testoutput::

            input: [-2, -1, 0, 1]
            abs: [2, 1, 0, 1]
    """
    pass


@function
def negative(x, /) -> TileOrScalar:
    """Same as `-x`.

    Args:
        x (Tile): input tile.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.arange(4, dtype=ct.int32)
            ty = -tx
            print(ty)

        .. testoutput::

            [0, -1, -2, -3]
    """


@_doc_unary_op
@function
def isnan(x, /) -> TileOrScalar:
    """
    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tx = ct.full((4,), 1.0, dtype=ct.float32)
            print(ct.isnan(tx))

        .. testoutput::

            [0, 0, 0, 0]
    """
    pass


# ======== Select ==============

@function
def where(cond, x, y, /) -> Tile:
    """Returns elements chosen from x or y depending on condition.

    Args:
        cond (Tile): Boolean tile of shape `S`.
        x (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is True.
        y (Tile): Tile of shape `S` and dtype `T`, selected if `cond` is False.

    Returns:
        Tile:

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            cond = ct.arange(4, dtype=ct.int32) >= 2
            x_true = ct.full((4,), 1, dtype=ct.int32)
            x_false = ct.full((4,), -1, dtype=ct.int32)
            y = ct.where(cond, x_true, x_false)
            print(y)

        .. testoutput::

            [-1, -1, 1, 1]
    """


@function
def extract(x, /, index, shape) -> Tile:
    """Extracts a smaller tile from input tile.

    Partition the input tile into a grid with subtile shape
    and return a tile given the index into the grid. Similar
    to :py:func:`load` but performed on a tile.

    Args:
        x (Tile): input tile.
        index (Shape): An index in the sub |tile space|.
        shape (Shape): The shape of the extracted tile.

    Returns:
        Tile:

    Examples:

        1D tile.

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.arange(8, dtype=ct.int32)
            sub = ct.extract(tile, (0,), shape=(4,))
            print(f'(0,): {sub}')
            sub = ct.extract(tile, (1,), shape=(4,))
            print(f'(1,): {sub}')

        .. testoutput::

            (0,): [0, 1, 2, 3]
            (1,): [4, 5, 6, 7]

        2D tile.

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.arange(16, dtype=ct.int32).reshape((4, 4))
            sub = ct.extract(tile, (0, 0), shape=(2, 2))
            print(f'(0, 0): {sub}')
            sub = ct.extract(tile, (0, 1), shape=(2, 2))
            print(f'(0, 1): {sub}')

        .. testoutput::

            (0, 0): [[0, 1], [4, 5]]
            (0, 1): [[2, 3], [6, 7]]
    """


# ============ Utility =================

@function
def printf(format, *args) -> None:
    """Print the values at runtime from the device

    Args:
        format (str): a c-printf style format string
            in the form of ``%[flags][width][.precision][length]specifier``,
            where specifier is limited to integer and float for now, i.e.
            ``[diuoxXeEfFgGaA]``

        *args (tuple[Tile, ...]):
            Only tile input is supported.

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.full((), 42, dtype=ct.int32)
            ct.printf("value: %d\\n", tile)

        .. testoutput::

            value: 42

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.
    """


@function
def print(*args, sep: str = ' ', end: str = '\n') -> None:
    """Print values at runtime from the device using Python-style syntax.

    Supports Python f-strings and positional arguments similar to Python's
    built-in ``print()`` function.

    Args:
        *args: Values to print. Each argument can be:
            - A string literal or f-string
            - A tile value (format inferred from dtype: int→``%d``, float→``%f``)
        sep (str): Separator inserted between arguments (default: ``' '``)
        end (str): String appended after the last argument (default: ``'\\n'``)

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.full((), 42, dtype=ct.int32)
            ct.print(f"value={tile}")
            print(f"value={tile}")

        .. testoutput::

            value=42
            value=42

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purposes.

        F-string expressions must evaluate to tile values. Constant compile-time
        values are supported as string-formatted segments.
    """


@function
def assert_(cond, /, message=None) -> None:
    """Assert that all elements of the given tile are True.

    Args:
        cond (Tile): Boolean tile.
        message (str): Message to print if condition is false.

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.


    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            tile = ct.full((4,), 5, dtype=ct.int32)
            ct.assert_(tile > 0)
            print("assertion passed")

        .. testoutput::

            assertion passed
    """


@function
def static_eval(expr, /):
    """Evaluates the given Python expression at compile time.

    The expression is evaluated using standard Python semantics, not Tile
    semantics. It can reference global variables and local variables from
    the surrounding tile function.

    If a referenced variable is a compile-time constant value, it will be represented
    with a corresponding Python object of that value. For example, a constant integer 3 will
    be passed as a plain ``int`` object of value 3.

    If a referenced variable has dynamic value, such as a tile or an array,
    it will be passed as a proxy object that allows querying compile-time attributes.
    For example, if ``x`` is a tile, one can use ``x.shape`` to obtain the tile shape
    as a tuple of integers.

    The expression is allowed to return a proxy object for a dynamic value.
    This can be used to select one of multiple dynamic values based on a compile-time
    condition. For example, if ``N`` is an integer constant and ``x``, ``y`` are dynamic
    tiles, then one can write ``x_or_y = ct.static_eval(x if N % 2 == 0 else y)`` to select
    either ``x`` or ``y`` at compile tile, depending on the parity of ``N``.

    However, the expression is not allowed to perform any run-time operations. For example,
    if ``x`` refers to a dynamic tile, then ``ct.static_eval(x + 1)`` will raise an error.

    The expression must not assign to local variables (e.g., via the walrus operator ``:=``).

    Despite being declared as a function, `static_eval()` is treated like a keyword:
    it skips the translation of the surrounded expression according to the Tile semantics.
    Moreover, the expression is allowed to use the full Python syntax, unlike the rest
    of the Tile code, which is limited to a stricter subset of the language.
    """


@function
def static_assert(condition, message=None, /):
    """Asserts that a condition is true at compile time.

    First, `condition` is evaluated using the same rules as :py:func:`static_eval`:
    it can reference global and local variables, and use the full
    Python syntax, but must not perform any run-time operations.

    The `condition` must evaluate to a compile-time constant boolean.
    If it evaluates to ``True``, compilation continues normally,
    and the `message` expression is not evaluated.

    If `condition` evaluates to ``False``, then the `message` expression is evaluated using
    the :py:func:`static_eval` semantics. If the result of the evaluation is None,
    it is replaced with an empty string. Otherwise, it is converted to a string using
    the builtin ``str()`` function. Then, a :py:class:`TileStaticAssertionError` is raised
    with the evaluated message string.

    Because `message` is evaluated using the :py:func:`static_eval` semantics,
    it can include useful debug information about local variables, for example:

    .. testcode::
        :template: kernel_wrapper.py

        x = ct.ones((4,), dtype=ct.int32)
        y = ct.ones((4,), dtype=ct.float32)
        ct.static_assert(x.dtype == y.dtype,
                         f"Expected {x} and {y} to have same dtype.")

    .. testoutput::
        :options: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL

        Traceback (most recent call last):
            ...
        TileStaticAssertionError: ...Expected ... and ... to have same dtype.

    Since the message is automatically converted to a string, one can use any object
    in its place, for example:

    .. testcode::
        :template: kernel_wrapper.py

        x = ct.ones((4,), dtype=ct.int32)
        ct.static_assert(x.dtype == ct.float32, x)

    .. testoutput::
        :options: +ELLIPSIS, +IGNORE_EXCEPTION_DETAIL

        Traceback (most recent call last):
            ...
        TileStaticAssertionError: Static assertion failed: <tile[int32, (4,)]>

    Despite being declared as a function, `static_assert()` is treated like a keyword:
    it skips the translation of the surrounded expressions according to the Tile semantics.
    Moreover, the expressions are allowed to use the full Python syntax, unlike the rest
    of the Tile code, which is limited to a stricter subset of the language.
    """


@function
def static_iter(iterable):
    """Iterates at compile time.

    Can only be used as the iterable of a `for` loop::

        for ... in ct.static_iter(...):
            ...

    The surrounded expression is evaluated using the same rules as :py:func:`static_eval`:
    it can reference global and local variables, and use the full Python syntax,
    but must not perform any run-time operations.

    The expression must return a Python iterable, whose length must not exceed some
    pre-defined number of iterations (currently, 1000). Before any further processing is done,
    the contents of the iterable are saved to a temporary list, and each item is checked
    to be valid, as if it were a result of a :py:func:`static_eval` expression
    (i.e., it must be a supported compile-time constant value or a proxy object
    for a dynamic value such as a tile).

    Finally, for each item of the iterable, the loop body is inlined, with the induction variable(s)
    bound to the item. The `break`, `continue`, and `return` statements are not allowed
    inside a `static_iter` loop.

    .. testcode::
        :template: kernel_wrapper.py

        tile = ct.zeros(4, dtype=ct.int32)
        size = 4

        states = ()
        for i in ct.static_iter(range(size)):
            states += (tile + i,)
        print(states)

        new_states = ()
        for i in ct.static_iter(range(size)):
            new_states += (states[(i + 1) % size] + states[i], )
        print(new_states)

    .. testoutput::

        ([0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3])
        ([1, 1, 1, 1], [3, 3, 3, 3], [5, 5, 5, 5], [3, 3, 3, 3])
    """


# ==== Private stubs ====


@function
def _m_array_slice(array, axis, start, stop): ...
# Array.slice(axis, start, stop)


@function
def _m_tile_item(tile): ...
# Tile.item()


def _inherit_kwdefaults(source):
    def decorator(f):
        target = getattr(f, '__wrapped__', f)
        source_fn = getattr(source, '__wrapped__', source)
        if kwdefaults := source_fn.__kwdefaults__:
            kwonly = {name for name, p in inspect.signature(target).parameters.items()
                      if p.kind == inspect.Parameter.KEYWORD_ONLY}
            assert kwdefaults.keys() <= kwonly
            target.__kwdefaults__ = dict(kwdefaults)
        return f
    return decorator


@_inherit_kwdefaults(Array.tiled_view)
@function
def _m_array_tiled_view(array, tile_shape, *, padding_mode): ...
# Array.tiled_view(shape, padding_mode=padding_mode)


@_inherit_kwdefaults(TiledView.load)
@function
def _m_tiled_view_load(tiled_view, index, *, latency, allow_tma): ...
# TiledView.load(index, latency=latency, allow_tma=allow_tma)


@_inherit_kwdefaults(TiledView.store)
@function
def _m_tiled_view_store(tiled_view, index, tile, *, latency, allow_tma): ...
# TiledView.store(index, tile, latency=latency, allow_tma=allow_tma)


@function
def _m_array_get_raw_memory(array: Array) -> RawArrayMemory: ...  # Array.get_raw_memory()


@function
def _m_raw_array_memory_load_offset(
        raw_array_memory: RawArrayMemory, offset: TileOrScalar, /, *,
        mask: Optional[Tile] = None,
        padding_value: TileOrScalar = 0,
        latency: Optional[int] = None) -> Tile: ...  # RawArrayMemory.load_offset()


@function
def _m_raw_array_memory_store_offset(
        raw_array_memory: RawArrayMemory, offset: TileOrScalar, value: TileOrScalar, /, *,
        mask: Optional[Tile] = None,
        latency: Optional[int] = None) -> None: ...  # RawArrayMemory.store_offset()
