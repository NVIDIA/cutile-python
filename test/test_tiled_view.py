# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable, NamedTuple, Set
import pytest
import torch
from torch.testing import make_tensor
from unittest.mock import patch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._compile import compile_tile
from cuda.tile._exception import TileTypeError, TileUnsupportedFeatureError
from cuda.tile.compilation import CallingConvention, KernelSignature
from cuda.tile._ir.cast_ops import _is_implicit_cast_ok
from cuda.tile._ir.typing_support import to_dtype
from conftest import arithmetic_dtypes, dtype_id, requires_tileiras
from util import (
    assert_equal, AtomicOp, int_32_64_dtypes, int_float_32_64_dtypes,
    is_hopper_or_newer, raises_if, ref_atomic_arith, ref_atomic_bitwise
)

ConstInt = ct.Constant[int]


def check_tiled_view_properties(tiled_view, dtype, tile_shape):
    tv_dtype, tv_tile_shape = tiled_view.dtype, tiled_view.tile_shape
    ct.static_assert(tv_dtype == dtype)
    ct.static_assert(tv_tile_shape == tile_shape)


@pytest.mark.parametrize("shape", [64, (225,)])
@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("allow_tma", [False, True])
def test_tiled_view_copy_1d(shape, tile_size, dtype, allow_tma):
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        bid = ct.bid(0)
        tv_x = x.tiled_view(TILE)
        check_tiled_view_properties(tv_x, x.dtype, (TILE,))
        tv_y = y.tiled_view(TILE)
        tv_y.store(bid, tv_x.load(bid, allow_tma=allow_tma), allow_tma=allow_tma)

    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    shape = shape[0] if isinstance(shape, tuple) else shape
    grid = (ct.cdiv(shape, tile_size),)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile_size))
    assert_equal(y, x)


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("shape", [(128, 256), (192, 134)])
@pytest.mark.parametrize("tile_size", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32], ids=dtype_id)
def test_tiled_view_copy_2d(shape, tile_size, dtype, noncontiguous):

    @ct.kernel
    def kernel(x, y, n, TILE_M: ConstInt, TILE_N: ConstInt):
        bidm = ct.bid(0)
        bidn = ct.bid(1)
        tv_x = x.tiled_view((TILE_M, TILE_N))
        check_tiled_view_properties(tv_x, x.dtype, (TILE_M, TILE_N))
        tv_y = y.tiled_view((TILE_M, TILE_N))
        tv_y.store((bidm, bidn), tv_x.load((bidm, bidn)))
        tv_n = n.tiled_view(())
        check_tiled_view_properties(tv_n, n.dtype, ())
        if bidm == 0 and bidn == 0:
            nt1, nt2 = tv_x.num_tiles(0), tv_x.num_tiles(1)
            tv_n.store(0, nt1)
            tv_n.store(1, nt2)

    x = make_tensor(shape, dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    y = torch.zeros_like(x)
    n = torch.zeros(len(shape), dtype=torch.int32, device='cuda')
    ref_n = torch.tensor([ct.cdiv(shape[0], tile_size[0]), ct.cdiv(shape[1], tile_size[1])],
                         dtype=torch.int32,
                         device='cuda')

    grid = (ct.cdiv(shape[0], tile_size[0]), ct.cdiv(shape[1], tile_size[1]))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, n, tile_size[0], tile_size[1]))
    assert_equal(y, x)
    assert_equal(n, ref_n)


_padding_mode_to_val = {
    ct.PaddingMode.ZERO: 0.0,
    ct.PaddingMode.NEG_ZERO: -0.0,
    ct.PaddingMode.NAN: math.nan,
    ct.PaddingMode.POS_INF: math.inf,
    ct.PaddingMode.NEG_INF: -math.inf,
}


@pytest.mark.parametrize("padding_mode", [
    ct.PaddingMode.ZERO,
    ct.PaddingMode.NEG_ZERO,
    ct.PaddingMode.NAN,
    ct.PaddingMode.POS_INF,
    ct.PaddingMode.NEG_INF
], ids=str)
def test_tiled_view_padding_mode(padding_mode):
    @ct.kernel
    def kernel(x, z, TILE: ConstInt):
        tv = x.tiled_view(TILE, padding_mode=padding_mode)
        tile = tv.load(1)
        ct.store(z, 0, tile=tile)

    x = make_tensor((100,), dtype=torch.float32, device='cuda')
    z = torch.zeros(1, dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, z, 128))

    if padding_mode == ct.PaddingMode.NAN:
        assert math.isnan(z.item())
    else:
        assert z.item() == _padding_mode_to_val[padding_mode]


@pytest.mark.parametrize("tile_size", [(1, 2), (1, 2, 3), (1, 2, 3, 4)])
def test_tiled_view_rank_mismatch(tile_size):
    @ct.kernel
    def kernel(x):
        x.tiled_view(tile_size)

    x = torch.zeros(16, dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError, match=f"Expected shape length to be 1, got {len(tile_size)}"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_store_tile_shape_mismatch():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        wrong_tile = ct.load(x, 0, (TILE * 2,))
        y.tiled_view(TILE).store(0, wrong_tile)

    x = torch.zeros(16, dtype=torch.float32, device='cuda')
    y = torch.zeros(16, dtype=torch.float32, device='cuda')
    match = r"Tile shape \(8,\) is not broadcastable to the tiled view's tile shape \(4,\)"
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4))


@pytest.mark.parametrize("src_shape,dst_shape", [
    ((),       (16, 16)),
    ((1, 16),  (128, 16)),
    ((16, 1),  (16, 64)),
    ((1, 1),   (32, 16)),
])
def test_tiled_view_store_broadcast(src_shape, dst_shape):
    @ct.kernel
    def kernel(x, y):
        tile = x.tiled_view(src_shape).load((0, 0))
        y.tiled_view(dst_shape).store((0, 0), tile)

    x_shape = src_shape if len(src_shape) > 0 else (1, 1)
    x = make_tensor(x_shape, dtype=torch.float32, device='cuda')
    y = torch.zeros(dst_shape, dtype=torch.float32, device='cuda')
    ref = torch.broadcast_to(x, dst_shape)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


@pytest.mark.parametrize("use_x", [True, False])
def test_tiled_view_ifelse_result(use_x):
    @ct.kernel
    def kernel(x, y, z, TILE: ConstInt, USE_X: ct.Constant[bool]):
        tv = x.tiled_view(TILE) if USE_X else y.tiled_view(TILE)
        for i in range(tv.num_tiles(0)):
            z.tiled_view(TILE).store(i, tv.load(i))

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = make_tensor((128,), dtype=torch.float32, device='cuda')
    z = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z, 64, use_x))
    assert_equal(z, x if use_x else y)


def test_tiled_view_loop_carried():
    @ct.kernel
    def kernel(x, y, z, TILE: ConstInt):
        tv = x.tiled_view(TILE)
        tv_z = z.tiled_view(TILE)
        for i in range(tv_z.num_tiles(0)):
            tv_z.store(i, tv.load(0))
            tv = y.tiled_view(TILE)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = make_tensor((128,), dtype=torch.float32, device='cuda')
    z = torch.zeros((256,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z, 128))
    ref_z = torch.cat((x, y))
    assert_equal(z, ref_z)


def test_tiled_view_ifelse_type_mismatch():
    @ct.kernel
    def kernel(x, cond: bool, TILE_A: ConstInt, TILE_B: ConstInt):
        if cond:
            tv = x.tiled_view(TILE_A)
        else:
            tv = x.tiled_view(TILE_B)
        tv.store(0, ct.full(TILE_A, 1.0, ct.float32))

    x = torch.zeros(128, dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError, match="depends on path taken"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, True, 64, 32))


def test_tiled_view_helper_func():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        def get_view(arr, tile_size):
            return arr.tiled_view(tile_size)

        def copy_tile(tv_src, tv_dst, i):
            tv_dst.store(i, tv_src.load(i))

        tv_x = get_view(x, TILE)
        tv_y = get_view(y, TILE)
        for i in range(tv_x.num_tiles(0)):
            copy_tile(tv_x, tv_y, i)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 64))
    assert_equal(y, x)


def test_tiled_view_closure():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        tv_x = x.tiled_view(TILE)

        def make_closure():
            tv_y = y.tiled_view(TILE)

            def copy(i):
                tv_y.store(i, tv_x.load(i))

            return copy

        func = make_closure()
        for i in range(tv_x.num_tiles(0)):
            func(i)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 64))
    assert_equal(y, x)


# ==================== traversal_steps ====================

@requires_tileiras(BytecodeVersion.V_13_3)
def test_tiled_view_traversal_steps_parity():
    """traversal_steps == tile_shape → same result as no traversal_steps."""
    @ct.kernel
    def kernel_default(x, y, TILE: ConstInt):
        tv_x = x.tiled_view(TILE)
        tv_y = y.tiled_view(TILE)
        for i in range(tv_x.num_tiles(0)):
            tv_y.store(i, tv_x.load(i))

    @ct.kernel
    def kernel_explicit(x, y, TILE: ConstInt):
        tv_x = x.tiled_view(TILE, traversal_steps=TILE)
        tv_y = y.tiled_view(TILE, traversal_steps=TILE)
        for i in range(tv_x.num_tiles(0)):
            tv_y.store(i, tv_x.load(i))

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y_default = torch.zeros_like(x)
    y_explicit = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel_default, (x, y_default, 64))
    ct.launch(torch.cuda.current_stream(), (1,), kernel_explicit, (x, y_explicit, 64))
    assert_equal(y_default, x)
    assert_equal(y_explicit, x)


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("tile_size,step,n", [
    (4, 2, 8),   # traversal_steps < tile_shape: overlapping tiles
    (4, 8, 16),  # traversal_steps > tile_shape: strided tiles with gaps
    (4, 3, 12),  # traversal_steps is not a power of two
], ids=["step_lt_tile", "step_gt_tile", "step_non_power_of_two"])
def test_tiled_view_traversal_steps_sliding_window(tile_size, step, n, dtype):
    @ct.kernel
    def kernel(x, out, TILE: ConstInt, STEP: ConstInt):
        tv = x.tiled_view(TILE, traversal_steps=STEP)
        tv_out = out.tiled_view(TILE, traversal_steps=STEP)
        for i in range(tv.num_tiles(0)):
            tv_out.store(i, tv.load(i))

    x = make_tensor(n, dtype=dtype, device='cuda')
    out = torch.zeros(n, dtype=dtype, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, out, tile_size, step))
    ref = torch.zeros(n, dtype=dtype, device='cuda')
    for start in range(0, n, step):
        ref[start:start + tile_size] = x[start:start + tile_size]
    assert_equal(out, ref)


@requires_tileiras(BytecodeVersion.V_13_3)
def test_tiled_view_2d_conv_no_padding():
    """2D box-filter using tiled_view as a sliding window (traversal_steps < tile_shape).
    No padding on top or left: window starts at (0, 0) and only covers valid positions.
    Each output element is the sum of the corresponding (KH, KW) input patch."""
    H, W = 6, 6
    KH, KW = 2, 2
    SH, SW = 1, 1

    @ct.kernel
    def kernel(x, out, KH: ConstInt, KW: ConstInt, SH: ConstInt, SW: ConstInt,
               OUT_H: ConstInt, OUT_W: ConstInt):
        tv = x.tiled_view((KH, KW), traversal_steps=(SH, SW))
        out_tv = out.tiled_view(())
        for i in range(OUT_H):
            for j in range(OUT_W):
                tile = tv.load((i, j))
                out_tv.store(i * OUT_W + j, ct.sum(tile))

    x = make_tensor((H, W), dtype=torch.int32, device='cuda', low=0, high=10)
    out_h = (H - KH) // SH + 1
    out_w = (W - KW) // SW + 1
    out = torch.zeros(out_h * out_w, dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel,
              (x, out, KH, KW, SH, SW, out_h, out_w))
    ref = x.unfold(0, KH, SH).unfold(1, KW, SW).sum(dim=(-2, -1)).flatten().to(torch.int32)
    assert_equal(out, ref)


@requires_tileiras(BytecodeVersion.V_13_3)
def test_tiled_view_traversal_steps_num_tiles():
    """num_tiles with traversal_steps returns correct count."""
    @ct.kernel
    def kernel(x, out, TILE: ConstInt, STEP: ConstInt):
        tv = x.tiled_view(TILE, traversal_steps=STEP)
        n = tv.num_tiles(0)
        out_tv = out.tiled_view(1)
        out_tv.store(0, n)

    N = 16
    TILE = 4
    STEP = 2
    x = torch.zeros(N, dtype=torch.float32, device='cuda')
    out = torch.zeros(1, dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, out, TILE, STEP))
    assert out[0].item() == ct.cdiv(N, STEP)


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("step_h,step_w", [
    (2, 3),  # broadcasted tile_shape (1,1) != (2,3) → StridedView
    (1, 1),  # broadcasted tile_shape (1,1) == (1,1) → PartitionView
], ids=["strided_view", "partition_view"])
def test_tiled_view_0d_tile_with_traversal_steps(step_h, step_w):
    H, W = 4, 6
    NUM_H = H // step_h
    NUM_W = W // step_w

    @ct.kernel
    def kernel(x, out, STEP_H: ConstInt, STEP_W: ConstInt,
               NUM_H: ConstInt, NUM_W: ConstInt):
        tv = x.tiled_view((), traversal_steps=(STEP_H, STEP_W))
        out_tv = out.tiled_view(())
        for i in range(NUM_H):
            for j in range(NUM_W):
                out_tv.store(i * NUM_W + j, tv.load((i, j)))

    x = torch.arange(H * W, dtype=torch.float32, device='cuda').reshape(H, W)
    out = torch.zeros(NUM_H * NUM_W, dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel,
              (x, out, step_h, step_w, NUM_H, NUM_W))
    assert_equal(out, x[::step_h, ::step_w].flatten().to(torch.float32))


@pytest.mark.parametrize("array_shape,tile_shape,traversal_steps", [

    ((16,),  4, 2),
    ((16,),  4, 4),
    ((4, 6), (), (2, 3)),
])
def test_tiled_view_traversal_steps_version_error(array_shape, tile_shape, traversal_steps):
    @ct.kernel
    def kernel(x):
        x.tiled_view(tile_shape, traversal_steps=traversal_steps)

    x = torch.zeros(array_shape, dtype=torch.float32, device='cuda')
    cconv = CallingConvention.cutile_python_v1()
    sig = KernelSignature.from_kernel_args(kernel, (x,), cconv)
    with patch('cuda.tile._compile._get_max_supported_bytecode_version',
               return_value=BytecodeVersion.V_13_2):
        with pytest.raises(TileUnsupportedFeatureError,
                           match=r"traversal_steps requires tileiras 13\.3"):
            compile_tile(kernel._annotated_function, [sig])


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("array_shape,tile_shape,traversal_steps", [
    ((16,),      4,      (2, 2)),  # 1D array, traversal_steps rank 2
    ((16, 32),   (4, 4), (2,)),    # 2D array, traversal_steps rank 1
    ((16, 32),   (),     ()),      # 2D array, 0-d tile and 0-d traversal_steps
], ids=["1d_array_2d_steps", "2d_array_1d_steps", "0d_tile_0d_steps"])
def test_tiled_view_traversal_steps_rank_mismatch(array_shape, tile_shape, traversal_steps):
    @ct.kernel
    def kernel(x):
        x.tiled_view(tile_shape, traversal_steps=traversal_steps)

    x = torch.zeros(array_shape, dtype=torch.float32, device='cuda')
    ndim = len(array_shape)
    with pytest.raises(TileTypeError,
                       match=f"Expected traversal_steps length to be {ndim},"
                             f" got {len(traversal_steps)}"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("traversal_steps", [-1, 0, (-1, 4), (4, -1), (0, 4), (4, 0)],
                         ids=["neg_1d", "zero_1d",
                              "neg_first_2d", "neg_second_2d",
                              "zero_first_2d", "zero_second_2d"])
def test_tiled_view_non_positive_traversal_steps(traversal_steps):
    is_2d = isinstance(traversal_steps, tuple)
    array_shape = (16, 32) if is_2d else (16,)
    tile_shape = (4, 4) if is_2d else 4

    @ct.kernel
    def kernel(x):
        x.tiled_view(tile_shape, traversal_steps=traversal_steps)

    x = torch.zeros(array_shape, dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError, match="of traversal_steps .* is not positive"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("array_shape,tile_shape,traversal_steps,expected_steps", [
    ((64,),     64,       None, (64,)),    # 1D, default: equals tile_shape
    ((64, 128), (64, 128), None, (64, 128)),  # 2D, default: equals tile_shape
    ((4, 6),    (),       None, (1, 1)),   # 0-d tile: broadcasted to (1, 1)
    ((64,),     64,       64,  (64,)),     # explicit steps == tile_shape
    pytest.param((64,), 4, 2, (2,),
                 marks=requires_tileiras(BytecodeVersion.V_13_3)),  # explicit steps != tile_shape
], ids=["1d_default", "2d_default", "0d_tile", "explicit_equal", "explicit_different"])
def test_tiled_view_traversal_steps_property(array_shape, tile_shape, traversal_steps,
                                             expected_steps):
    @ct.kernel
    def kernel(x):
        tv = x.tiled_view(tile_shape, traversal_steps=traversal_steps)
        tv_traversal_steps = tv.traversal_steps
        ct.static_assert(tv_traversal_steps == expected_steps)

    x = torch.zeros(array_shape, dtype=torch.float32, device='cuda')
    grid = (1,) * len(array_shape)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x,))


class AtomicConfig(NamedTuple):
    get_tv_method: Callable
    torch_op: Callable
    supported_dtypes: Set


def add_method(tv): return tv.atomic_store_add
def and_method(tv): return tv.atomic_store_and
def max_method(tv): return tv.atomic_store_max
def min_method(tv): return tv.atomic_store_min
def or_method(tv): return tv.atomic_store_or
def xor_method(tv): return tv.atomic_store_xor


tv_atomic_configs = {
    AtomicOp.ADD: AtomicConfig(add_method, lambda x, y: x + y,
                               set(int_float_32_64_dtypes + [torch.float16, torch.bfloat16])),
    AtomicOp.MAX: AtomicConfig(max_method, torch.maximum,
                               set(int_32_64_dtypes)),
    AtomicOp.MIN: AtomicConfig(min_method, torch.minimum,
                               set(int_32_64_dtypes)),
    AtomicOp.AND: AtomicConfig(and_method, lambda x, y: x & y,
                               set(int_32_64_dtypes)),
    AtomicOp.OR: AtomicConfig(or_method, lambda x, y: x | y,
                              set(int_32_64_dtypes)),
    AtomicOp.XOR: AtomicConfig(xor_method, lambda x, y: x ^ y,
                               set(int_32_64_dtypes)),
}


def make_tv_atomic_kernel(get_tv_method, traversal_steps):
    @ct.kernel
    def kernel(x, y, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int]):
        bidm = ct.bid(0)
        bidn = ct.bid(1)
        tv_x = x.tiled_view((TILE_M, TILE_N), traversal_steps=traversal_steps)
        tv_y = y.tiled_view((TILE_M, TILE_N), traversal_steps=traversal_steps)
        update = tv_y.load((bidm, bidn))
        get_tv_method(tv_x)((bidm, bidn), update)
    return kernel


def make_ref_tv_atomic(ref_fn, torch_op, tile_size, traversal_steps):
    def ref_tv_atomic(x, y):
        for r0 in range(0, x.shape[0], traversal_steps[0]):
            for c0 in range(0, x.shape[1], traversal_steps[1]):
                r = slice(r0, r0 + tile_size[0])
                c = slice(c0, c0 + tile_size[1])
                ref_x, _ = ref_fn(x[r, c], y[r, c], torch_op)
                x[r, c] = ref_x
    return ref_tv_atomic


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("atomic_op", [AtomicOp.ADD, AtomicOp.MAX, AtomicOp.MIN,
                                       AtomicOp.AND, AtomicOp.OR, AtomicOp.XOR])
@pytest.mark.parametrize("x_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
def test_tiled_view_atomic(atomic_op, x_dtype, y_dtype):
    shape = (200, 256)
    tile_size = (128, 128)
    get_tv_method, torch_op, supported_dtypes = tv_atomic_configs[atomic_op]
    x = make_tensor(shape, dtype=x_dtype, device='cuda')
    y = make_tensor(shape, dtype=y_dtype, device='cuda')
    ref_x = x.clone()
    kernel = make_tv_atomic_kernel(get_tv_method, None)

    if x_dtype not in supported_dtypes:
        should_raise = True
        err_match = "Unsupported tiled view dtype"
        ref_fn = None
    elif atomic_op.is_bitwise():
        should_raise = x_dtype != y_dtype
        err_match = "to exactly match the target dtype"
        ref_fn = make_ref_tv_atomic(ref_atomic_bitwise, torch_op, tile_size, tile_size)
    else:
        if x_dtype == torch.bfloat16 and not is_hopper_or_newer():
            pytest.skip("bf16 is only supported on hopper or newer")

        should_raise = not _is_implicit_cast_ok(to_dtype(y_dtype), to_dtype(x_dtype))
        err_match = "cannot implicitly cast"
        ref_fn = make_ref_tv_atomic(ref_atomic_arith, torch_op, tile_size, tile_size)

    with raises_if(should_raise, TileTypeError, match=err_match):
        grid = (ct.cdiv(x.shape[0], tile_size[0]), ct.cdiv(x.shape[1], tile_size[1]),)
        ct.launch(torch.cuda.current_stream(), grid, kernel,
                  (x, y, tile_size[0], tile_size[1]))
        ref_fn(ref_x, y)
        torch.testing.assert_close(x, ref_x)


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("atomic_op", [AtomicOp.ADD, AtomicOp.XOR])
@pytest.mark.parametrize("shape", [(50, 60), (512, 256)])
@pytest.mark.parametrize("tile_size", [(128, 128)])
@pytest.mark.parametrize("traversal_steps", [(110, 110), (130, 130)])
def test_tiled_view_atomic_traversal_steps(atomic_op, shape, tile_size, traversal_steps):
    dtype = torch.int32
    get_tv_method, torch_op, _ = tv_atomic_configs[atomic_op]
    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = make_tensor(shape, dtype=dtype, device='cuda')
    ref_x = x.clone()
    kernel = make_tv_atomic_kernel(get_tv_method, traversal_steps)
    ref_fn = ref_atomic_bitwise if atomic_op.is_bitwise() else ref_atomic_arith
    ref_tv = make_ref_tv_atomic(ref_fn, torch_op, tile_size, traversal_steps)

    grid = (ct.cdiv(x.shape[0], traversal_steps[0]), ct.cdiv(x.shape[1], traversal_steps[1]),)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (x, y, tile_size[0], tile_size[1]))
    ref_tv(ref_x, y)
    torch.testing.assert_close(x, ref_x)


@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.parametrize("tile_size,update_size", [
    ((16, 16),  ()),
    ((128, 16), (1, 16)),
    ((16, 64),  (16, 1)),
    ((32, 16),  (1, 1)),
])
def test_tiled_view_atomic_broadcast(tile_size, update_size):
    @ct.kernel
    def kernel(x, y):
        tv_x = x.tiled_view(tile_size)
        update = y.tiled_view(update_size).load((0, 0))
        tv_x.atomic_store_add((0, 0), update)

    y_shape = update_size if len(update_size) > 0 else (1, 1)
    x = make_tensor(tile_size, dtype=torch.float32, device='cuda')
    y = make_tensor(y_shape, dtype=torch.float32, device='cuda')
    ref = x + torch.broadcast_to(y, tile_size)

    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    torch.testing.assert_close(x, ref)


@requires_tileiras(BytecodeVersion.V_13_3)
def test_tiled_view_atomic_shape_mismatch():
    @ct.kernel
    def kernel(x, y):
        tv_x = x.tiled_view(16)
        update = y.tiled_view(8).load(0)
        tv_x.atomic_store_add(0, update)

    x = torch.zeros((16,), dtype=torch.float32, device='cuda')
    y = torch.zeros((8,), dtype=torch.float32, device='cuda')

    with pytest.raises(TileTypeError, match=r"Update shape \(8,\) is not broadcastable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
