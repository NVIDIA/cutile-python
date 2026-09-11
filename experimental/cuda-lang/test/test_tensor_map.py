# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import cuda.lang as cl
from cuda.lang._exception import TypeCheckingError
from cuda.tile._cext import cconv_v3_enabled
from .util import require_blackwell_or_newer, require_hopper_or_newer


pytestmark = [
    pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled"),
    require_hopper_or_newer(),
]


@pytest.mark.parametrize(
    ("torch_dtype", "tile_shape"),
    (
        pytest.param(torch.uint8, (16, 4), id="uint8"),
        pytest.param(torch.int8, (16, 4), id="int8"),
        pytest.param(torch.int32, (4, 4), id="int32"),
        pytest.param(torch.int64, (2, 4), id="int64"),
        pytest.param(torch.float16, (8, 4), id="float16"),
        pytest.param(torch.bfloat16, (8, 4), id="bfloat16"),
        pytest.param(torch.float32, (4, 4), id="float32"),
        pytest.param(torch.float64, (2, 4), id="float64"),
    ),
)
def test_data_type_and_tile_shape(torch_dtype, tile_shape):
    source = torch.empty((8, 32), dtype=torch_dtype, device="cuda")
    tensor_map = cl.tensor_map_tiled(source, tile_shape, order="F")
    assert isinstance(tensor_map, cl.TensorMap)
    assert tensor_map.dtype is cl.tensor_map_descriptor
    assert len(bytes(tensor_map)) == 128
    assert not hasattr(tensor_map, "tile_shape")


def test_single_int_tile_shape_equals_tuple():
    source = torch.empty(32, dtype=torch.float32, device="cuda")
    assert bytes(cl.tensor_map_tiled(source, 32)) == bytes(
        cl.tensor_map_tiled(source, (32,))
    )


_BLACKWELL_SWIZZLES = {
    cl.SwizzleMode.SWIZZLE_128B_ATOM_32B,
    cl.SwizzleMode.SWIZZLE_128B_ATOM_32B_FLIP_8B,
    cl.SwizzleMode.SWIZZLE_128B_ATOM_64B,
}


@pytest.mark.parametrize(
    "swizzle",
    [
        pytest.param(swizzle, marks=require_blackwell_or_newer())
        if swizzle in _BLACKWELL_SWIZZLES else swizzle
        for swizzle in cl.SwizzleMode
    ],
)
@pytest.mark.parametrize("l2_promotion", tuple(cl.TensorMapL2Promotion))
@pytest.mark.parametrize("oob_fill", tuple(cl.TensorMapFloatOOBFill))
def test_options(swizzle, l2_promotion, oob_fill):
    source = torch.empty((8, 32), dtype=torch.float32, device="cuda")
    tensor_map = cl.tensor_map_tiled(source, (4, 4), order="F",
                                     swizzle=swizzle,
                                     l2_promotion=l2_promotion,
                                     oob_fill=oob_fill,)
    assert tensor_map.dtype is cl.tensor_map_descriptor


@pytest.mark.parametrize(
    ("interleave", "swizzle", "tile_shape"),
    (
        (cl.TensorMapInterleave.INTERLEAVE_16B, cl.SwizzleMode.SWIZZLE_NONE,
         (4, 4, 2)),
        (cl.TensorMapInterleave.INTERLEAVE_16B, cl.SwizzleMode.SWIZZLE_NONE,
         (8, 4, 2)),
        (cl.TensorMapInterleave.INTERLEAVE_32B, cl.SwizzleMode.SWIZZLE_32B,
         (16, 4, 2)),
    ),
)
@require_blackwell_or_newer()
def test_interleaved_tensor_map_eager_and_compiled(interleave, swizzle, tile_shape):
    @cl.kernel
    def copy_descriptor(descriptor, output):
        shared = cl.shared_array(1, cl.tensor_map_descriptor, alignment=128).pointer()
        lane = cl.thread_index(0)
        if lane == 0:
            shared[0] = descriptor.load()
        cl.barrier_sync_block_aligned()
        shared_bytes = cl.bitcast(shared, cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED))
        output[lane] = shared_bytes[lane]

    @cl.host_entry
    def launcher(stream, source, output):
        descriptor = cl.tensor_map_tiled(
            source, tile_shape, order="F", interleave=interleave, swizzle=swizzle,
        )
        cl.launch(stream, (1,), (128,), copy_descriptor, (descriptor, output))

    @cl.kernel
    def device_created(source, output):
        descriptor = cl.tensor_map_tiled(
            source, tile_shape, order="F", interleave=interleave, swizzle=swizzle,
        )
        shared = cl.shared_array(1, cl.tensor_map_descriptor, alignment=128).pointer()
        lane = cl.thread_index(0)
        if lane == 0:
            shared[0] = descriptor.load()
        cl.barrier_sync_block_aligned()
        shared_bytes = cl.bitcast(shared, cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED))
        output[lane] = shared_bytes[lane]

    source = torch.empty((4, 4, 32), dtype=torch.float16, device="cuda")
    eager = cl.tensor_map_tiled(
        source, tile_shape, order="F", interleave=interleave, swizzle=swizzle,
    )
    if tile_shape[0] * source.element_size() < 16:
        with pytest.raises(ValueError, match="multiple of 128 bits"):
            cl.tensor_map_tiled(source, tile_shape, order="F", swizzle=swizzle)
    else:
        noninterleaved = cl.tensor_map_tiled(source, tile_shape, order="F", swizzle=swizzle)
        assert bytes(eager) != bytes(noninterleaved)
    output = torch.empty(128, dtype=torch.uint8, device="cuda")
    launcher(torch.cuda.current_stream(), source, output)
    assert bytes(output.cpu().tolist()) == bytes(eager)
    cl.launch(torch.cuda.current_stream(), (1,), (128,), device_created, (source, output))
    assert bytes(output.cpu().tolist()) == bytes(eager)


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize(
    ("source_kind", "interleave", "swizzle", "match"),
    (
        ("rank_two", cl.TensorMapInterleave.INTERLEAVE_16B,
         cl.SwizzleMode.SWIZZLE_NONE, "interleaved tensor maps require rank >= 3"),
        ("aligned", cl.TensorMapInterleave.INTERLEAVE_32B,
         cl.SwizzleMode.SWIZZLE_NONE, "32-byte interleave requires SWIZZLE_32B"),
        ("misaligned_base", cl.TensorMapInterleave.INTERLEAVE_32B,
         cl.SwizzleMode.SWIZZLE_32B, "global address must be aligned to 32 bytes"),
        ("misaligned_stride", cl.TensorMapInterleave.INTERLEAVE_32B,
         cl.SwizzleMode.SWIZZLE_32B, "strides to be a multiple of 32"),
    ),
)
def test_interleaved_tensor_map_rejects_invalid_layout(
    mode, source_kind, interleave, swizzle, match,
):
    @cl.kernel
    def prefetch(descriptor):
        cl.prefetch_tensor_map(descriptor)

    @cl.host_entry
    def launcher(stream, source):
        descriptor = cl.tensor_map_tiled(
            source, tile_shape, order="F", interleave=interleave, swizzle=swizzle,
        )
        cl.launch(stream, (1,), (1,), prefetch, (descriptor,))

    if source_kind == "rank_two":
        source = torch.empty((4, 32), dtype=torch.float16, device="cuda")
        tile_shape = (16, 2)
    elif source_kind == "misaligned_base":
        source = torch.empty((4, 4, 48), dtype=torch.float16, device="cuda")[:, :, 8:40]
        tile_shape = (16, 4, 2)
    elif source_kind == "misaligned_stride":
        source = torch.empty((4, 4, 24), dtype=torch.float16, device="cuda")
        tile_shape = (16, 4, 2)
    else:
        source = torch.empty((4, 4, 32), dtype=torch.float16, device="cuda")
        tile_shape = (16, 4, 2)
    with pytest.raises(ValueError, match=match):
        if mode == "eager":
            cl.tensor_map_tiled(
                source, tile_shape, order="F", interleave=interleave, swizzle=swizzle,
            )
        else:
            launcher(torch.cuda.current_stream(), source)


def test_tensor_map_tiled_rejects_invalid_interleave_type():
    source = torch.empty((4, 4, 32), dtype=torch.float16, device="cuda")
    with pytest.raises(TypeError, match="interleave must be a TensorMapInterleave"):
        cl.tensor_map_tiled(source, (16, 4, 2), order="F", interleave="32B")


def test_contiguous_array():
    source = torch.empty((8, 32), dtype=torch.int32, device="cuda")
    tensor_map = cl.tensor_map_tiled(source, (8, 4), order="F")
    assert tensor_map.dtype is cl.tensor_map_descriptor


def test_transposed_array():
    source = torch.empty((32, 8), dtype=torch.int32, device="cuda").T
    tensor_map = cl.tensor_map_tiled(source, (4, 8), order="C")
    assert tensor_map.dtype is cl.tensor_map_descriptor


def test_padded_array():
    source = torch.empty((8, 64), dtype=torch.int32, device="cuda")[:, :32]
    tensor_map = cl.tensor_map_tiled(source, (8, 4), order="F")
    assert tensor_map.dtype is cl.tensor_map_descriptor


@pytest.mark.parametrize("host_mode", ["eager", "compiled"])
def test_tensor_map_passed_to_kernel(host_mode):
    @cl.kernel
    def kernel(tensor_map, out):
        cl.static_assert(
            cl.dtype_of(tensor_map) == cl.pointer_dtype(cl.tensor_map_descriptor)
        )
        cl.prefetch_tensor_map(tensor_map)
        out[0] = 1

    def eager_launcher(stream, array, out):
        tensor_map = cl.tensor_map_tiled(
            array,
            (8, 4),
            order="F",
            swizzle=cl.SwizzleMode.SWIZZLE_32B,
            l2_promotion=cl.TensorMapL2Promotion.L2_128B,
            oob_fill=cl.TensorMapFloatOOBFill.NAN_REQUEST_ZERO_FMA,
        )
        cl.launch(stream, (1,), (1,), kernel, (tensor_map, out))

    @cl.host_entry
    def compiled_launcher(stream, array, out):
        eager_launcher(stream, array, out)

    array = torch.empty((8, 32), dtype=torch.float16, device="cuda")
    out = torch.empty((1,), dtype=torch.int32, device='cuda')
    stream = torch.cuda.current_stream()

    if host_mode == 'eager':
        out.zero_()
        eager_launcher(stream, array, out)
        assert out.item() == 1
    elif host_mode == 'compiled':
        out.zero_()
        compiled_launcher(stream, array, out)
        assert out.item() == 1
    else:
        assert False


def test_copy_descriptor_argument_to_shared():
    @cl.kernel
    def kernel(descriptor, output):
        shared = cl.shared_array(1, cl.tensor_map_descriptor, alignment=128).pointer()
        lane = cl.thread_index(0)
        if lane == 0:
            shared[0] = descriptor.load()
            value = shared[0]
            cl.static_assert(cl.dtype_of(value) == cl.tensor_map_descriptor)
        cl.barrier_sync_block_aligned()
        shared_bytes = cl.bitcast(shared, cl.pointer_dtype(cl.uint8, cl.MemorySpace.SHARED))
        output[lane] = shared_bytes[lane]

    source = torch.arange(32, dtype=torch.float32, device="cuda")
    descriptor = cl.tensor_map_tiled(source, (32,))
    output = torch.empty(128, dtype=torch.uint8, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (128,), kernel, (descriptor, output))
    assert bytes(output.cpu().tolist()) == bytes(descriptor)


@pytest.mark.parametrize("host_mode", ["eager", "compiled"])
@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("tile_shape", [32, (32,)])
def test_tensor_map_value_in_dynamic_control_flow(host_mode, count, tile_shape):
    @cl.kernel
    def copy_tile(descriptor, output, row):
        tile = cl.shared_array(32, cl.float32, alignment=128).pointer()
        barrier = cl.shared_array(1, cl.mbarrier, alignment=8).pointer()
        cl.mbarrier_initialize(barrier, 1)
        cl.fence(
            cl.MemoryOrder.RELEASE,
            cl.MemoryScope.CLUSTER,
            restriction=cl.FenceRestriction.mbarrier_initialize(),
        )
        cl.mbarrier_arrive_expect_transaction(barrier, 128)
        cl.copy_async_bulk_tensor_global_to_shared(descriptor, (0,), tile, barrier)
        cl.mbarrier_wait_parity(barrier, 0)
        for i in range(32):
            output[row, i] = tile[i]

    def run(stream, first, second, output, count):
        if count > 0:
            previous = cl.tensor_map_tiled(first, tile_shape)
            for i in range(count):
                if i % 2 == 0:
                    source = second
                else:
                    source = first
                current = cl.tensor_map_tiled(source, tile_shape)
                # Retaining the previous iteration's value must preserve its bytes.
                cl.launch(stream, (1,), (1,), copy_tile, (previous, output, i))
                previous = current

    launcher = run if host_mode == "eager" else cl.host_entry(run)
    first = torch.full((32,), 1.0, device="cuda")
    second = torch.full((32,), 2.0, device="cuda")
    output = torch.zeros((max(count, 1), 32), device="cuda")
    launcher(torch.cuda.current_stream(), first, second, output, count)
    if count == 0:
        torch.testing.assert_close(output, torch.zeros_like(output))
    for i in range(count):
        torch.testing.assert_close(output[i], first if i % 2 == 0 else second)


@pytest.mark.parametrize(
    ("source_shape", "tile_shape", "match"),
    (
        pytest.param((), (), r"rank must be between one and five", id="rank-zero"),
        pytest.param(
            (32,),
            (0,),
            r"tile dimensions must be between 1 and 256",
            id="zero-extent",
        ),
        pytest.param(
            (32,),
            (257,),
            r"tile dimensions must be between 1 and 256",
            id="extent-too-large",
        ),
        pytest.param(
            (32,),
            (3,),
            r"first tile dimension times the element bit width",
            id="misaligned-extent",
        ),
    ),
)
def test_tensor_map_tiled_invalid_tile_shape(source_shape, tile_shape, match):
    source = torch.empty(source_shape, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=match):
        cl.tensor_map_tiled(source, tile_shape)


@pytest.mark.parametrize(
    ("swizzle", "dtype", "tile_width"),
    (
        pytest.param(cl.SwizzleMode.SWIZZLE_32B, torch.float32, 12, id="32b-float32"),
        pytest.param(cl.SwizzleMode.SWIZZLE_64B, torch.float16, 40, id="64b-float16"),
        pytest.param(cl.SwizzleMode.SWIZZLE_128B, torch.float32, 36, id="128b"),
        pytest.param(
            cl.SwizzleMode.SWIZZLE_128B_ATOM_32B, torch.float32, 36,
            id="128b-atom-32b",
        ),
        pytest.param(
            cl.SwizzleMode.SWIZZLE_128B_ATOM_32B_FLIP_8B, torch.float32, 36,
            id="128b-atom-32b-flip-8b",
        ),
        pytest.param(
            cl.SwizzleMode.SWIZZLE_128B_ATOM_64B, torch.float32, 36,
            id="128b-atom-64b",
        ),
    ),
)
def test_tensor_map_tiled_rejects_tile_wider_than_swizzle(swizzle, dtype, tile_width):
    source = torch.empty((8, 160), dtype=dtype, device="cuda")
    with pytest.raises(
        ValueError, match=r"first tile dimension spans .* exceeding the .* swizzle span"
    ):
        cl.tensor_map_tiled(source, (tile_width, 4), order="F", swizzle=swizzle)


@pytest.mark.parametrize(
   ("order", "error", "match"),
   (
       pytest.param(
           "invalid",
           ValueError,
           r"order must be 'C', 'F', or an axis permutation",
           id="invalid-string",
       ),
       pytest.param(
           (0,),
           ValueError,
           r"order must be a permutation of all array axes",
           id="wrong-rank",
       ),
       pytest.param(
           (0, 0),
           ValueError,
           r"order must be a permutation of all array axes",
           id="duplicate-axis",
       ),
   ),
)
def test_tensor_map_tiled_invalid_order(order, error, match):
    source = torch.empty((8, 32), dtype=torch.int32, device="cuda")
    with pytest.raises(error, match=match):
        cl.tensor_map_tiled(source, (8, 4), order=order)


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.int16, id="int16"),
        pytest.param(torch.float4_e2m1fn_x2, id="float4"),
    ),
)
def test_tensor_map_tiled_rejects_unsupported_data_type(dtype):
    source = torch.empty((8, 32), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match=r"is not supported by tensor map"):
        cl.tensor_map_tiled(source, (8, 4), order="F")


def test_tensor_map_tiled_rejects_misaligned_global_address():
    source = torch.empty(33, dtype=torch.int32, device="cuda")[1:]
    with pytest.raises(
        ValueError, match=r"global address must be aligned to 16 bytes"
    ):
        cl.tensor_map_tiled(source, (4,))


def test_tensor_map_tiled_rejects_non_unit_innermost_stride():
    source = torch.empty((8, 64), dtype=torch.int32, device="cuda")[:, ::2]
    with pytest.raises(
        ValueError, match=r"stride of descriptor axis zero must be 1"
    ):
        cl.tensor_map_tiled(source, (8, 4), order="F")


def test_tensor_map_tiled_rejects_misaligned_outer_stride():
    source = torch.empty((8, 31), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=r"byte strides must be a multiple of 16"):
        cl.tensor_map_tiled(source, (8, 4), order="F")


def test_compiled_tensor_map_rejects_rank_mismatch():
    @cl.kernel
    def kernel(descriptor):
        cl.prefetch_tensor_map(descriptor)

    @cl.host_entry
    def launcher(stream, source):
        descriptor = cl.tensor_map_tiled(source, (4,))
        cl.launch(stream, (1,), (1,), kernel, (descriptor,))

    source = torch.empty((32, 8), dtype=torch.float32, device="cuda").T
    with pytest.raises(TypeCheckingError, match="tile shape must match the array rank"):
        launcher(torch.cuda.current_stream(), source)
