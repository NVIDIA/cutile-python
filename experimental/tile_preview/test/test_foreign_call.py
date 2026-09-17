# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.tile as ct
from cuda import tile_preview as tp
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._cext import CallingConvention
from cuda.tile._compile import compile_tile
from cuda.tile._datatype import is_foreign_pointer_dtype
from cuda.tile._exception import TileTypeError, TileUnsupportedFeatureError
from cuda.tile._ir.control_flow_ops import IfElse, Loop
from cuda.tile._ir.ops import (
    ForeignPointerCast,
    JoinTokens,
    TileStore,
)
from cuda.tile._ir.type import TileTy
from cuda.tile.compilation import ArrayConstraint, KernelSignature
from cuda.tile._bytecode.basic import encode_varint
from cuda.tile_preview import PreviewForeignCall


def _bytecode_from_int(x):
    buf = bytearray()
    encode_varint(x, buf)
    return bytes(buf)


PREVIEW_CALL_OPCODE = 16386  # from encodings.py
PREVIEW_CALL_OP_BYTECODE = _bytecode_from_int(PREVIEW_CALL_OPCODE)


def _compile(
    kernel,
    num_arrays=1,
    bytecode_version=BytecodeVersion.V_13_5,
    index_dtype=ct.int32,
):
    constraint = ArrayConstraint(
        dtype=ct.float32,
        ndim=1,
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
    )
    signature = KernelSignature(
        [constraint for _ in range(num_arrays)],
        CallingConvention.cutile_python_v2(),
        symbol="kernel",
    )
    return compile_tile(
        kernel,
        [signature],
        bytecode_version=bytecode_version,
        return_final_ir=True,
        return_bytecode=True,
        return_cubin=False,
    )


def test_foreign_call_requires_bytecode_v13_5(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tile = ct.load(x, (0,), shape=(16,))
        tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="fn",
            constant_params=(),
            inputs=(tile,),
            output_types=((ct.float32, (16,)),),
        )

    with pytest.raises(TileUnsupportedFeatureError):
        _compile(kernel, bytecode_version=BytecodeVersion.V_13_4)


def test_tile_and_scalar_call(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tile = ct.load(x, (0,), shape=(16,))
        result, _, _ = tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="tile_fn",
            constant_params=(16, "mode", ct.float32),
            inputs=(tile, ct.int32(0), 1.0),
            output_types=((ct.float32, (16,)), ct.int32, (ct.float32, ())),
        )
        ct.store(x, (0,), result)

    result = _compile(kernel)
    calls = [op for op in result.final_ir[0].traverse()
             if isinstance(op, PreviewForeignCall)]
    assert len(calls) == 1
    assert calls[0].tilelibs == ((tilelib.resolve(), "1"),)
    assert calls[0].token is not None
    assert int.from_bytes(result.bytecode[10:12], "little") & 1, "preview flag not set"
    assert PREVIEW_CALL_OP_BYTECODE in result.bytecode, "preview$call opcode not in bytecode"
    assert b"tile_fn" in result.bytecode, "symbol name not in bytecode"

    scalar_int32_ty = TileTy(ct.int32, ())
    assert calls[0].inputs[1].get_type() == scalar_int32_ty
    assert calls[0].result_vars[1].get_type() == scalar_int32_ty
    scalar_float32_ty = TileTy(ct.float32, ())
    assert calls[0].inputs[2].get_type() == scalar_float32_ty
    assert calls[0].result_vars[2].get_type() == scalar_float32_ty


def test_foreign_pointer_array_round_trip(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        pointer = x.foreign_pointer
        remote_pointer, = tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="array_fn",
            constant_params=(),
            inputs=(pointer,),
            output_types=((ct.foreign_pointer_dtype(ct.float32), ()),),
        )
        remote = ct.Array.from_foreign(
            remote_pointer, x.shape, x.strides
        )
        ct.store(x, (0,), ct.load(remote, (0,), shape=(16,)))

    result = _compile(kernel)
    calls = [op for op in result.final_ir[0].traverse()
             if isinstance(op, PreviewForeignCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.token is not None
    pointer_ty = call.inputs[0].get_type()
    assert isinstance(pointer_ty, TileTy)
    assert pointer_ty.shape == ()
    assert is_foreign_pointer_dtype(pointer_ty.dtype)
    casts = [op for op in result.final_ir[0].traverse()
             if isinstance(op, ForeignPointerCast)]
    assert len(casts) == 2
    assert int.from_bytes(result.bytecode[10:12], "little") & 1, "preview flag not set"
    assert PREVIEW_CALL_OP_BYTECODE in result.bytecode, "preview$call opcode not in bytecode"
    assert b"array_fn" in result.bytecode, "symbol name not in bytecode"


@pytest.mark.parametrize("output_types,match", [
    (((ct.float32,),), "must be a dtype or .*dtype, shape"),
    (((),), "must be a dtype or .*dtype, shape"),
    (((ct.float32, (), "extra"),), "must be a dtype or .*dtype, shape"),
    (((ct.float32, ct.int32),), "Expected a tuple"),
])
def test_foreign_call_rejects_invalid_output_descriptor(tmp_path, output_types, match):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="fn",
            constant_params=(),
            inputs=(),
            output_types=output_types,
        )

    with pytest.raises(TileTypeError, match=match):
        _compile(kernel)


def test_tile_call_is_not_removed_by_dce(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tile = ct.load(x, (0,), shape=(16,))
        result, = tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="tile_fn",
            constant_params=(16, "mode", ct.float32),
            inputs=(tile,),
            output_types=((ct.float32, (16,)),),
        )

    result = _compile(kernel)
    calls = [op for op in result.final_ir[0].traverse()
             if isinstance(op, PreviewForeignCall)]
    assert len(calls) == 1
    assert calls[0].token is not None
    assert PREVIEW_CALL_OP_BYTECODE in result.bytecode


def test_tile_call_orders_all_memory_chains(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x, y):
        before = ct.zeros((16,), dtype=ct.float32)
        ct.store(x, (0,), before)
        ct.store(y, (0,), before)
        result, = tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="tile_fn",
            constant_params=(),
            inputs=(before,),
            output_types=((ct.float32, (16,)),),
        )
        ct.store(x, (0,), result)
        ct.store(y, (0,), result)

    result = _compile(kernel, num_arrays=2)
    root_operations = result.final_ir[0].operations
    call = next(op for op in root_operations if isinstance(op, PreviewForeignCall))
    stores = [op for op in root_operations if isinstance(op, TileStore)]
    assert len(stores) == 4

    call_input_join = next(
        op for op in root_operations
        if isinstance(op, JoinTokens) and op.result_var is call.token
    )
    assert stores[0].result_var in call_input_join.tokens
    assert stores[1].result_var in call_input_join.tokens

    call_result_token = call.result_vars[-1]
    for store in stores[2:]:
        store_input_join = next(
            op for op in root_operations
            if isinstance(op, JoinTokens) and op.result_var is store.token
        )
        assert call_result_token in store_input_join.tokens


def test_foreign_call_rejects_array_input(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="array_fn",
            constant_params=(),
            inputs=(x,),
            output_types=(),
        )

    with pytest.raises(TileTypeError, match="foreign_pointer"):
        _compile(kernel)


def test_tile_call_token_order_across_control_flow(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tile = ct.zeros((16,), dtype=ct.float32)
        if ct.bid(0) == 0:
            tp.foreign_call(
                tilelibs=(provider,),
                symbol_name="inside_if",
                constant_params=(),
                inputs=(tile,),
                output_types=((ct.float32, (16,)),),
            )
        tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="after_if",
            constant_params=(),
            inputs=(tile,),
            output_types=((ct.float32, (16,)),),
        )

    result = _compile(kernel)
    root_operations = result.final_ir[0].operations
    if_op = next(op for op in root_operations if isinstance(op, IfElse))
    calls = [op for op in result.final_ir[0].traverse()
             if isinstance(op, PreviewForeignCall)]
    assert len(calls) == 2
    outer_call = next(op for op in calls if op.symbol_name == "after_if")
    token_join = next(
        op for op in root_operations
        if isinstance(op, JoinTokens) and op.result_var is outer_call.token
    )
    assert any(token in if_op.result_vars for token in token_join.tokens)


def test_tile_call_token_order_across_loop(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)

    def kernel(x):
        tile = ct.zeros((16,), dtype=ct.float32)
        for _ in range(2):
            tile, = tp.foreign_call(
                tilelibs=(provider,),
                symbol_name="inside_loop",
                constant_params=(),
                inputs=(tile,),
                output_types=((ct.float32, (16,)),),
            )
        tp.foreign_call(
            tilelibs=(provider,),
            symbol_name="after_loop",
            constant_params=(),
            inputs=(tile,),
            output_types=((ct.float32, (16,)),),
        )

    result = _compile(kernel)
    root_operations = result.final_ir[0].operations
    loop_op = next(op for op in root_operations if isinstance(op, Loop))
    outer_call = next(
        op for op in root_operations
        if isinstance(op, PreviewForeignCall) and op.symbol_name == "after_loop"
    )
    token_join = next(
        op for op in root_operations
        if isinstance(op, JoinTokens) and op.result_var is outer_call.token
    )
    assert any(token in loop_op.result_vars for token in token_join.tokens)


def test_tilelib_normalizes_path(tmp_path):
    tilelib = tmp_path / "provider.tilelib"
    tilelib.write_bytes(b"tilelib")
    provider = tp.Tilelib(version="1", path=tilelib)
    assert provider.version == "1"
    assert provider.path == str(tilelib.resolve())


def test_tilelibs_allow_compatible_references(tmp_path):
    path_a = tmp_path / "a.tilelib"
    path_b = tmp_path / "b.tilelib"
    path_a.write_bytes(b"tilelib")
    path_b.write_bytes(b"tilelib")
    tilelib_a = tp.Tilelib(path=path_a, version="1")
    tilelib_b = tp.Tilelib(path=path_b, version="1")

    def kernel(x):
        tp.foreign_call(
            tilelibs=(tilelib_b, tilelib_a, tilelib_b),
            symbol_name="fn",
            constant_params=(),
            inputs=(),
            output_types=(),
        )

    _compile(kernel)


def test_tilelibs_reject_conflicting_versions_for_same_path(tmp_path):
    path = tmp_path / "provider.tilelib"
    path.write_bytes(b"tilelib")
    tilelib_v1 = tp.Tilelib(path=path, version="1")
    tilelib_v2 = tp.Tilelib(path=path, version="2")

    def kernel(x):
        tp.foreign_call(
            tilelibs=(tilelib_v1, tilelib_v2),
            symbol_name="fn",
            constant_params=(),
            inputs=(),
            output_types=(),
        )

    with pytest.raises(ValueError, match="Conflicting versions for tilelib"):
        _compile(kernel)
