# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch.cuda

import cuda.tile._bytecode as bc
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._compile import compile_cubin, get_sm_arch
from cuda.tile import _cext
import cuda.tile as ct


def test_write_simple_module():
    buf = bytearray()
    with bc.write_bytecode(1, buf, version=bc.BytecodeVersion.V_13_1) as writer:
        tt = writer.type_table
        with writer.function(name="foo",
                             parameter_types=[tt.tile(tt.pointer(tt.F32), ()),
                                              tt.tile(tt.I32, ()),
                                              tt.tile(tt.I32, ()),
                                              tt.tile(tt.pointer(tt.F32), ()),
                                              tt.tile(tt.I32, ()),
                                              tt.tile(tt.I32, ()),
                                              tt.tile(tt.pointer(tt.F32), ()),
                                              tt.tile(tt.I32, ()),
                                              tt.tile(tt.I32, ())],
                             result_types=[],
                             entry_point=True,
                             hints=None,
                             debug_attr=bc.MISSING_DEBUG_ATTR_ID) as (builder, parameters):
            x, x_len, x_stride, y, y_len, y_stride, result, result_len, result_stride = parameters
            view_ty = tt.tensor_view(tt.F32, (bc.DYNAMIC_SHAPE,), (1,))
            partition_ty = tt.partition_view((1,), view_ty, [0], bc.PaddingValue.Zero)

            partitions = []
            for ptr, size in [(x, x_len), (y, y_len), (result, result_len)]:
                view = bc.encode_MakeTensorViewOp(builder,
                                                  result_type=view_ty,
                                                  base=ptr,
                                                  dynamicShape=[size],
                                                  dynamicStrides=[])
                partitions.append(bc.encode_MakePartitionViewOp(builder, partition_ty, view))

            x_pv, y_pv, result_pv = partitions

            index_ty = tt.tile(tt.I32, ())
            zero_i32 = bc.encode_ConstantOp(builder, index_ty, (0).to_bytes(4, "little"))

            tile_ty = tt.tile(tt.F32, (1,))
            tiles = []
            for pv in x_pv, y_pv:
                tile, _ = bc.encode_LoadViewTkoOp(
                        builder,
                        tile_type=tile_ty,
                        result_token_type=tt.Token,
                        view=pv,
                        index=[zero_i32],
                        memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
                        memory_scope=None,
                        optimization_hints=None,
                        token=None)
                tiles.append(tile)

            x_tile, y_tile = tiles
            result_tile = bc.encode_AddFOp(
                    builder,
                    lhs=x_tile,
                    rhs=y_tile,
                    result_type=tile_ty,
                    rounding_mode=bc.RoundingMode.NEAREST_EVEN,
                    flush_to_zero=False)

            bc.encode_StoreViewTkoOp(
                    builder,
                    result_token_type=tt.Token,
                    tile=result_tile,
                    view=result_pv,
                    index=[zero_i32],
                    token=None,
                    memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
                    memory_scope=None,
                    optimization_hints=None)

            bc.encode_ReturnOp(builder, [])

    with NamedTemporaryFile() as f:
        f.write(buf)
        f.flush()
        cubin_path = compile_cubin(f.name, CompilerOptions(), get_sm_arch(), None)
        cubin = Path(cubin_path).read_bytes()

    kernel = _HackKernel(cubin, "foo")
    x_tensor = torch.tensor([3.0], dtype=torch.float32, device="cuda")
    y_tensor = torch.tensor([5.0], dtype=torch.float32, device="cuda")
    result = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x_tensor, y_tensor, result))
    assert result.cpu().item() == 8.0


class _HackKernel(_cext.TileDispatcher):
    def __init__(self, cubin: bytes, func_name: str):
        self._cubin = cubin
        self._func_name = func_name
        super().__init__((False, False, False), (False, False, False), (False, False, False))

    def _compile(self, signature, ctx):
        assert len(signature.parameters) == 3
        for x in signature.parameters:
            assert x.ndim == 1
            assert x.dtype == ct.float32
        return self._cubin, self._func_name
