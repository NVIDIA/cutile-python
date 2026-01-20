# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import override
from util import filecheck, get_bytecode

import pytest
import torch
import numpy as np

import cuda.tile as ct


class TestMemoryBehavior:

    def store_buffer(X, TILE: ct.Constant[int]):
        tx0 = ct.arange(TILE, dtype=X.dtype)
        ct.store(X, index=(0,), tile=tx0)
        # reverse so that each SIMT thread is less likely to be assigned the same address
        X_ALIAS = X
        reverse_offset = TILE - 1 - ct.arange(TILE, dtype=np.int32)
        tx1 = ct.gather(X_ALIAS, reverse_offset)
        ct.store(X_ALIAS, index=(0,), tile=tx1)

    def store_buffer_alternative(X, TILE: ct.Constant[int]):
        reverse_offset = TILE - 1 - ct.arange(TILE, dtype=np.int32)
        tx2 = ct.arange(TILE, dtype=X.dtype)
        ct.scatter(X, reverse_offset, tx2)
        tx3 = ct.load(X, index=(0,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx3)

    def serialized_for_loop(X, TILE: ct.Constant[int]):
        ct.store(X, index=(0,), tile=ct.arange(TILE, dtype=X.dtype))
        # flip the buffer 3 times
        for i in range(3):
            reverse_offset = TILE - 1 - ct.arange(TILE, dtype=np.int32)
            tx = ct.gather(X, reverse_offset)
            ct.store(X, index=(0,), tile=tx)

    def serialized_while_loop(X, TILE: ct.Constant[int]):
        ct.store(X, index=(0,), tile=ct.arange(TILE, dtype=X.dtype))
        # flip the buffer 3 times
        i = 0
        while i < 3:
            reverse_offset = TILE - 1 - ct.arange(TILE, dtype=np.int32)
            tx = ct.gather(X, reverse_offset)
            ct.store(X, index=(0,), tile=tx)
            i += 1

    @staticmethod
    @ct.kernel
    def spinning_lock(X, L):
        while ct.atomic_cas(L, (), 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
            pass
        x_scalar = ct.gather(X, ())
        x_scalar += 1
        ct.scatter(X, (), x_scalar)
        ct.atomic_xchg(L, (), 0, memory_order=ct.MemoryOrder.RELEASE)

    @pytest.mark.parametrize("kernel", [
        store_buffer, store_buffer_alternative,
        serialized_for_loop, serialized_while_loop
    ], ids=lambda f: f.__name__)
    def test_memory_behavior(self, kernel):
        tile_size = 1024
        X = torch.zeros(tile_size, device="cuda", dtype=torch.int32)
        expected = torch.flip(torch.arange(tile_size, device="cuda", dtype=torch.int32), [0])
        ct.launch(torch.cuda.current_stream(), (1,), ct.kernel(kernel), (X, tile_size))
        torch.testing.assert_close(X, expected)

    def test_spinning_lock(self):
        n = 1024
        X = torch.tensor(3.14, device="cuda", dtype=torch.float32)
        L = torch.tensor(0, device="cuda", dtype=torch.int32)
        expected = X + n
        ct.launch(torch.cuda.current_stream(), (n,), self.spinning_lock, (X, L))
        torch.testing.assert_close(X, expected)


@pytest.mark.use_mlir
class MLIRTestBase:
    def compile_kernel(self, kernel) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def test_mlir(self, kernel, check_directive: str):
        bytecode = self.compile_kernel(kernel)
        filecheck(bytecode, check_directive)


def make_cases(*tuples):
    return [pytest.param(*t, id=t[0].__name__) for t in tuples]


NoControlFlowCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN3:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[VAL1:.*]], %[[TOKEN4:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN5:.*]] = join_tokens %[[TOKEN3]], %[[TOKEN4]]
// CHECK: %[[TOKEN6:.*]] = store_view_tko {{.*}} token = %[[TOKEN5]]
// CHECK: %[[VAL7:.*]], %[[TOKEN7:.*]] = load_ptr_tko {{.*}} token=%[[TOKEN6]]
// CHECK: %[[TOKEN8:.*]] = join_tokens %[[TOKEN6]], %[[TOKEN7]]
// CHECK: %[[TOKEN9:.*]] = store_ptr_tko {{.*}} token=%[[TOKEN8]]
"""


class TestNoControlFlowMLIR(MLIRTestBase):
    def no_control_flow(X, TILE: ct.Constant[int]):
        tx1 = ct.load(X, index=(0,), shape=(TILE,))
        tx2 = ct.load(X, index=(1,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx1 + tx2)
        offset1 = ct.arange(TILE, dtype=np.int32)
        tx3 = ct.gather(X, offset1)
        offset2 = TILE + ct.arange(TILE, dtype=np.int32)
        ct.scatter(X, offset2, tx3)

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.ones((2 * tile_size,), device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (no_control_flow, NoControlFlowCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


IfElseLoadCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENPAIR:.*]]:2 = if
// CHECK:     %[[VAL3:.*]], %[[TOKEN3:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TOKEN2]], %[[TOKEN3]]
// CHECK:     yield %[[VAL3:.*]], %[[TOKEN4]]
// CHECK: else
// CHECK:     yield %[[VAL1:.*]], %[[TOKEN2]]
// CHECK: %[[TOKEN5:.*]] = store_view_tko {{.*}} token = %[[TOKENPAIR]]#1
"""


IfElseLoadStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENPAIR:.*]]:2 = if
// CHECK:     %[[TOKEN3:.*]] = store_view_tko {{.*}} token = %[[TOKEN2]]
// CHECK:     %[[VAL4:.*]], %[[TOKEN4:.*]] = load_view_tko {{.*}} token = %[[TOKEN3]]
// CHECK:     %[[TOKEN5:.*]] = join_tokens %[[TOKEN3]], %[[TOKEN4]]
// CHECK:     yield %[[VAL4:.*]], %[[TOKEN5]]
// CHECK: else
// CHECK:     %[[VAL5:.*]], %[[TOKEN6:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK:     %[[TOKEN7:.*]] = join_tokens %[[TOKEN2]], %[[TOKEN6]]
// CHECK:     yield %[[VAL5:.*]], %[[TOKEN7]]
// CHECK: %[[TOKEN8:.*]] = store_view_tko {{.*}} token = %[[TOKENPAIR]]#1
"""


class TestIfElseMLIR(MLIRTestBase):

    def ifelse_load(X, cond, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        if cond:
            tx = ct.load(X, index=(0,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx)

    def ifelse_load_store(X, cond, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        if cond:
            tx += 1
            ct.store(X, index=(0,), tile=tx)
            tx = ct.load(X, index=(0,), shape=(TILE,))
        else:
            tx = ct.load(X, index=(0,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx)

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, True, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (ifelse_load, IfElseLoadCheckDirective),
        (ifelse_load_store, IfElseLoadStoreCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


ForLoopLoadCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENPAIR:.*]]:2 = for {{.*}} iter_values({{.*}}, %[[TKNARG:.*]] = %[[TOKEN2]])
// CHECK:     %[[VAL3:.*]], %[[TOKEN3:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TKNARG]], %[[TOKEN3]]
// CHECK:     continue %[[VAL3:.*]], %[[TOKEN4]]
// CHECK: %[[TOKEN5:.*]] = store_view_tko {{.*}} token = %[[TOKENPAIR]]#1
"""


ForLoopLoadStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENTRIPLET:.*]]:3 = for {{.*}} iter_values(
// CHECK-SAME: {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN2]], %[[TKNARG1:.*]] = %[[TOKEN0]])
// CHECK:     %[[VAL3:.*]], %[[TOKEN3:.*]] = load_view_tko {{.*}} token = %[[TKNARG1]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TKNARG0]], %[[TOKEN3]]
// CHECK:     %[[TOKEN5:.*]] = store_view_tko {{.*}} token = %[[TOKEN4]]
// CHECK:     continue {{.*}}, %[[TOKEN5]], %[[TOKEN5]]
// CHECK: %[[TOKEN6:.*]] = store_view_tko {{.*}} token = %[[TOKENTRIPLET]]#1
"""


class TestForLoopMLIR(MLIRTestBase):

    def for_loop_load(X, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        for i in range(n):
            tx = ct.load(X, index=(i,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx)

    def for_loop_load_store(X, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        for i in range(n):
            tx = ct.load(X, index=(0,), shape=(TILE,))
            tx += 1
            ct.store(X, index=(0,), tile=tx)
        ct.store(X, index=(0,), tile=tx)

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, 10, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (for_loop_load, ForLoopLoadCheckDirective),
        (for_loop_load_store, ForLoopLoadStoreCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


ForLoopParallelStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[LOOP_TOK:.*]] = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN2]])
// CHECK:     %[[TOKEN3:.*]] = store_view_tko {{.*}} token = %[[TOKEN2]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TKNARG0]], %[[TOKEN3]]
// CHECK:     continue %[[TOKEN4]]
// CHECK: %[[TOKEN5:.*]] = store_view_tko {{.*}} token = %[[LOOP_TOK]]
"""


ForLoopTwoParallelStoresCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[XTOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[XTOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[XTOKEN1]]
// CHECK: for {{.*}}
// CHECK:     %[[XTOKEN3:.*]] = store_view_tko {{.*}} token = %[[XTOKEN2]]
// CHECK:     %[[YTOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
"""


ForLoopTwoParallelStoresNestedCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[XTOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[XTOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[XTOKEN1]]
// CHECK: %[[Y_OUTER_LOOP_TOK:.*]] = for {{.*}} iter_values(
// CHECK-SAME: %[[YTKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     %[[XTOKEN3:.*]] = store_view_tko {{.*}} token = %[[XTOKEN2]]
// CHECK:     %[[Y_INNER_LOOP_TOK:.*]] = for {{.*}} iter_values(
// CHECK-SAME: %[[YTKNARG2:.*]] = %[[YTKNARG0]])
// CHECK:         %[[YTOKEN1:.*]] = store_view_tko {{.*}} token = %[[YTKNARG0]]
// CHECK:         %[[YTOKEN2:.*]] = join_tokens %[[YTKNARG2]], %[[YTOKEN1]]
// CHECK:         continue %[[YTOKEN2]]
// CHECK:     continue %[[Y_INNER_LOOP_TOK]]
"""

ForLoopOneParallelOneSerialStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[XTOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[XTOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[XTOKEN1]]
// CHECK: %[[TOKENPAIR:.*]]:2 = for {{.*}} iter_values(
// CHECK-SAME: %[[YTKNARG0:.*]] = %[[TOKEN0]], %[[YTKNARG1:.*]] = %[[TOKEN0]])
// CHECK:     %[[XTOKEN3:.*]] = store_view_tko {{.*}} token = %[[XTOKEN2]]
// CHECK:     {{.*}}, %[[YTOKEN1:.*]] = load_view_tko {{.*}} token = %[[YTKNARG1]]
// CHECK:     %[[YTOKEN2:.*]] = join_tokens %[[YTKNARG0]], %[[YTOKEN1]]
// CHECK:     %[[YTOKEN3:.*]] = store_view_tko {{.*}} token = %[[YTOKEN2]]
// CHECK:     continue %[[YTOKEN3]], %[[YTOKEN3]]
"""


class TestForLoopParallelStoreMLIR(MLIRTestBase):

    def parallel_store(X, _Y, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        for i in range(n-1):
            ct.store(X, index=(i,), tile=tx)
        ct.store(X, index=(n-1,), tile=tx)

    def two_parallel_stores(X, Y, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        range_n = range(n)
        for i in range_n:
            ct.store(X, index=(i,), tile=tx)
            ct.store(Y, index=(i,), tile=tx + 1)

    def two_parallel_stores_with_nested_block(X, Y, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        for i in range(n):
            ct.store(X, index=(i,), tile=tx)
            for j in range(n):
                ct.store(Y, index=(j,), tile=tx + j)

    def one_parallel_one_serial_store(X, Y, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        for i in range(n):
            # parallel store
            ct.store(X, index=(i,), tile=tx)
            ty = ct.load(Y, index=(i,), shape=(TILE,))
            # serial store
            ct.store(Y, index=(i,), tile=ty)

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        n = 10
        X = torch.arange(tile_size * 10, device="cuda", dtype=torch.int32)
        Y = torch.arange(tile_size * 10, device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, Y, n, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (parallel_store, ForLoopParallelStoreCheckDirective),
        (two_parallel_stores, ForLoopTwoParallelStoresCheckDirective),
        (two_parallel_stores_with_nested_block,
         ForLoopTwoParallelStoresNestedCheckDirective),
        (one_parallel_one_serial_store,
         ForLoopOneParallelOneSerialStoreCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


ForLoopNonParallelStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[XTOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[XTOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[XTOKEN1]]
// CHECK: %[[TOKENPAIR:.*]]:2 = for {{.*}} iter_values(
// CHECK-SAME: %[[XTKNARG0:.*]] = %[[XTOKEN2]],
// CHECK-SAME: %[[YTKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     %[[XTOKEN3:.*]] = store_view_tko {{.*}} token = %[[XTKNARG0]]
// CHECK:     %[[YTOKEN1:.*]] = store_view_tko {{.*}} token = %[[YTKNARG0]]
// CHECK:     continue %[[XTOKEN3]], %[[YTOKEN1]]
"""


class TestForLoopNonParallelStoreMLIR(MLIRTestBase):

    def non_parallel_store_non_disjoint(X, Y, n: int, TILE: ct.Constant[int]):
        bidx = ct.bid(0)
        tx = ct.load(X, index=(bidx, 0), shape=(TILE, TILE))
        for i in range(n):
            ct.store(X, index=(bidx, i), tile=tx)
            ct.store(Y, index=(bidx, i), tile=tx + i)

    @override
    def compile_kernel(self, kernel):
        tile_size = 128
        n = 8
        # X, Y each of non-disjoint elements in memory
        X = torch.tensor(1., dtype=torch.float32, device="cuda").broadcast_to(
            (tile_size * n, tile_size * n))
        Y = torch.randn((tile_size * n, tile_size * n), device="cuda",
                        dtype=torch.float32)
        Y = torch.as_strided(Y, size=(tile_size * n, tile_size * n),
                             stride=(2, 1))
        bytecode = get_bytecode(kernel, (X, Y, n, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (non_parallel_store_non_disjoint, ForLoopNonParallelStoreCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


WhileLoopLoadCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENTRIPLET:.*]]:3 = loop iter_values(
// CHECK-SAME: {{.*}}, {{.*}}, %[[TKNARG:.*]] = %[[TOKEN2]])
// CHECK:     else
// CHECK:         break {{.*}}, {{.*}}, %[[TKNARG]]
// CHECK:     %[[VAL3:.*]], %[[TOKEN3:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TKNARG]], %[[TOKEN3]]
// CHECK:     continue {{.*}}, %[[VAL3]], %[[TOKEN4]]
// CHECK: %[[VAL5:.*]], %[[TOKEN5:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN6:.*]] = join_tokens %[[TOKENTRIPLET]]#2, %[[TOKEN5]]
// CHECK: %[[TOKEN7:.*]] = store_view_tko {{.*}} token = %[[TOKEN6]]
"""


WhileLoopLoadStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKENQUARTET:.*]]:4 = loop iter_values(
// CHECK-SAME: {{.*}}, {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN2]], %[[TKNARG1:.*]] = %[[TOKEN0]])
// CHECK:     else
// CHECK:         break {{.*}}, {{.*}}, %[[TKNARG0]], %[[TKNARG1]]
// CHECK:     %[[VAL3:.*]], %[[TOKEN3:.*]] = load_view_tko {{.*}} token = %[[TKNARG1]]
// CHECK:     %[[TOKEN4:.*]] = join_tokens %[[TKNARG0]], %[[TOKEN3]]
// CHECK:     %[[TOKEN5:.*]] = store_view_tko {{.*}} token = %[[TOKEN4]]
// CHECK:     continue {{.*}}, {{.*}}, %[[TOKEN5]], %[[TOKEN5]]
// CHECK: %[[TOKEN6:.*]] = store_view_tko {{.*}} token = %[[TOKENQUARTET]]#2
"""


class TestWhileLoopMLIR(MLIRTestBase):

    def while_loop_load(X, n: int, TILE: ct.Constant[int]):
        tx1 = ct.load(X, index=(0,), shape=(TILE,))
        i = 0
        while i < n:
            tx1 = ct.load(X, index=(i,), shape=(TILE,))
            i += 1
        tx2 = ct.load(X, index=(0,), shape=(TILE,))
        ct.store(X, index=(0,), tile=tx1 + tx2)

    def while_loop_load_store(X, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        i = 0
        while i < n:
            tx = ct.load(X, index=(0,), shape=(TILE,))
            tx += 1
            ct.store(X, index=(0,), tile=tx)
            i += 1
        ct.store(X, index=(0,), tile=tx)

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, 10, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (while_loop_load, WhileLoopLoadCheckDirective),
        (while_loop_load_store, WhileLoopLoadStoreCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


RelaxedNoControlFlowCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko relaxed {{.*}} token=%[[TOKEN0]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
// CHECK: %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
"""  # noqa: E501

AcquireNoControlFlowCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acquire {{.*}} token=%[[TOKEN0]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
// CHECK: %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
"""  # noqa: E501

ReleaseNoControlFlowCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[Y_TOKEN1:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN2]]
// CHECK: %[[VAL2:.*]], %[[Y_TOKEN2:.*]] = atomic_rmw_tko release {{.*}} token=%[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
// CHECK: %[[Y_TOKEN3:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN2]]
"""  # noqa: E501


AcqRelNoControlFlowCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[Y_TOKEN1:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN2]]
// CHECK: %[[VAL2:.*]], %[[Y_TOKEN2:.*]] = atomic_rmw_tko acq_rel {{.*}} token=%[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[Y_TOKEN2]]
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
// CHECK: %[[Y_TOKEN3:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN2]]
"""  # noqa: E501


RelaxedIfElseCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = if
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko relaxed {{.*}} token=%[[TOKEN0]]
// CHECK:     yield %[[VAL2]], %[[Y_TOKEN1]]
// CHECK: else
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN2:.*]] = load_ptr_tko weak {{.*}} token=%[[TOKEN0]]
// CHECK:     %[[Y_TOKEN3:.*]] = join_tokens %[[TOKEN0]], %[[Y_TOKEN2]]
// CHECK:     yield %[[VAL3]], %[[Y_TOKEN3]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
// CHECK: %[[Y_TOKEN4:.*]] = store_ptr_tko weak {{.*}} token=%[[TOKENTUPLE]]#1
"""  # noqa: E501


AcquireIfElseCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:3 = if
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acquire {{.*}} token=%[[TOKEN0]]
// CHECK:     yield %[[VAL2]], %[[Y_TOKEN1]], %[[Y_TOKEN1]]
// CHECK: else
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN2:.*]] = load_ptr_tko weak {{.*}} token=%[[TOKEN0]]
// CHECK:     %[[Y_TOKEN3:.*]] = join_tokens %[[TOKEN0]], %[[Y_TOKEN2]]
// CHECK:     yield %[[VAL3]], %[[Y_TOKEN3]], %[[TOKEN0]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[TOKENTUPLE]]#2
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
// CHECK: %[[Y_TOKEN4:.*]] = join_tokens %[[TOKENTUPLE]]#1, %[[TOKENTUPLE]]#2
// CHECK: %[[Y_TOKEN5:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN4]]
"""  # noqa: E501


ReleaseIfElseCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = if
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN2]]
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko release {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     yield %[[VAL2]], %[[Y_TOKEN1]]
// CHECK: else
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN2:.*]] = load_ptr_tko weak {{.*}} token=%[[TOKEN0]]
// CHECK:     %[[Y_TOKEN3:.*]] = join_tokens %[[TOKEN0]], %[[Y_TOKEN2]]
// CHECK:     yield %[[VAL3]], %[[Y_TOKEN3]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
// CHECK: %[[Y_TOKEN4:.*]] = store_ptr_tko weak {{.*}} token=%[[TOKENTUPLE]]#1
"""  # noqa: E501


AcqRelIfElseCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:3 = if
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN2]]
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acq_rel {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     yield %[[VAL2]], %[[Y_TOKEN1]], %[[Y_TOKEN1]]
// CHECK: else
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN2:.*]] = load_ptr_tko weak {{.*}} token=%[[TOKEN0]]
// CHECK:     %[[Y_TOKEN3:.*]] = join_tokens %[[TOKEN0]], %[[Y_TOKEN2]]
// CHECK:     yield %[[VAL3]], %[[Y_TOKEN3]], %[[TOKEN0]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[TOKENTUPLE]]#2
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
// CHECK: %[[Y_TOKEN4:.*]] = join_tokens %[[TOKENTUPLE]]#1, %[[TOKENTUPLE]]#2
// CHECK: %[[Y_TOKEN5:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN4]]
"""  # noqa: E501


RelaxedForLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[LOOP_TOK:.*]] = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko relaxed {{.*}} token=%[[TKNARG0]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue %[[Y_TOKEN2]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
"""  # noqa: E501


AcquireForLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN0]], %[[TKNARG2:.*]] = %[[TOKEN0]])
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[TKNARG2]]
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acquire {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue %[[Y_TOKEN2]], %[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[TOKENTUPLE]]#1
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
"""  # noqa: E501


ReleaseForLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[LOOP_TOK:.*]] = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[X_TOKEN2]]
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko release {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue %[[Y_TOKEN2]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN2]]
"""  # noqa: E501


AcqRelForLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN0]], %[[TKNARG1:.*]] = %[[TOKEN0]])
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[X_TOKEN2]], %[[TKNARG1]]
// CHECK:     %[[VAL2:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acq_rel {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue %[[Y_TOKEN2]], %[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN3:.*]] = join_tokens %[[X_TOKEN2]], %[[TOKENTUPLE]]#1
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
"""  # noqa: E501


RelaxedWhileLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[X_TOKEN3:.*]] = atomic_rmw_tko relaxed {{.*}} token=%[[X_TOKEN2]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = loop iter_values(
// CHECK-SAME: {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     else
// CHECK:         break {{.*}}, %[[TKNARG0]]
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko relaxed {{.*}} token=%[[TKNARG0]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue {{.*}}, %[[Y_TOKEN2]]
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
"""  # noqa: E501


AcquireWhileLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[X_TOKEN3:.*]] = atomic_rmw_tko acquire {{.*}} token=%[[X_TOKEN2]]
// CHECK: %[[TOKENTUPLE:.*]]:3 = loop iter_values(
// CHECK-SAME: {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN0]], %[[TKNARG1:.*]] = %[[X_TOKEN3]])
// CHECK:     else
// CHECK:         break {{.*}}, %[[TKNARG0]], %[[TKNARG1]]
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[TKNARG1]]
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acquire {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue {{.*}}, %[[Y_TOKEN2]], %[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN4:.*]] = join_tokens %[[X_TOKEN3]], %[[TOKENTUPLE]]#2
// CHECK: %[[X_TOKEN5:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN4]]
"""  # noqa: E501


ReleaseWhileLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[X_TOKEN3:.*]] = atomic_rmw_tko release {{.*}} token=%[[X_TOKEN2]]
// CHECK: %[[TOKENTUPLE:.*]]:2 = loop iter_values(
// CHECK-SAME: {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN0]])
// CHECK:     else
// CHECK:         break {{.*}}, %[[TKNARG0]]
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[X_TOKEN3]]
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko release {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue {{.*}}, %[[Y_TOKEN2]]
// CHECK: %[[X_TOKEN4:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN3]]
"""  # noqa: E501


AcqRelWhileLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[VAL2:.*]], %[[X_TOKEN3:.*]] = atomic_rmw_tko acq_rel {{.*}} token=%[[X_TOKEN2]]
// CHECK: %[[TOKENTUPLE:.*]]:3 = loop iter_values(
// CHECK-SAME: {{.*}}, %[[TKNARG0:.*]] = %[[TOKEN0]], %[[TKNARG1:.*]] = %[[X_TOKEN3]])
// CHECK:     else
// CHECK:         break {{.*}}, %[[TKNARG0]], %[[TKNARG1]]
// CHECK:     %[[JOINT_Y_TOKEN:.*]] = join_tokens %[[TKNARG0]], %[[X_TOKEN3]], %[[TKNARG1]]
// CHECK:     %[[VAL3:.*]], %[[Y_TOKEN1:.*]] = atomic_rmw_tko acq_rel {{.*}} token=%[[JOINT_Y_TOKEN]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_ptr_tko weak {{.*}} token=%[[Y_TOKEN1]]
// CHECK:     continue {{.*}}, %[[Y_TOKEN2]], %[[Y_TOKEN1]]
// CHECK: %[[X_TOKEN4:.*]] = join_tokens %[[X_TOKEN3]], %[[TOKENTUPLE]]#2
// CHECK: %[[X_TOKEN5:.*]] = store_view_tko weak {{.*}} token = %[[X_TOKEN4]]
"""  # noqa: E501


class TestMemoryOrderMLIR(MLIRTestBase):

    def make_no_control_flow_kernel(memory_order):
        @ct.kernel
        def no_control_flow_kernel(X, Y, n: int, TILE: ct.Constant[int]):
            tx = ct.load(X, index=(0,), shape=(TILE,))
            y_scalar = ct.atomic_add(Y, 0, 1, memory_order=memory_order)
            ct.store(X, index=(0,), tile=tx + 1)
            ct.scatter(Y, 0, y_scalar + 1)

        return no_control_flow_kernel

    def make_if_else_kernel(memory_order: ct.MemoryOrder):
        @ct.kernel
        def if_else_kernel(X, Y, n: int, TILE: ct.Constant[int]):
            tx = ct.load(X, index=(0,), shape=(TILE,))
            if n:
                y_scalar = ct.atomic_add(Y, 0, 1, memory_order=memory_order)
            else:
                y_scalar = ct.gather(Y, 0)
            ct.store(X, index=(0,), tile=tx + 1)
            ct.scatter(Y, 0, y_scalar + 1)

        return if_else_kernel

    def make_for_loop_kernel(memory_order: ct.MemoryOrder):
        @ct.kernel
        def for_loop_kernel(X, Y, n: int, TILE: ct.Constant[int]):
            tx = ct.load(X, index=(0,), shape=(TILE,))
            for i in range(n):
                y_scalar = ct.atomic_add(Y, 0, 1, memory_order=memory_order)
                ct.scatter(Y, 0, y_scalar + 1)
            ct.store(X, index=(0,), tile=tx + 1)

        return for_loop_kernel

    def make_while_loop_kernel(memory_order: ct.MemoryOrder):
        @ct.kernel
        def while_loop_kernel(X, Y, n: int, TILE: ct.Constant[int]):
            tx = ct.load(X, index=(0,), shape=(TILE,))
            ct.atomic_add(X, 0, 1, memory_order=memory_order)
            i = 0
            while i < n:
                y_scalar = ct.atomic_add(Y, 0, 1, memory_order=memory_order)
                ct.scatter(Y, 0, y_scalar + 1)
                i += 1
            ct.store(X, index=(0,), tile=tx + 1)

        return while_loop_kernel

    kernel_directive_map = {
        "no_control_flow_kernel": {
            ct.MemoryOrder.RELAXED: RelaxedNoControlFlowCheckDirective,
            ct.MemoryOrder.ACQUIRE: AcquireNoControlFlowCheckDirective,
            ct.MemoryOrder.RELEASE: ReleaseNoControlFlowCheckDirective,
            ct.MemoryOrder.ACQ_REL: AcqRelNoControlFlowCheckDirective,
        },
        "if_else_kernel": {
            ct.MemoryOrder.RELAXED: RelaxedIfElseCheckDirective,
            ct.MemoryOrder.ACQUIRE: AcquireIfElseCheckDirective,
            ct.MemoryOrder.RELEASE: ReleaseIfElseCheckDirective,
            ct.MemoryOrder.ACQ_REL: AcqRelIfElseCheckDirective,
        },
        "for_loop_kernel": {
            ct.MemoryOrder.RELAXED: RelaxedForLoopCheckDirective,
            ct.MemoryOrder.ACQUIRE: AcquireForLoopCheckDirective,
            ct.MemoryOrder.RELEASE: ReleaseForLoopCheckDirective,
            ct.MemoryOrder.ACQ_REL: AcqRelForLoopCheckDirective,
        },
        "while_loop_kernel": {
            ct.MemoryOrder.RELAXED: RelaxedWhileLoopCheckDirective,
            ct.MemoryOrder.ACQUIRE: AcquireWhileLoopCheckDirective,
            ct.MemoryOrder.RELEASE: ReleaseWhileLoopCheckDirective,
            ct.MemoryOrder.ACQ_REL: AcqRelWhileLoopCheckDirective,
        }
    }

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        Y = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        n = 10
        bytecode = get_bytecode(kernel, (X, Y, n, tile_size))
        return bytecode

    @pytest.mark.parametrize("make_kernel", [
        make_no_control_flow_kernel, make_if_else_kernel,
        make_for_loop_kernel, make_while_loop_kernel])
    @pytest.mark.parametrize("memory_order", [
        ct.MemoryOrder.RELAXED, ct.MemoryOrder.ACQUIRE,
        ct.MemoryOrder.RELEASE, ct.MemoryOrder.ACQ_REL])
    @override
    def test_mlir(self, make_kernel, memory_order):
        kernel = make_kernel(memory_order)
        check_directive = self.kernel_directive_map[kernel._pyfunc.__name__][memory_order]
        super().test_mlir(kernel, check_directive)


IfElseAliasCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: {{.*}}, %[[X_TOKEN1:.*]] = load_view_tko weak {{.*}} token = %[[TOKEN0]]
// CHECK: %[[X_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]]
// CHECK: %[[JOINT_TOKEN1:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN2]]
// CHECK: %[[UNI_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN1]]
// CHECK: %[[JOINT_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[UNI_TOKEN1]]
// CHECK: %[[Y_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN2]]
// CHECK: %[[JOINT_TOKEN3:.*]] = join_tokens %[[UNI_TOKEN1]], %[[TOKEN0]], %[[Y_TOKEN1]]
// CHECK: {{.*}}, %[[UNI_TOKEN2:.*]] = load_view_tko weak {{.*}} token = %[[JOINT_TOKEN3]]
// CHECK: %[[UNI_TOKEN3:.*]] = join_tokens %[[UNI_TOKEN1]], %[[UNI_TOKEN2]]
// CHECK: %[[JOINT_TOKEN4:.*]] = join_tokens %[[X_TOKEN2]], %[[UNI_TOKEN3]]
// CHECK: %[[X_TOKEN3:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN4]]
"""  # noqa: E501


ForLoopAliasCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[X_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[Y_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[LOOP_RESULTS:.*]]:6 = for {{.*}} iter_values(
// CHECK-SAME: {{.*}}, {{.*}}, {{.*}}, %[[UNI_TKNARG0:.*]] = %[[TOKEN0]],
// CHECK-SAME: %[[Y_TKNARG0:.*]] = %[[Y_TOKEN1]], %[[Y_TKNARG1:.*]] = %[[Y_TOKEN1]])
// CHECK:     %[[JOINT_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]], %[[Y_TKNARG1]]
// CHECK:     {{.*}}, %[[UNI_TOKEN2:.*]] = load_view_tko weak {{.*}} token = %[[JOINT_TOKEN2]]
// CHECK:     %[[UNI_TOKEN3:.*]] = join_tokens %[[UNI_TKNARG0]], %[[UNI_TOKEN2]]
// CHECK:     %[[JOINT_TOKEN3:.*]] = join_tokens %[[Y_TKNARG0]], %[[UNI_TOKEN3]]
// CHECK:     %[[Y_TOKEN2:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN3]]
// CHECK:     continue  {{.*}}, {{.*}}, {{.*}}, %[[UNI_TOKEN3]], %[[Y_TOKEN2]], %[[Y_TOKEN2]]
// CHECK: %[[JOINT_TOKEN4:.*]] = join_tokens %[[LOOP_RESULTS]]#4, %[[LOOP_RESULTS]]#3
// CHECK: %[[Y_TOKEN3:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN4]]
// CHECK: %[[Z_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
"""  # noqa: E501


WhileLoopAliasCheckDirective = """
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[X_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[Y_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[JOINT_TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[X_TOKEN1]], %[[Y_TOKEN1]]
// CHECK: %[[UNI_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN2]]
// CHECK: %[[JOINT_TOKEN4:.*]] = join_tokens %[[Y_TOKEN1]], %[[UNI_TOKEN1]]
// CHECK: %[[Y_TOKEN2:.*]] = store_view_tko {{.*}} token = %[[JOINT_TOKEN4]]
"""  # noqa: E501


ControlFlowTupleAliasCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[X_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[Y_TOKEN1:.*]] = store_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[LOOP_RESULTS:.*]]:4 = for {{.*}} iter_values(
// CHECK-SAME: {{.*}}, {{.*}}, %[[Y_TKNARG0:.*]] = %[[Y_TOKEN1]], %[[Y_TKNARG1:.*]] = %[[Y_TOKEN1]])
// CHECK:     %[[IFELSE_RESULTS:.*]]:4 = if
// CHECK:         %[[Y_TOKEN2:.*]] = store_view_tko {{.*}} token = %[[Y_TKNARG0]]
// CHECK:         yield {{.*}}, {{.*}}, %[[Y_TOKEN2]], %[[Y_TOKEN2]]
// CHECK:     else
// CHECK:         %[[Y_TOKEN3:.*]] = store_view_tko {{.*}} token = %[[Y_TKNARG0]]
// CHECK:         yield {{.*}}, {{.*}}, %[[Y_TOKEN3]], %[[Y_TOKEN3]]
// CHECK:     continue {{.*}}, {{.*}}, %[[IFELSE_RESULTS]]#2, %[[IFELSE_RESULTS]]#3
// CHECK: {{.*}}, %[[Y_TOKEN4:.*]] = load_view_tko weak {{.*}} token = %[[LOOP_RESULTS]]#3
// CHECK: %[[Y_TOKEN5:.*]] = join_tokens %[[LOOP_RESULTS]]#2, %[[Y_TOKEN4]]
// CHECK: %[[Y_TOKEN6:.*]] = store_view_tko {{.*}} token = %[[Y_TOKEN5]]
// CHECK: %[[X_TOKEN2:.*]] = store_view_tko {{.*}} token = %[[X_TOKEN1]]
"""  # noqa: E501


class TestRuntimeAlias(MLIRTestBase):

    def ifelse_alias(X, Y, _Z, n: int, TILE: ct.Constant[int]):
        tx = ct.load(X, index=(0,), shape=(TILE,))
        if n:
            alias = Y
        else:
            alias = X
        ct.store(alias, index=(0,), tile=tx)
        ct.store(Y, index=(0,), tile=tx)
        ta = ct.load(alias, index=(0,), shape=(TILE,))
        ct.store(X, index=(0,), tile=ta)

    def for_loop_alias(X, Y, Z, n: int, TILE: ct.Constant[int]):
        alias = X
        ct.store(alias, index=(0,), tile=ct.full((TILE,), 1, dtype=X.dtype))
        ct.store(Y, index=(0,), tile=ct.full((TILE,), 2, dtype=X.dtype))
        ta = ct.zeros((TILE,), dtype=X.dtype)
        for i in range(n):
            alias2 = alias
            ta = ct.load(alias2, index=(i,), shape=(TILE,))
            alias = Y
            ct.store(alias, index=(i,), tile=ta)
        ct.store(Y, index=(0,), tile=ta)
        ct.store(Z, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))

    def while_loop_alias(X, Y, _Z, n: int, TILE: ct.Constant[int]):
        alias = X
        ct.store(alias, index=(0,), tile=ct.full((TILE,), 1, dtype=X.dtype))
        ct.store(Y, index=(0,), tile=ct.full((TILE,), 2, dtype=X.dtype))
        i = 0
        while True:
            if i == n:
                break
            alias = Y
            i += 1
        ct.store(alias, index=(0,), tile=ct.full((TILE,), 2, dtype=X.dtype))
        ct.store(Y, index=(0,), tile=ct.full((TILE,), 3, dtype=X.dtype))

    def control_flow_tuple_alias(X, Y, _Z, n: int, TILE: ct.Constant[int]):
        ct.store(X, index=(0,), tile=ct.full((TILE,), 1, dtype=X.dtype))
        ct.store(Y, index=(0,), tile=ct.full((TILE,), 2, dtype=X.dtype))
        alias = Y
        for i in range(n):
            if i:
                alias = (Y, X)[0]
                ct.store(alias, index=(i,), tile=ct.full((TILE,), 4, dtype=X.dtype))
            else:
                alias = Y
                ct.store(alias, index=(i,), tile=ct.full((TILE,), 5, dtype=X.dtype))
        ty = ct.load(Y, index=(0,), shape=(TILE,))
        ct.store(alias, index=(0,), tile=ty)
        ct.store(X, index=(0,), tile=ty)

    @override
    def compile_kernel(self, kernel):
        tile_size = 128
        X = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        Y = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        Z = torch.arange(tile_size, device="cuda", dtype=torch.int32)
        bytecode = get_bytecode(kernel, (X, Y, Z, 10, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (ifelse_alias, IfElseAliasCheckDirective),
        (for_loop_alias, ForLoopAliasCheckDirective),
        (while_loop_alias, WhileLoopAliasCheckDirective),
        (control_flow_tuple_alias, ControlFlowTupleAliasCheckDirective)),
    )
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)


ArraySliceLoadStoreCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TOKEN0]]
// CHECK: %[[TOKEN2:.*]] = join_tokens %[[TOKEN0]], %[[TOKEN1]]
// CHECK: %[[TOKEN3:.*]] = store_view_tko {{.*}} token = %[[TOKEN2]]
// CHECK: %[[VAL2:.*]], %[[TOKEN4:.*]] = load_view_tko {{.*}} token = %[[TOKEN3]]
// CHECK: %[[TOKEN5:.*]] = join_tokens %[[TOKEN3]], %[[TOKEN4]]
// CHECK: %[[TOKEN6:.*]] = store_view_tko {{.*}} token = %[[TOKEN5]]
"""


ArraySliceForLoopCheckDirective = """\
// CHECK: %[[TOKEN0:.*]] = make_token
// CHECK: %[[TOKENPAIR:.*]]:2 = for {{.*}} iter_values(
// CHECK-SAME: %[[TKNARG0:.*]] = %[[TOKEN0]], %[[TKNARG1:.*]] = %[[TOKEN0]])
// CHECK:     %[[VAL1:.*]], %[[TOKEN1:.*]] = load_view_tko {{.*}} token = %[[TKNARG1]]
// CHECK:     %[[TOKEN2:.*]] = join_tokens %[[TKNARG0]], %[[TOKEN1]]
// CHECK:     %[[TOKEN3:.*]] = store_view_tko {{.*}} token = %[[TOKEN2]]
// CHECK:     continue %[[TOKEN3]], %[[TOKEN3]]
// CHECK: %[[TOKEN4:.*]] = store_view_tko {{.*}} token = %[[TOKENPAIR]]#0
"""


class TestArraySliceMLIR(MLIRTestBase):

    def array_slice_load_store(X, TILE: ct.Constant[int]):
        first_half = X.slice(axis=0, start=0, stop=TILE)
        second_half = X.slice(axis=0, start=TILE, stop=2*TILE)
        tx = ct.load(first_half, index=(0,), shape=(TILE,))
        ct.store(second_half, index=(0,), tile=tx + 1)

        ty = ct.load(second_half, index=(0,), shape=(TILE,))
        ct.store(first_half, index=(0,), tile=ty)

    def array_slice_for_loop(X, n: int, TILE: ct.Constant[int]):
        for i in range(n):
            chunk = X.slice(axis=0, start=i*TILE, stop=(i+1)*TILE)
            tx = ct.load(chunk, index=(0,), shape=(TILE,))
            ct.store(chunk, index=(0,), tile=tx + 1)

        first_chunk = X.slice(axis=0, start=0, stop=TILE)
        ct.store(first_chunk, index=(0,), tile=ct.full((TILE,), 0, dtype=X.dtype))

    @override
    def compile_kernel(self, kernel):
        tile_size = 1024
        X = torch.arange(tile_size * 2, device="cuda", dtype=torch.int32)
        if kernel.__name__ == "array_slice_load_store":
            bytecode = get_bytecode(kernel, (X, tile_size))
        else:
            bytecode = get_bytecode(kernel, (X, 2, tile_size))
        return bytecode

    @pytest.mark.parametrize("kernel, check_directive", make_cases(
        (array_slice_load_store, ArraySliceLoadStoreCheckDirective),
        (array_slice_for_loop, ArraySliceForLoopCheckDirective),
    ))
    @override
    def test_mlir(self, kernel, check_directive):
        super().test_mlir(kernel, check_directive)
