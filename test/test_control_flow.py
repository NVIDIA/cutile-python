# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from math import ceil
import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from util import assert_equal


class TestForLoop:

    @staticmethod
    @ct.kernel
    def plus_n_one_arg(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for _ in range(n):
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    @ct.kernel
    def plus_n_two_args(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for _ in range(0, n):
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    @ct.kernel
    def plus_n_step2(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for _ in range(0, n * 2, 2):
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    @ct.kernel
    def plus_n_two_loops(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        k = 2
        for _ in range(k):
            xi += 1
        for _ in range(k, n):
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    @ct.kernel
    def plus_n_two_loops_once_each(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        xi2 = ct.load(x, index=(i,), shape=(tile,))
        for _ in range(n):
            xi += 1
        for _ in range(n):
            xi2 += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    @ct.kernel
    def plus_n_scalar_acc(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        acc = 0
        for _ in range(n):
            acc += 1
        tx = ct.full((tile,), acc, dtype=x.dtype)
        ct.store(x, index=(i,), tile=tx)

    @pytest.mark.parametrize(
        "func_name",
        [
            "plus_n_one_arg",
            "plus_n_two_args",
            "plus_n_step2",
            "plus_n_two_loops",
            "plus_n_two_loops_once_each",
            "plus_n_scalar_acc",
        ],
    )
    def test_basic_for_loop(self, func_name):
        func = getattr(self, func_name)
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        n = 5
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, func, (x, n, tile))
        ref = torch.full_like(x, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def plus_n_nested_for_loops(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            for index2 in range(n):
                xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_n_nested_for_loops_ref(x, n):
        for index in range(n):
            for index2 in range(n):
                x += 1
        return x

    @staticmethod
    @ct.kernel
    def plus_n_nested_for_loops_for_two_variables(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        xi2 = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            xi += 1
            for index2 in range(n):
                xi2 += 1
        x_out = xi + xi2
        ct.store(x, index=(i,), tile=x_out)

    @staticmethod
    def plus_n_nested_for_loops_for_two_variables_ref(x, n):
        x2 = x.clone()
        for index in range(n):
            x += 1
            for index2 in range(n):
                x2 += 1
        return x + x2

    @pytest.mark.parametrize(
        "func_name",
        [
            "plus_n_nested_for_loops",
            "plus_n_nested_for_loops_for_two_variables",
        ],
    )
    def test_nested_for_loops(self, func_name):
        func = getattr(self, func_name)
        ref_func = getattr(self, f"{func_name}_ref")
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        n = 5
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, func, (x, n, tile))
        ref = ref_func(ref, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def plus_n_until_2(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            if index == 2:
                break
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_n_until_2_ref(x, n):
        for index in range(n):
            if index == 2:
                break
            x += 1

    @pytest.mark.parametrize("n", [5, 1, 0, 2, 3])
    def test_break_in_for_loop_1(self, n):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.plus_n_until_2, (x, n, tile))
        self.plus_n_until_2_ref(ref, n)

        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def nested_cont_break_if(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            for inner in range(n):
                if inner == 2:
                    break
            if index == 2:
                continue
            elif index > 4:
                break
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def nested_cont_break_if_ref(x, n):
        for index in range(n):
            for inner in range(n):
                if inner == 2:
                    break
            if index == 2:
                continue
            elif index > 4:
                break
            x += 1

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 6, 8])
    def test_nested_break_continue_in_for_loop(self, n):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.nested_cont_break_if, (x, n, tile))
        self.nested_cont_break_if_ref(ref, n)

        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def tuple_fibonacci(x):
        t = ct.load(x, index=(0,), shape=(1,)), ct.load(x, index=(1,), shape=(1,))
        for i in range(5):
            t = t[1], t[0] + t[1]
        ct.store(x, index=(2,), tile=t[1])

    def test_tuple_carried_variable(self):
        x = torch.ones(3, dtype=torch.float32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), self.tuple_fibonacci, (x,))
        assert x.cpu().numpy()[2] == 13.0


class TestWhileLoop:
    @staticmethod
    @ct.kernel
    def plus_n(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        count = 0
        while count < n:
            xi += 1
            count += 1
        ct.store(x, index=(i,), tile=xi)

    def test_basic_while_loop(self):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        n = 5
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.plus_n, (x, n, tile))
        ref = torch.full_like(x, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def break_in_while(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        count = 0
        while count < n:
            if count == 2:
                break
            xi += 1
            count += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def break_in_while_ref(x, n):
        count = 0
        while count < n:
            if count == 2:
                break
            x += 1
            count += 1

    def test_break_in_while_loop(self):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        n = 5
        ct.launch(torch.cuda.current_stream(), grid, self.break_in_while, (x, n, tile))
        self.break_in_while_ref(ref, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def continue_in_while(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        count = 0
        while count < n:
            count += 1
            if count > 2:
                continue
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def continue_in_while_ref(x, n):
        count = 0
        while count < n:
            count += 1
            if count > 2:
                continue
            x += 1

    def test_continue_in_while_loop(self):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = x.clone()
        grid = ((N // tile), 1, 1)
        n = 5
        ct.launch(torch.cuda.current_stream(), grid, self.continue_in_while, (x, n, tile))
        self.continue_in_while_ref(ref, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def constant_assigned_inside_loop(x):
        a = ct.bid(0) + 3.0
        while ct.bid(0) == 1:
            a = 10.0  # This shouldn't be constant-propagated as the result of the loop
            break
        t = ct.full((1,), a, x.dtype)
        ct.store(x, (0,), t)

    def test_constant_assigned_inside_loop(self):
        x = torch.zeros(1, dtype=torch.float32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1,), self.constant_assigned_inside_loop, (x,))
        assert x.cpu().item() == 3.0

    @staticmethod
    @ct.kernel
    def same_constant_two_branches_inside_loop(x):
        a = 0
        while True:
            if ct.bid(0) == 0:
                a = 1
                break
            a = 1
            break
        # Use `a` as tile shape to verify that it has been inferred as constant
        t = ct.full((a,), 3.0, x.dtype)
        ct.store(x, (0,), t)

    def test_same_constant_two_branches_inside_loop(self):
        x = torch.zeros(1, dtype=torch.float32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1,),
                  self.same_constant_two_branches_inside_loop, (x,))
        assert x.cpu().item() == 3.0

    @staticmethod
    @ct.kernel
    def different_constants_two_branches_inside_loop(x):
        a = 0
        while True:
            if ct.bid(0) == 0:
                a = 1
                break
            a = 2
            break
        # This should error out because `a` is not a constant
        t = ct.full((a,), 3.0, x.dtype)
        ct.store(x, (0,), t)

    def test_different_constant_two_branches_inside_loop(self):
        x = torch.zeros(1, dtype=torch.float32, device='cuda')
        with pytest.raises(TileTypeError,
                           match='Invalid argument "shape" of full\\(\\): Expected a const'):
            ct.launch(torch.cuda.current_stream(), (1,),
                      self.different_constants_two_branches_inside_loop, (x,))

    @staticmethod
    @ct.kernel
    def break_const_value(x):
        a = 0
        while True:
            if ct.bid(0) == 0:
                # the result variable will be const
                a = 1
                break
            # even if we increment a
            # the continue should not propagate non-constness
            a += 1

        t = ct.full((a,), 1.0, ct.int32)
        ct.store(x, (0,), t)

    def test_break_with_const_value(self):
        x = torch.zeros(1, dtype=torch.int32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1,),
                  self.break_const_value, (x,))
        assert x.item() == 1.0


class TestIfCondtion:

    @staticmethod
    @ct.kernel
    def plus_one_if_true(x, condition: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_one_if_true_ref(x, condition: bool):
        if condition:
            x += 1

    @staticmethod
    @ct.kernel
    def plus_or_minus_one(x, condition: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition:
            xi += 1
        else:
            xi -= 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_or_minus_one_ref(x, condition: bool):
        if condition:
            x += 1
        else:
            x -= 1

    @pytest.mark.parametrize("func_name", ["plus_one_if_true", "plus_or_minus_one"])
    @pytest.mark.parametrize("condition", [True, False, 1, 0])
    def test_basic_if(self, func_name, condition):
        func = getattr(self, func_name)
        ref_func = getattr(self, f"{func_name}_ref")
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = x.clone()
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, func, (x, condition, tile))
        ref_func(ref, condition)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def plus_one_two_ifs(x, condition0: bool, condition1: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition0:
            xi += 1
        if condition1:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @pytest.mark.parametrize("condition0", [1, 0])
    @pytest.mark.parametrize("condition1", [1, 0])
    def test_two_ifs(self, condition0, condition1):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.plus_one_two_ifs,
                  (x, condition0, condition1, tile))
        ref += condition0 + condition1
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def plus_one_nested_ifs(x, condition0: bool, condition1: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition0:
            xi += 1
            if condition1:
                xi += 1
        else:
            xi -= 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_one_nested_ifs_ref(x, condition0: bool, condition1: bool):
        if condition0:
            x += 1
            if condition1:
                x += 1
        else:
            x -= 1

    @pytest.mark.parametrize("condition0", [True, False])
    @pytest.mark.parametrize("condition1", [True, False])
    def test_nested_ifs(self, condition0, condition1):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.plus_one_nested_ifs,
                  (x, condition0, condition1, tile))
        self.plus_one_nested_ifs_ref(ref, condition0, condition1)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def plus_one_two_conditions_and(x, condition0: bool, condition1: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition0 and condition1:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_one_two_conditions_and_ref(x, condition0: bool, condition1: bool):
        if condition0 and condition1:
            x += 1

    @staticmethod
    @ct.kernel
    def plus_one_and_in_variable(x, condition0: bool, condition1: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        cond = condition0 and condition1
        if cond:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_one_and_in_variable_ref(x, condition0: bool, condition1: bool):
        cond = condition0 and condition1
        if cond:
            x += 1

    @staticmethod
    @ct.kernel
    def plus_one_two_conditions_or(x, condition0: bool, condition1: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if condition0 or condition1:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_one_two_conditions_or_ref(x, condition0: bool, condition1: bool):
        if condition0 or condition1:
            x += 1

    @pytest.mark.parametrize(
        "func_name",
        [
            "plus_one_two_conditions_and",
            "plus_one_and_in_variable",
            "plus_one_two_conditions_or",
        ],
    )
    @pytest.mark.parametrize("condition0", [1, 0])
    @pytest.mark.parametrize("condition1", [1, 0])
    def test_if_two_conditions(self, func_name, condition0, condition1):
        func = getattr(self, func_name)
        ref_func = getattr(self, func_name + "_ref")
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, func, (x, condition0, condition1, tile))
        ref_func(ref, condition0, condition1)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def multiple_conditions_in_if(
        x, condition0: bool | int, condition1: bool | int, condition2: bool | int,
        base: int, tile: ct.Constant[int]
    ):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))

        if condition0 or condition1 or condition2:
            xi += 1
        else:
            xi -= 1

        if condition0 and condition1 and condition2:
            xi += 1

        if condition0 or condition1 and condition2 and base > 50:
            xi += 1

        if condition0 and (condition1 or condition2) or base > 50:
            xi += 1

        cond = (condition0 or condition1) and condition2
        if cond:
            xi += 1

        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def multiple_conditions_in_if_ref(
        x, condition0: bool | int, condition1: bool | int, condition2: bool | int, base: int
    ):
        if condition0 or condition1 or condition2:
            x += 1
        else:
            x -= 1

        if condition0 and condition1 and condition2:
            x += 1

        if condition0 or condition1 and condition2 and base > 50:
            x += 1

        if condition0 and (condition1 or condition2) or base > 50:
            x += 1

        cond = (condition0 or condition1) and condition2
        if cond:
            x += 1

    @pytest.mark.parametrize("condition0", [1, 0, True, False])
    @pytest.mark.parametrize("condition1", [5, 0, True, False])
    @pytest.mark.parametrize("condition2", [-5, 0, True, False])
    @pytest.mark.parametrize("base", [100, 0])
    def test_if_multiple_conditions(self, condition0, condition1, condition2, base):
        if sys.platform == "win32" and (isinstance(condition0, bool) + isinstance(condition1, bool)
                                        + isinstance(condition2, bool) >= 2):
            pytest.xfail("This results in Access Violation on Windows (tileiras bug 6039732)")

        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.multiple_conditions_in_if,
                  (x, condition0, condition1, condition2, base, tile))
        self.multiple_conditions_in_if_ref(ref, condition0, condition1, condition2, base)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def if_else_assignment(
        x, condition0: bool, tile: ct.Constant[int]
    ):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        xi = xi + 1 if condition0 else xi - 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def if_else_assignment_ref(
        x, condition0: bool
    ):
        return x + 1 if condition0 else x - 1

    @pytest.mark.parametrize("condition0", [True, False])
    def test_if_else_assignment(self, condition0):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.if_else_assignment,
                  (x, condition0, tile))
        ref = self.if_else_assignment_ref(ref, condition0)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def if_else_assignment_type_mismatch(
        x, condition0: bool, tile: ct.Constant[int]
    ):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        xi = condition0 if condition0 else xi - 1
        ct.store(x, index=(i,), tile=xi)

    @pytest.mark.parametrize("condition0", [True, False])
    def test_if_else_assignment_type_mismatch(self, condition0):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        grid = ((N // tile), 1, 1)
        with pytest.raises(TileTypeError):
            ct.launch(torch.cuda.current_stream(), grid, self.if_else_assignment_type_mismatch,
                      (x, condition0, tile))

    @staticmethod
    @ct.kernel
    def if_else_assignment_type_match(
        x, condition0: bool, tile: ct.Constant[int]
    ):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        cond = condition0 if condition0 else 100
        xi += cond
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def if_else_assignment_type_match_ref(
        x, condition0: int
    ):
        return x + condition0 if condition0 else x + 100

    @pytest.mark.parametrize("condition0", [5, 0])
    def test_if_else_assignment_type_match(self, condition0):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.if_else_assignment_type_match,
                  (x, condition0, tile))
        ref = self.if_else_assignment_type_match_ref(ref, condition0)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def chain_comparison(
        x, left, right, tile: ct.Constant[int]
    ):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if left < i < right:
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def chain_comparison_ref(
        x, left, right, N, tile
    ):
        for i in range(N):
            tile_id = i // tile
            if left < tile_id < right:
                x[i] += 1
        return x

    def test_chain_comparison(self):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        left = 0
        right = 2

        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.chain_comparison,
                  (x, left, right, tile))
        ref = self.chain_comparison_ref(ref, left, right, N, tile)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def tuple_if_else(x):
        t = ct.load(x, index=(0,), shape=(1,))
        if ct.bid(0) > 0:
            a = t, t
        else:
            a = t + 3, t + 5
        ct.store(x, index=(1,), tile=a[0])
        ct.store(x, index=(2,), tile=a[1])

    def test_if_else_tuple_result(self):
        x = torch.ones(3, dtype=torch.float32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), self.tuple_if_else, (x,))
        assert x.cpu().numpy()[1] == 4.0
        assert x.cpu().numpy()[2] == 6.0

    @staticmethod
    @ct.kernel
    def array_if_else(x, y):
        if ct.bid(0) == 0:
            a = x
        else:
            a = y
        tile = ct.full((1,), ct.bid(0) + 10, dtype=x.dtype)
        ct.store(a, index=(0,), tile=tile)

    def test_if_else_array_result(self):
        x = torch.zeros([1], dtype=torch.int32, device="cuda")
        y = torch.zeros([1], dtype=torch.int32, device="cuda")
        ct.launch(torch.cuda.current_stream(), (2,), self.array_if_else, (x, y))
        assert x.cpu().numpy()[0] == 10
        assert y.cpu().numpy()[0] == 11


class TestMixedControlFlow:

    @staticmethod
    @ct.kernel
    def plus_n_skip_2(x, n, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            if index == 2:
                continue
            xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def plus_n_skip_2_ref(x, n):
        for index in range(n):
            if index == 2:
                continue
            x += 1

    @pytest.mark.parametrize("n", [5, 1])
    def test_basic_continue(self, n):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.plus_n_skip_2, (x, n, tile))
        self.plus_n_skip_2_ref(ref, n)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def more_nested_control_flow(x, n, cond: bool, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for index in range(n):
            if cond:
                xi += 1
            for index2 in range(n):
                if index2 == 2:
                    continue
                else:
                    xi += 1
        ct.store(x, index=(i,), tile=xi)

    @staticmethod
    def more_nested_control_flow_ref(x, n, cond: bool):
        for index in range(n):
            if cond:
                x += 1
            for index2 in range(n):
                if index2 == 2:
                    continue
                else:
                    x += 1

    @pytest.mark.parametrize("n", [5, 1])
    def test_more_nested_control_flow(self, n):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        cond = False
        ct.launch(torch.cuda.current_stream(), grid, self.more_nested_control_flow,
                  (x, n, cond, tile))
        self.more_nested_control_flow_ref(ref, n, cond)
        assert_equal(x, ref)

    @staticmethod
    @ct.kernel
    def switch_cases_kernel(x, option: int, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        match option:
            case 0:
                xi = xi
            case 1:
                xi += 1
            case 2:
                xi += 2
            case 3:
                xi += 3
        ct.store(x, index=(i,), tile=xi)

    def test_switch_cases(self):
        pytest.xfail("TODO: Unsupported syntax `match`")
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        ref = torch.zeros_like(x)
        grid = ((N // tile), 1, 1)
        option = 1
        ct.launch(torch.cuda.current_stream(), grid, self.switch_cases_kernel, (x, option, tile))
        ref += 1
        assert_equal(x, ref)


class TestUndefinedVariable:

    @staticmethod
    @ct.kernel
    def valid_undefined_variable(x, y, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if i == 0:
            # acc is undefined variable, but not used after the if statement
            acc = ct.load(x, index=(i,), shape=(tile,))
            yi = acc + xi
        else:
            yi = xi
        ct.store(y, index=(i,), tile=yi)

    def test_valid_undefined_variable(self):
        N = 256
        tile = 128
        x = torch.ones(N, dtype=torch.float32, device='cuda')
        y = torch.zeros(N, dtype=torch.float32, device='cuda')
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.valid_undefined_variable, (x, y, tile))
        ref = torch.ones_like(x)
        ref[:tile] += 1
        assert_equal(y, ref)

    @staticmethod
    @ct.kernel
    def valid_undefined_variable_in_loop(x, y, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        for _ in range(10):
            if i == 0:
                # acc is undefined variable, but not used after the if statement
                acc = ct.load(x, index=(i,), shape=(tile,))
                yi = acc + xi
            else:
                yi = xi
            ct.store(y, index=(i,), tile=yi)

    def test_valid_undefined_variable_in_loop(self):
        N = 256
        tile = 128
        x = torch.ones(N, dtype=torch.float32, device='cuda')
        y = torch.zeros(N, dtype=torch.float32, device='cuda')
        grid = ((N // tile), 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, self.valid_undefined_variable, (x, y, tile))
        ref = torch.ones_like(x)
        ref[:tile] += 1
        assert_equal(y, ref)

    @staticmethod
    @ct.kernel
    def invalid_undefined_variable(x, y, tile: ct.Constant[int]):
        i = ct.bid(0)
        xi = ct.load(x, index=(i,), shape=(tile,))
        if i == 0:
            # acc is undefined variable, and is used after the if statement
            acc = ct.load(y, index=(i,), shape=(tile,))
            yi = acc + xi
        else:
            yi = xi
        yi += acc
        ct.store(y, index=(i,), tile=yi)

    def test_invalid_undefined_variable(self):
        N = 256
        tile = 128
        x = torch.zeros(N, dtype=torch.float32, device='cuda')
        y = torch.zeros(N, dtype=torch.float32, device='cuda')
        grid = ((N // tile), 1, 1)
        with pytest.raises(TileTypeError):
            ct.launch(torch.cuda.current_stream(), grid, self.invalid_undefined_variable,
                      (x, y, tile))

    @staticmethod
    @ct.kernel
    def same_constant_both_branches(x):
        if ct.bid(0) == 1:
            a = 1
        else:
            a = 1
        # Use `a` as tile shape to make sure it is inferred as constant
        t = ct.full((a,), 3.0, x.dtype)
        ct.store(x, (0,), t)

    def test_same_constant_both_branches(self):
        x = torch.zeros(1, dtype=torch.float32, device='cuda')
        ct.launch(torch.cuda.current_stream(), (1,), self.same_constant_both_branches, (x,))
        assert x.cpu().item() == 3.0

    @staticmethod
    @ct.kernel
    def different_constant_two_branches(x):
        if ct.bid(0) == 1:
            a = 1
        else:
            a = 2
        # This should raise a type error because a non-constant value is used as tile shape
        t = ct.full((a,), 3.0, x.dtype)
        ct.store(x, (0,), t)

    def test_different_constant_two_branches(self):
        x = torch.zeros(1, dtype=torch.float32, device='cuda')
        with pytest.raises(TileTypeError,
                           match='Invalid argument "shape" of full\\(\\): Expected a const'):
            ct.launch(torch.cuda.current_stream(), (1,), self.different_constant_two_branches, (x,))

    @staticmethod
    @ct.kernel
    def loop3_kernel(
        input,
        output,
    ):
        bid_d = ct.bid(0)
        bid_h = ct.bid(1)
        bid_w = ct.bid(2)
        max_val = ct.full((1, 1, 1), ct.float32("-inf"), dtype=ct.float32)

        for d_idx in range(1):
            for h_idx in range(1):
                for w_idx in range(1):
                    if d_idx >= 0 and h_idx >= 0 and w_idx >= 0:
                        val = ct.load(
                            input,
                            index=(d_idx, h_idx, w_idx),
                            shape=(1, 1, 1),
                        )
                        max_val = max(max_val, val)

        ct.store(output, index=(bid_d, bid_h, bid_w), tile=max_val)

    def test_3loops(self):
        depth, height, width = 8, 16, 16
        input = torch.randn(depth, height, width, dtype=torch.float32, device="cuda")
        output = torch.zeros_like(input)
        ct.launch(torch.cuda.current_stream(), (1,), self.loop3_kernel, (input, output))


@ct.kernel
def early_return_kernel(x, y, output,
                        B: ct.Constant[int], N: ct.Constant[int], early_return: bool):
    px = ct.bid(0)
    tile_x = ct.load(x, index=(px, 0), shape=(B, N))
    tile_y = ct.load(y, index=(px, 0), shape=(B, 1))
    if early_return:
        return
    out = tile_x + tile_y
    ct.store(output, index=(px, 0), tile=out)


@pytest.mark.parametrize("early_return", [True, False])
def test_early_return(early_return):
    shape = (512, 128)
    tile = 16
    x = torch.rand(shape, dtype=torch.float32, device="cuda")
    y = torch.rand((shape[0], 1), dtype=torch.float32, device="cuda")
    z = torch.zeros_like(x)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, early_return_kernel,
              (x, y, z, tile, shape[1], early_return))
    if early_return:
        ref_result = torch.zeros_like(x)
    else:
        ref_result = x + y
    assert_equal(z, ref_result)


def test_yield_previously_undefined_tuple_from_loop():
    @ct.kernel
    def kernel(x):
        i = 0
        while True:
            if i > 0:
                tx = ct.gather(x, ()) + i
                tx2 = (tx, tx)
                break
            i += 10
        ct.scatter(x, (), tx2[0])

    x = torch.ones((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.item() == 11
