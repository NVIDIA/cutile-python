# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import ast
import inspect

from cuda.tile._passes.ast_util import ast_get_all_local_names


def deco(whatever):
    def wrap(f):
        return f
    return wrap


def func(p1, *p2, **p3):
    p1 = 2
    print([i*i for i in range(10) if (local_i := i) % 2 == 0])
    del nonexistent
    a = 123

    @deco(sneaky_1 := 6)
    def nested_func(p3, p4) -> list[sneaky_2 := 7]:
        nested_var = 40
        nonlocal a
        a = 456

    async def nested_async_def(): pass

    b: list[sneaky_3 := 8]

    c: list[sneaky_4 := 9] = [10]

    global g
    g = 45

    match p1:
        case (pat1, pat2): pass
        case {"key": pat3, **pat4}: pass
        case list() as match_alias if (walrus := 10): pass
        case (pat5, *pat6): pass
        case _: pass

    try:
        from os import getcwd
        from os import chdir as cd
    except Exception as e:
        print(e)

    print(a, b, c, sneaky_1, sneaky_2, sneaky_3, sneaky_4, local_i)


def test_get_all_local_names():
    parsed_ast = ast.parse(inspect.getsource(func))
    func_ast = parsed_ast.body[0]
    local_names, global_names, nonlocal_names = ast_get_all_local_names(func_ast)
    expected = ['a', 'b', 'c', 'cd', 'e', 'getcwd', 'local_i', 'match_alias', 'nested_async_def',
                'nested_func', 'nonexistent', 'p1', 'p2', 'p3',
                'pat1', 'pat2', 'pat3', 'pat4', 'pat5', 'pat6',
                'sneaky_1', 'sneaky_2', 'sneaky_3', 'sneaky_4', 'walrus']
    assert sorted(local_names) == expected
    assert global_names == {"g"}
    assert nonlocal_names == set()
