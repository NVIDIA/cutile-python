# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.tile.compilation import mangle_kernel_name, demangle_kernel_name
from cuda.tile.compilation import (KernelSignature, ScalarConstraint, ArrayConstraint,
                                   ListConstraint, CallingConvention)
from cuda.tile._datatype import (bool_, uint8, uint16, uint32, uint64, int8, int16, int32, int64,
                                 float16, float32, float64, bfloat16, tfloat32,
                                 float8_e4m3fn, float8_e5m2, float8_e8m0fnu)


_SIMPLE_2D = ArrayConstraint(float32, 2, stride_lower_bound_incl=0,
                             alias_groups=(), may_alias_internally=False)


@pytest.mark.parametrize("parameters, expected_suffix", [
    # All scalar dtypes
    pytest.param(
        [ScalarConstraint(bool_), ScalarConstraint(uint8), ScalarConstraint(uint16),
         ScalarConstraint(uint32), ScalarConstraint(uint64), ScalarConstraint(int8),
         ScalarConstraint(int16), ScalarConstraint(int32), ScalarConstraint(int64),
         ScalarConstraint(float16), ScalarConstraint(float32), ScalarConstraint(float64),
         ScalarConstraint(bfloat16), ScalarConstraint(tfloat32),
         ScalarConstraint(float8_e4m3fn), ScalarConstraint(float8_e5m2),
         ScalarConstraint(float8_e8m0fnu)],
        "_Sb8_Su8_Su16_Su32_Su64_Si8_Si16_Si32_Si64"
        "_Sf16_Sf32_Sf64_Sbf16_Stf32_Sf8m3fn_Sf8m2_Sf8m0fnu",
        id="scalar_all_dtypes",
    ),

    # Bool, int and float constants
    pytest.param(
        [True, False, 42, -7, 0, 3.14, -0.0, float("inf"), float("-inf"), float("nan")],
        "_B1_B0_I42_I_7_I0"
        "_F40091eb851eb851f_F8000000000000000_F7ff0000000000000_Ffff0000000000000"
        "_F7ff8000000000000",
        id="constants",
    ),

    # Simple 2D array, no special constraints
    pytest.param(
        [_SIMPLE_2D],
        "_A2f32_3l0",
        id="array_simple",
    ),

    # 3D array with stride_constant, stride_divisible_by, shape_divisible_by
    # (dims 0 and 1 share shape_divisible_by=16), stride_lower_bound_incl,
    # and base_addr_divisible_by
    pytest.param(
        [ArrayConstraint(float32, 3,
                         stride_lower_bound_incl=0,
                         alias_groups=(),
                         may_alias_internally=False,
                         stride_constant=[None, None, 1],
                         stride_divisible_by=[8, 1, 1],
                         shape_divisible_by=[16, 16, 1],
                         base_addr_divisible_by=16)],
        "_A3f32_1v8_3i16l0_4t1_p16",
        id="array_axis_predicates",
    ),

    # Two arrays sharing an alias group, one with may_alias_internally
    pytest.param(
        [ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                         alias_groups=("x",), may_alias_internally=True),
         ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                         alias_groups=("x",), may_alias_internally=False)],
        "_A2f32_g0i_A2f32_g0",
        id="array_alias_may_alias_internally",
    ),

    # Three arrays with overlapping alias groups: first two share one group,
    # last two share another
    pytest.param(
        [ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                         alias_groups=("ab",), may_alias_internally=False),
         ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                         alias_groups=("ab", "bc"), may_alias_internally=False),
         ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                         alias_groups=("bc",), may_alias_internally=False)],
        "_A2f32_g0_A2f32_g0g1_A2f32_g1",
        id="array_overlapping_alias_groups",
    ),

    # Simple list of 2D arrays
    pytest.param(
        [ListConstraint(_SIMPLE_2D, alias_groups=(), elements_may_alias=False)],
        "_LA2f32_3l0",
        id="list_simple",
    ),

    # List with elements_may_alias
    pytest.param(
        [ListConstraint(_SIMPLE_2D, alias_groups=(), elements_may_alias=True)],
        "_LiA2f32_3l0",
        id="list_elements_may_alias",
    ),

    # List with alias group and elements_may_alias
    pytest.param(
        [ListConstraint(_SIMPLE_2D, alias_groups=("y",), elements_may_alias=True),
         ListConstraint(_SIMPLE_2D, alias_groups=("y",), elements_may_alias=False)],
        "_Lg0iA2f32_3l0_Lg0A2f32_3l0",
        id="list_alias_group_elements_may_alias",
    ),

    # Two lists where each has list-level alias group "x" and element alias group "y"
    pytest.param(
        [ListConstraint(
            ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                            alias_groups=("y",), may_alias_internally=False),
            alias_groups=("x",), elements_may_alias=False),
         ListConstraint(
            ArrayConstraint(float32, 2, stride_lower_bound_incl=None,
                            alias_groups=("y",), may_alias_internally=False),
            alias_groups=("x",), elements_may_alias=False)],
        "_Lg0A2f32_g1_Lg0A2f32_g1",
        id="two_lists_with_element_and_list_alias_groups",
    ),

    # Mixed: all constraint types in a single signature
    pytest.param(
        [42,
         ArrayConstraint(float32, 2, stride_lower_bound_incl=0,
                         alias_groups=("a",), may_alias_internally=False),
         True,
         ScalarConstraint(bfloat16),
         ListConstraint(
             ArrayConstraint(int64, 3, stride_lower_bound_incl=None,
                             alias_groups=("a",), may_alias_internally=True),
             alias_groups=(), elements_may_alias=False),
         -1.5,
         False,
         ArrayConstraint(float32, 2, stride_lower_bound_incl=0,
                         alias_groups=("a",), may_alias_internally=False),
         ScalarConstraint(int64),
         0],
        "_I42_A2f32_3l0_g0_B1_Sbf16_LA3i64_g0i_Fbff8000000000000_B0_A2f32_3l0_g0_Si64_I0",
        id="mixed",
    ),
])
def test_name_mangling_cutile_python_v1(parameters, expected_suffix):
    func_name = "my_kernel"
    cconv = CallingConvention.cutile_python_v1()
    sig = KernelSignature(parameters, cconv)
    expected = func_name + "_K" + cconv.code + expected_suffix
    # mangle_kernel_name internally round-trips through demangle and asserts
    # equality, so we only need to check the mangled string here.
    mangled = mangle_kernel_name(func_name, sig)
    assert mangled == expected, f"Expected {expected!r}, got {mangled!r}"
    # Also verify that the public demangle_kernel_name doesn't crash.
    demangled_name, demangled_sig = demangle_kernel_name(mangled)
    assert demangled_name == func_name
