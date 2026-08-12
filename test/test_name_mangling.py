# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest

import cuda.tile as ct
from cuda.tile._cext import cconv_v3_enabled
from cuda.tile.compilation import (
    mangle_kernel_name, demangle_kernel_name, KernelSignature, ScalarConstraint, ArrayConstraint,
    ListConstraint, TupleConstraint, CallingConvention
)

# FIXME: import from `cuda.tile.compilation` when cconv_v3_enabled() guard is removed
from cuda.tile.compilation._signature import DataclassConstraint

from cuda.tile._datatype import (bool_, uint8, uint16, uint32, uint64, int8, int16, int32, int64,
                                 float16, float32, float64, bfloat16, tfloat32,
                                 float8_e4m3fn, float8_e5m2, float8_e8m0fnu)
from cuda.tile.compilation._name_mangling import _mangle_string, _demangle_string, _Cursor, \
    _demangle_kernel_name

_SIMPLE_2D = ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=0,
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

    # 1D array with uint32 index type
    pytest.param(
        [ArrayConstraint(float32, 1, index_dtype=uint32, stride_lower_bound_incl=0,
                         alias_groups=(), may_alias_internally=False)],
        "_A1f32_1l0_u",
        id="array_simple",
    ),

    # 1D array with int64 index type
    pytest.param(
        [ArrayConstraint(float32, 1, index_dtype=int64, stride_lower_bound_incl=0,
                         alias_groups=(), may_alias_internally=False)],
        "_A1f32_1l0_w",
        id="array_simple",
    ),

    # 3D array with stride_constant, stride_divisible_by, shape_divisible_by
    # (dims 0 and 1 share shape_divisible_by=16), stride_lower_bound_incl,
    # and base_addr_divisible_by
    pytest.param(
        [ArrayConstraint(float32, 3,
                         index_dtype=int32,
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
        [ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                         alias_groups=("x",), may_alias_internally=True),
         ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                         alias_groups=("x",), may_alias_internally=False)],
        "_A2f32_g0i_A2f32_g0",
        id="array_alias_may_alias_internally",
    ),

    # Three arrays with overlapping alias groups: first two share one group,
    # last two share another
    pytest.param(
        [ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                         alias_groups=("ab",), may_alias_internally=False),
         ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                         alias_groups=("ab", "bc"), may_alias_internally=False),
         ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
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

    # List of 2D arrays with int64 index type
    pytest.param(
        [ListConstraint(
            ArrayConstraint(float32, 2, index_dtype=int64, stride_lower_bound_incl=0,
                            alias_groups=(), may_alias_internally=False),
            alias_groups=(), elements_may_alias=False)],
        "_LA2f32_3l0_w",
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
            ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                            alias_groups=("y",), may_alias_internally=False),
            alias_groups=("x",), elements_may_alias=False),
         ListConstraint(
            ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=None,
                            alias_groups=("y",), may_alias_internally=False),
            alias_groups=("x",), elements_may_alias=False)],
        "_Lg0A2f32_g1_Lg0A2f32_g1",
        id="two_lists_with_element_and_list_alias_groups",
    ),

    # Mixed: all constraint types in a single signature
    pytest.param(
        [42,
         ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=0,
                         alias_groups=("a",), may_alias_internally=False),
         True,
         ScalarConstraint(bfloat16),
         ListConstraint(
             ArrayConstraint(int64, 3, index_dtype=int32, stride_lower_bound_incl=None,
                             alias_groups=("a",), may_alias_internally=True),
             alias_groups=(), elements_may_alias=False),
         -1.5,
         False,
         ArrayConstraint(float32, 2, index_dtype=int32, stride_lower_bound_incl=0,
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


@pytest.mark.parametrize("parameters, expected_suffix", [
    pytest.param(
        [TupleConstraint([])],
        "_T0",
        id="empty_tuple",
    ),
    pytest.param(
        [TupleConstraint([ScalarConstraint(int32)])],
        "_T1Si32",
        id="tuple_single_scalar",
    ),
    pytest.param(
        [TupleConstraint([ScalarConstraint(int32), ScalarConstraint(float32)])],
        "_T2Si32Sf32",
        id="tuple_two_scalars",
    ),
    pytest.param(
        [TupleConstraint([_SIMPLE_2D])],
        "_T1A2f32_3l0",
        id="tuple_single_array",
    ),
    pytest.param(
        [TupleConstraint([_SIMPLE_2D, ScalarConstraint(int32)])],
        "_T2A2f32_3l0Si32",
        id="tuple_mixed_array_scalar",
    ),
    pytest.param(
        [TupleConstraint([ScalarConstraint(int32), ScalarConstraint(float32)]), _SIMPLE_2D],
        "_T2Si32Sf32_A2f32_3l0",
        id="tuple_followed_by_array",
    ),

    # Nested tuple: outer tuple contains one inner empty tuple
    pytest.param(
        [TupleConstraint([TupleConstraint([])])],
        "_T1T0",
        id="tuple_nested_empty",
    ),

    # Nested tuple: outer tuple contains one inner tuple of two scalars
    pytest.param(
        [TupleConstraint([TupleConstraint([ScalarConstraint(int32), ScalarConstraint(float32)])])],
        "_T1T2Si32Sf32",
        id="tuple_nested_two_scalars",
    ),

    # Nested tuple: outer tuple mixes scalar and inner tuple
    pytest.param(
        [TupleConstraint([ScalarConstraint(int32), TupleConstraint([ScalarConstraint(float32)])])],
        "_T2Si32T1Sf32",
        id="tuple_scalar_and_nested",
    ),

    # Three levels of nesting
    pytest.param(
        [TupleConstraint([TupleConstraint([TupleConstraint([ScalarConstraint(int32)])])])],
        "_T1T1T1Si32",
        id="tuple_triple_nested",
    ),

    # Static shape is encoded as a shape-constant axis predicate and suppresses
    # redundant shape-divisibility predicates on the same axis.
    pytest.param(
        [ArrayConstraint(float32, 2,
                         index_dtype=int32,
                         stride_lower_bound_incl=0,
                         alias_groups=(),
                         may_alias_internally=False,
                         shape_constant=[16, None],
                         shape_divisible_by=[16, 1])],
        "_A2f32_1s16_3l0",
        id="array_static_shape",
    ),
])
def test_name_mangling_cutile_python_v2(parameters, expected_suffix):
    func_name = "my_kernel"
    cconv = CallingConvention.cutile_python_v2()
    sig = KernelSignature(parameters, cconv)
    expected = func_name + "_K" + cconv.code + expected_suffix
    mangled = mangle_kernel_name(func_name, sig)
    assert mangled == expected, f"Expected {expected!r}, got {mangled!r}"
    demangled_name, demangled_sig = demangle_kernel_name(mangled)
    assert demangled_name == func_name


def test_demangle_tuple_with_v1_raises():
    symbol = "my_kernel_Kt1_T1Si32"
    with pytest.raises(ValueError, match="version >= 2"):
        demangle_kernel_name(symbol)


def test_demangle_static_shape_with_v1_raises():
    symbol = "my_kernel_Kt1_A1f32_1s8l0"
    with pytest.raises(ValueError, match="version >= 2"):
        demangle_kernel_name(symbol)


@pytest.mark.parametrize("s, expected", [
    ("", "_z"),
    ("0123foobarBAZQUX456", "0123foobarBAZQUX456_z"),
    ("a_b", "a__b_z"),
    ("_.<>", "___2e_3c_3e_z"),
    ("буквы 🤔", "_u0431_u0443_u043a_u0432_u044b_20_w0001f914_z"),
    ("A_BC.DEF<G>HIJK", "A__BC_2eDEF_3cG_3eHIJ_k_z"),
    ("K", "_k_z"),  # make sure to escape "K" because "_K" is used to separate the mangled name
    ("Kk", "_kk_z"),
    ("kK", "k_k_z"),
])
def test_mangle_string(s, expected):
    mangled = _mangle_string(s)
    assert mangled == expected
    cursor = _Cursor(mangled, mangled, 0)
    demangled = _demangle_string(cursor)
    assert s == demangled
    assert cursor.remaining == ""


@dataclass(frozen=True)
class DClassTwoFields:
    x: Any
    y: Any


@dataclass(frozen=True)
class DClassOneField:
    v: Any


class First:
    @dataclass(frozen=True)
    class Second:
        v: int

    @dataclass(frozen=True)
    class Third:
        v: "First.Second"


def _make_local_dclass():
    # Defined inside a function so that __qualname__ contains "<locals>".
    @dataclass(frozen=True)
    class Local_K_Class:
        v: int
    return Local_K_Class


_LOCAL_DCLASS = _make_local_dclass()

_MOD = "test__name__mangling_z"


class MyEnum(Enum):
    OKAY = 123
    Буквы = 456


@pytest.mark.parametrize("parameters, expected_suffix", [
    # Simple dataclass: two scalar fields.
    pytest.param(
        [DataclassConstraint(DClassTwoFields, [ScalarConstraint(int32),
                                               ScalarConstraint(float32)])],
        f"_D{_MOD}DClassTwoFields_z2Si32Sf32",
        id="dataclass_simple",
    ),

    # A nested class has a dotted qualname, encoded with "_d".
    pytest.param(
        [DataclassConstraint(First.Third,
                             [DataclassConstraint(First.Second, [ScalarConstraint(int32)])])],
        f"_D{_MOD}First_2eThird_z1D{_MOD}First_2eSecond_z1Si32",
        id="dataclass_qualname_dotted",
    ),

    # Every escape of _mangle_string() in a single qualname:
    # "_"->"__", "."->"_d", "<"->"_l", ">"->"_g", "K"->"_k".
    pytest.param(
        [DataclassConstraint(_LOCAL_DCLASS, [ScalarConstraint(int32)])],
        f"_D{_MOD}__make__local__dclass_2e_3clocals_3e_2eLocal___k__Class_z1Si32",
        id="dataclass_qualname_all_escapes",
    ),

    # Array and scalar fields.
    pytest.param(
        [DataclassConstraint(DClassTwoFields, [_SIMPLE_2D, ScalarConstraint(int32)])],
        f"_D{_MOD}DClassTwoFields_z2A2f32_3l0Si32",
        id="dataclass_array_and_scalar_fields",
    ),

    # List-of-arrays and constant fields.
    pytest.param(
        [DataclassConstraint(DClassTwoFields,
                             [ListConstraint(_SIMPLE_2D, alias_groups=(),
                                             elements_may_alias=False),
                              42])],
        f"_D{_MOD}DClassTwoFields_z2LA2f32_3l0I42",
        id="dataclass_list_and_constant_fields",
    ),

    # Tuple field inside a dataclass.
    pytest.param(
        [DataclassConstraint(DClassOneField,
                             [TupleConstraint([ScalarConstraint(int32),
                                               ScalarConstraint(float32)])])],
        f"_D{_MOD}DClassOneField_z1T2Si32Sf32",
        id="dataclass_containing_tuple",
    ),

    # Dataclass field inside a dataclass.
    pytest.param(
        [DataclassConstraint(DClassOneField,
                             [DataclassConstraint(DClassOneField, [ScalarConstraint(int32)])])],
        f"_D{_MOD}DClassOneField_z1D{_MOD}DClassOneField_z1Si32",
        id="dataclass_nested",
    ),

    # Two sibling dataclasses of different types inside a tuple.
    pytest.param(
        [TupleConstraint([DataclassConstraint(DClassOneField, [ScalarConstraint(int32)]),
                          DataclassConstraint(DClassTwoFields, [ScalarConstraint(int32),
                                                                ScalarConstraint(float32)])])],
        f"_T2D{_MOD}DClassOneField_z1Si32D{_MOD}DClassTwoFields_z2Si32Sf32",
        id="tuple_of_two_dataclasses",
    ),

    # Two top-level dataclass parameters of different types.
    pytest.param(
        [DataclassConstraint(DClassOneField, [ScalarConstraint(int32)]),
         DataclassConstraint(DClassTwoFields, [ScalarConstraint(int32),
                                               ScalarConstraint(float32)])],
        f"_D{_MOD}DClassOneField_z1Si32_D{_MOD}DClassTwoFields_z2Si32Sf32",
        id="two_top_level_dataclasses",
    ),

    # Dataclass alongside other parameter kinds.
    pytest.param(
        [_SIMPLE_2D,
         DataclassConstraint(DClassOneField, [ScalarConstraint(int32)]),
         ScalarConstraint(float32)],
        f"_A2f32_3l0_D{_MOD}DClassOneField_z1Si32_Sf32",
        id="dataclass_among_other_params",
    ),

    # Enum value
    pytest.param(
        [MyEnum.OKAY, MyEnum.Буквы],
        f"_Ce_{_MOD}MyEnum_zO_kAY_z_Ce_{_MOD}MyEnum_z_u0411_u0443_u043a_u0432_u044b_z",
        id="enum_constant",
    ),

    # DType constant
    pytest.param(
        [ct.float32, ct.int32],
        "_Cd_f32_Cd_i32",
        id="dtype_constant",
    ),

    # None constant
    pytest.param(
        [None, 123, None],
        "_Cn__I123_Cn_",
        id="none_constant",
    ),

    # String constant
    pytest.param(
        ["Hello", "world!"],
        "_Cs_Hello_z_Cs_world_21_z",
        id="string_constant",
    ),
] if cconv_v3_enabled() else [])
@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_name_mangling_cutile_python_v3(parameters, expected_suffix):
    func_name = "my_kernel"
    cconv = CallingConvention.cutile_python_v3()
    sig = KernelSignature(parameters, cconv)
    expected = func_name + "_K" + cconv.code + expected_suffix
    mangled = mangle_kernel_name(func_name, sig)
    assert mangled == expected, f"Expected {expected!r}, got {mangled!r}"
    # TODO: change to public demangle_kernel_name() once cconv_v3_enabled() guard is removed
    allowed_dataclasses = [DClassTwoFields, DClassOneField,
                           First.Second, First.Third, _LOCAL_DCLASS]
    allowed_enums = [MyEnum]
    demangled_name, demangled_sig = _demangle_kernel_name(
            mangled, None, allowed_dataclasses, allowed_enums)
    assert demangled_name == func_name
    assert demangled_sig.parameters == sig.parameters


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_demangle_dataclass_with_v2_raises():
    symbol = f"my_kernel_Kt2_D{_MOD}DClassOneField_z1Si32"
    with pytest.raises(ValueError, match="version >= 3"):
        _demangle_kernel_name(symbol, None, allowed_dataclasses=[DClassOneField], allowed_enums=[])


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_demangle_dataclass_class_not_allowed_raises():
    symbol = f"my_kernel_Kt2_D{_MOD}DClassOneField_z1Si32"
    with pytest.raises(ValueError, match="not found in the 'allowed_dataclasses' list"):
        demangle_kernel_name(symbol)
