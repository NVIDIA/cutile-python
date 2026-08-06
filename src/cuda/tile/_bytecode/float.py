# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
import math
import struct
from typing import NamedTuple

from .type import SimpleType


class NonFiniteBehavior(Enum):
    IEEE = auto()
    NanOnlyAllOnes = auto()
    FiniteOnly = auto()


class _FloatSpec(NamedTuple):
    bitwidth: int
    emin: int
    emax: int
    exp_bits: int
    precision: int
    non_finite_behavior: NonFiniteBehavior = NonFiniteBehavior.IEEE
    have_sign: bool = True
    have_zero_and_subnormals: bool = True


_specs = {
    SimpleType.F16: _FloatSpec(16, -14, 15, 5, 10),
    SimpleType.BF16: _FloatSpec(16, -126, 127, 8, 7),
    SimpleType.F32: _FloatSpec(32, -126, 127, 8, 23),
    SimpleType.TF32: _FloatSpec(19, -126, 127, 8, 10),
    SimpleType.F8E4M3FN: _FloatSpec(8, -6, 8, 4, 3, NonFiniteBehavior.NanOnlyAllOnes),
    SimpleType.F8E5M2: _FloatSpec(8, -14, 15, 5, 2),
    SimpleType.F8E8M0FNU: _FloatSpec(8, -127, 127, 8, 0, NonFiniteBehavior.NanOnlyAllOnes,
                                     have_sign=False,
                                     have_zero_and_subnormals=False),
    SimpleType.F4E2M1FN: _FloatSpec(4, 0, 2, 2, 1, NonFiniteBehavior.FiniteOnly),
    SimpleType.F8E5M3FNU: _FloatSpec(8, -14, 16, 5, 3, NonFiniteBehavior.NanOnlyAllOnes,
                                     have_sign=False),
}


def float_bit_size(ty: SimpleType) -> int:
    return 64 if ty == SimpleType.F64 else _specs[ty].bitwidth


def float_to_bits(val: float, ty: SimpleType) -> int:
    if ty == SimpleType.F64:
        return struct.unpack("<Q", struct.pack("<d", val))[0]

    spec = _specs[ty]
    if spec.have_zero_and_subnormals:
        return _convert_float(val, *spec)
    else:
        assert not spec.have_sign and not spec.have_zero_and_subnormals
        assert spec.precision == 0 and spec.exp_bits == spec.bitwidth
        return _convert_fXeXm0fnu(val, spec.bitwidth, spec.emin, spec.emax)


def float_from_bits(bits: int, ty: SimpleType) -> float:
    if ty != SimpleType.F64:
        spec = _specs[ty]

        # Extract mantissa & exponent
        m_mask = ~(-1 << spec.precision)
        m = bits & m_mask
        e_mask = ~(-1 << spec.exp_bits)
        e = (bits >> spec.precision) & e_mask

        # Leave only the sign bit
        bits = (bits >> (spec.bitwidth - 1) if spec.have_sign else 0) << 63

        if e == e_mask and (spec.non_finite_behavior == NonFiniteBehavior.IEEE
                            or (spec.non_finite_behavior == NonFiniteBehavior.NanOnlyAllOnes
                                and m == m_mask)):
            e = 0x7ff
            if spec.precision == 0:
                bits |= 1 << 51
        elif e == 0 and spec.have_zero_and_subnormals:
            if m != 0:
                # Subnormal number becomes normal
                shift = spec.precision - m.bit_length() + 1
                m = (m << shift) & m_mask
                e = spec.emin + 1023 - shift
        else:
            e = (e - spec.have_zero_and_subnormals) + spec.emin + 1023

        bits |= (e << 52) | (m << (52 - spec.precision))

    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _convert_fXeXm0fnu(val: float, bitwidth: int, emin: int, emax: int) -> int:
    nan = (1 << bitwidth) - 1  # NaN is encoded as all ones
    smallest_representable = 0  # 2^(-127)
    bias = -emin  # no subnormals

    if math.copysign(1.0, val) < 0:
        raise ValueError("Negative values cannot be represented in an unsigned float format")

    if val == 0.0:
        return smallest_representable
    if not math.isfinite(val):
        return nan

    m, e = math.frexp(val)
    m *= 2   # [1.0, 2.0)
    e -= 1   # val = m * 2^e

    if e > emax:
        return nan
    if e < emin:
        return smallest_representable

    m -= 1.0
    e += bias

    # With 0 mantissa bits, the only implicit significand value is 1 (odd),
    # so round to nearest even (RNE) ties always round up.
    round_up = (m >= 0.5)
    if round_up:
        m = 0
        e += 1
        if e > emax + bias:
            return nan

    return e


def _convert_float(val: float,
                   bitwidth: int,
                   emin: int,
                   emax: int,
                   exp_bits: int,
                   precision: int,
                   non_finite_behavior: NonFiniteBehavior,
                   have_sign: bool,
                   _have_zero_and_subnormals: bool) -> int:

    if not have_sign and math.copysign(1.0, val) < 0:
        raise ValueError("Negative values cannot be represented in an unsigned float format")

    if val == 0.0:
        sign = math.copysign(1.0, val) < 0.0
        return sign << (bitwidth - 1)
    elif not math.isfinite(val):
        return _convert_nonfinite(val, bitwidth, exp_bits, precision, non_finite_behavior,
                                  have_sign)

    sign, val = (1, -val) if (val < 0) else (0, val)
    m, e = math.frexp(val)
    m *= 2   # [1, 2)
    e -= 1

    if e > emax:
        return _convert_nonfinite(-math.inf if sign else math.inf,
                                  bitwidth, exp_bits, precision, non_finite_behavior, have_sign)

    if e < emin:
        m = math.ldexp(m, e - emin)
        e = 0
    else:
        m -= 1.0
        e += -emin + 1

    # Round to nearest, ties to even (RNE)
    # The following RNE implementation breaks when precision is 0
    assert precision > 0
    m = round(m * (1 << precision))
    if m == (1 << precision):
        m = 0
        e += 1
        if e > emax - emin + 1:
            return _convert_nonfinite(-math.inf if sign else math.inf,
                                      bitwidth, exp_bits, precision, non_finite_behavior,
                                      have_sign)
    bits = (sign << (bitwidth - 1)) | (e << precision) | m
    return bits


def _convert_nonfinite(val, bitwidth, exp_bits, precision, non_finite_behavior, have_sign) -> int:
    match non_finite_behavior:
        case NonFiniteBehavior.NanOnlyAllOnes:
            if not have_sign:
                return (1 << bitwidth) - 1
            # NaN is encoded as all ones
            sign = math.copysign(1.0, val) < 0.0
            return (sign << (bitwidth - 1)) | ((1 << (bitwidth - 1)) - 1)
        case NonFiniteBehavior.FiniteOnly:
            if math.isnan(val):
                raise ValueError("NaN cannot be represented in a finite-only float format")
            # Clamp to max representable magnitude, preserve sign
            sign = math.copysign(1.0, val) < 0.0
            return (sign << (bitwidth - 1)) | ((1 << (bitwidth - 1)) - 1)
        case NonFiniteBehavior.IEEE:
            float64_bits, = struct.unpack("<Q", struct.pack("<d", val))

            # Exponent is all ones
            hi_bits = (float64_bits >> (63 - exp_bits)) << precision

            if math.isnan(val) and (float64_bits & (1 << 51)) == 0:
                # Handles signaling NaN sepcifically since truncating the low bits may cut off
                # all the 1 bits and turn it into infinity
                precision_bits = 1
            else:
                # Truncate the low bits, preserve the rest of the payload
                precision_bits = (float64_bits >> (52 - precision)) & ((1 << precision) - 1)
            return hi_bits | precision_bits
        case _:
            assert False
