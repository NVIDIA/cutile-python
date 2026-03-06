<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Add `ct.float8_e8m0fnu` dtype (8-bit, unsigned, 8 exponent bits, 0 mantissa bits). A restricted float type.
- Add `ct.float4_e2m1fn` dtype (4-bit, 1 sign bit, 2 exponent bits, 1 mantissa bit). A restricted float type.
- Compiling float8_e8m0fnu, float4_e2m1fn operations for SM80 family, or SM90 family will raise `TileUnsupportedFeatureError`
