<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added `ct.assume_divisible_by(x, divisor)`, a compiler hint that declares an integer scalar to be divisible by a constant. The divisibility fact is propagated through arithmetic, e.g., allowing the compiler to prove alignment for derived indices and pointer offsets to emit wider memory operations than it could with unknown divisibility.
