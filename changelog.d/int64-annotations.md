<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `ct.IndexedWithInt64` annotation for array kernel parameters whose shape
  or stride values exceed the range of a 32-bit integer. Arrays without the
  annotation continue to use `int32` for shape and stride.
- New `ct.ScalarInt64` annotation that forces a scalar integer kernel parameter
  to be inferred as `int64` instead of the default `int32`.
