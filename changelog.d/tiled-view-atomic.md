<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `TiledView.atomic_add`, `TiledView.atomic_max`, `TiledView.atomic_min`,
  `TiledView.atomic_and`, `TiledView.atomic_or`, and `TiledView.atomic_xor`
  methods for performing element-wise atomic read-modify-write operations on
  a tiled view at a given tile index.
