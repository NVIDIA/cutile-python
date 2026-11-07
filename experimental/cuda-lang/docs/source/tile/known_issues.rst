.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0


Known Issues
============

1. FP8 Torch Tensor requires `torch>=2.10`. Older version of PyTorch does not support converting fp8
   datatype through `dlpack` protocol and will `leak memory <https://github.com/pytorch/pytorch/issues/171820>`__
   when conversion to dlpack tensor fails.
