# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Source-compatible Kaiming initialization for BF16 batched-GEMM inputs."""

import math

import torch


def kaiming_uniform_tensor(
    shape: tuple[int, ...],
    *,
    fan_in: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Kaiming-uniform init: symmetric bound based on fan-in (1/sqrt(fan_in))."""
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    bound = 1.0 / math.sqrt(fan_in)
    tensor = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
        -bound, bound
    )
    return tensor.to(dtype)
