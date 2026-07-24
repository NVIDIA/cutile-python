# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._execution import static_def


@static_def
def static_inspect(*args, **kwargs):
    breakpoint()


@static_def
def static_print(*args, **kwargs):
    print(*args, **kwargs)
