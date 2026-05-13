# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import Var


class AggregateValue:
    def as_tuple(self) -> tuple["Var", ...]:
        raise NotImplementedError()
