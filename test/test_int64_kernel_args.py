# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import cuda.tile as ct
from util import assert_equal


@ct.kernel
def kernel_with_large_int(x, large_int: ct.ScalarInt64):
    bid = ct.bid(0)
    if large_int > 0:
        ct.store(x, index=(bid,), tile=1)


@pytest.mark.parametrize("large_int", [2**31, 2**40, 2**63 - 1])
def test_int64_kernel_arg_with_annotation(large_int):
    x = torch.zeros(1, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel_with_large_int, (x, large_int))
    assert_equal(x, 1)
