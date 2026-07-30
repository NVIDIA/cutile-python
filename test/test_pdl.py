# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
from util import require_hopper_or_newer
from cuda.tile._bytecode import BytecodeVersion
from conftest import requires_tileiras


@require_hopper_or_newer()
@requires_tileiras(BytecodeVersion.V_13_4)
def test_pdl():
    @ct.kernel
    def producer(a, producer_out):
        bid = ct.bid(0)

        # grid_dependency_control_launch_dependents() allows consumer
        # to run everything before grid_dependency_control_wait()
        ct.grid_dependency_control_launch_dependents()

        ta = ct.load(a, index=(bid,), shape=(32,))
        for _ in range(1_000_000):
            ta += 1
        ct.store(producer_out, index=(bid,), tile=ta)

    @ct.kernel
    def consumer(b, producer_out, consumer_out):
        bid = ct.bid(0)

        tb = ct.load(b, index=(bid,), shape=(32,))
        for _ in range(1_000_000):
            tb += 1

        # everything after grid_dependency_control_wait() runs after
        # producer() finishes
        ct.grid_dependency_control_wait()

        tpo = ct.load(producer_out, index=(bid,), shape=(32,))
        ct.store(consumer_out, index=(bid,), tile=tpo + tb)

    a = torch.arange(32, dtype=torch.float32, device="cuda")
    b = torch.arange(32, dtype=torch.float32, device="cuda")
    producer_out = torch.zeros_like(a)
    consumer_out = torch.zeros_like(a)

    stream = torch.cuda.current_stream()

    # Compile and load both kernels before profiling the PDL launch
    ct.launch(stream, (1,), producer, (a, producer_out))
    ct.launch(stream, (1,), consumer, (b, producer_out, consumer_out))
    torch.cuda.synchronize()

    producer_out.zero_()
    consumer_out.zero_()

    # PDL execution
    ct.launch(stream, (1,), producer, (a, producer_out))
    ct.launch(stream, (1,), consumer, (b, producer_out, consumer_out),
              programmatic_dependent_launch=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(consumer_out, a + b + 2_000_000)
