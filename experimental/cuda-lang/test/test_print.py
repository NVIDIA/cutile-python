# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import subprocess
import cuda.lang as cl
import torch

sys.path.insert(0, str(Path(__file__).parent))


def run_print_scalars():

    @cl.kernel
    def kernel(A):
        # CHECK-COUNT-4: 5
        for dtype in cl.static_iter((cl.int8, cl.int16, cl.int32, cl.int64)):
            print(dtype(A[0]))

        # CHECK-COUNT-3: 5.{{0+}}
        print(cl.float16(A[0]))
        print(cl.float32(A[0]))
        print(cl.float64(A[0]))

        # CHECK: True
        print(True)

        # CHECK: False
        print(False)

        # FIXME: should be True
        # CHECK: 1
        print(cl.bool_(A[0]))

        # FIXME: should be False
        # CHECK: 0
        print(not cl.bool_(A[0]))

        # CHECK: 0x{{([0-9a-f]{16})}}
        print(A.get_base_pointer())

        # CHECK: 0x{{([0-9a-f]{16})}}
        print(f"{A.get_base_pointer()}")

        mbarriers = cl.shared_array(1, cl.mbarrier, alignment=8)

        # CHECK: smem pointer 0x{{([0-9a-f]{8})}}
        print("smem pointer", mbarriers.get_base_pointer())

        cl.mbarrier_initialize(mbarriers.get_base_pointer(), 1)

        # CHECK: mbarrier 0x{{([0-9a-f]{8})}}
        print("mbarrier", mbarriers[0])

        # CHECK: mbarrier 0x{{([0-9a-f]{8})}}
        print(f"mbarrier {mbarriers[0]}")

    A = torch.tensor([5], dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    torch.cuda.synchronize()


def run_print_vectors():

    @cl.kernel
    def kernel(A: cl.Array):
        v_dyn = A.load_element(0, count=4)

        # CHECK: <0, 1, 2, 3>
        print(v_dyn)

        # CHECK: <0, 1, 2, 3>
        print(f"{v_dyn}")

        # CHECK: (<0, 1, 2, 3>,)
        print((v_dyn,))

        # FIXME: should be <False, True, True, True>
        # CHECK: <0, 1, 1, 1>
        print(v_dyn.astype(cl.bool_))

        # CHECK: <2, 3>
        print(v_dyn[2:])

        # CHECK: <3, 2, 1, 0>
        print(v_dyn[::-1])

        # CHECK: <1, 2, 3, 4>
        print(cl.Vector(1, 2, 3, 4))

        # CHECK: <1.{{0+}}, 2.{{0+}}, 3.{{0+}}, 4.{{0+}}>
        print(cl.Vector(1, 2, 3, 4, dtype=cl.float32))

    A = torch.tensor(range(4), dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    torch.cuda.synchronize()


def test_print_kernels():
    from .util import filecheck

    result = subprocess.run([sys.executable, __file__], capture_output=True, check=True)
    filecheck(result.stdout.decode(), Path(__file__).read_text())


if __name__ == "__main__":
    run_print_scalars()
    run_print_vectors()
