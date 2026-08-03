# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir


def test_addf_printing():
    f32t = mlir.Float32Type()
    lhs, rhs = mlir.Value(f32t), mlir.Value(f32t)
    lhs.value_id = "x"
    rhs.value_id = "y"
    with mlir.Block().append_here() as block:
        mlir.arith.add_AddFOp(lhs=lhs, rhs=rhs)
    op = block[-1]
    expected = (
        '%0 = "arith.addf"(%x, %y)'
        " <{fastmath = #arith<fastmath <none>>}> : (f32, f32) -> f32"
    )
    assert str(op) == expected


def test_cond_br_printing():
    i1t = mlir.IntegerType.signless(1)
    f32t = mlir.Float32Type()
    cond = mlir.Value(i1t)
    cond.value_id = "c"
    x = mlir.Value(f32t)
    x.value_id = "x"
    y = mlir.Value(f32t)
    y.value_id = "y"
    true_label = mlir.BlockLabel("foo")
    false_label = mlir.BlockLabel("bar")
    with mlir.Block().append_here() as block:
        mlir.cf.add_CondBranchOp(
            condition=cond,
            trueDestOperands=[x, y],
            falseDestOperands=[x],
            trueDest=true_label,
            falseDest=false_label,
        )
    op = block[-1]
    expected = (
        '"cf.cond_br"(%c, %x, %y, %x) [^foo, ^bar]'
        " <{operandSegmentSizes = array<i32: 1, 2, 1>}> : (i1, f32, f32, f32) -> ()"
    )
    assert str(op) == expected


def test_operation_location_printing():
    location = mlir.FileLineColRange(
        filename=mlir.StringAttr(value="kernel.py"),
        start_line=2,
        start_column=3,
        end_line=2,
        end_column=3,
    )
    with mlir.Block().append_here() as block:
        with mlir.use_location(location):
            mlir.arith.add_ConstantOp(
                value=mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 1)
            )

    assert str(block[-1]).endswith(' : () -> i32 loc("kernel.py":2:3)')


def test_operation_location_context_is_nested():
    outer = mlir.FileLineColRange(
        filename=mlir.StringAttr(value="kernel.py"),
        start_line=2,
        start_column=3,
        end_line=2,
        end_column=3,
    )
    inner = mlir.FileLineColRange(
        filename=mlir.StringAttr(value="helper.py"),
        start_line=4,
        start_column=5,
        end_line=4,
        end_column=5,
    )
    with mlir.Block().append_here() as block:
        with mlir.use_location(outer):
            mlir.arith.add_ConstantOp(
                value=mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 1)
            )
            with mlir.use_location(inner):
                mlir.arith.add_ConstantOp(
                    value=mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 2)
                )
            mlir.arith.add_ConstantOp(
                value=mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 3)
            )
        mlir.arith.add_ConstantOp(
            value=mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 4)
        )

    assert [operation.location for operation in block.operations] == [
        outer,
        inner,
        outer,
        None,
    ]
