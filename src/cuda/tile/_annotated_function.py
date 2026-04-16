# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from types import FunctionType
from typing import (get_origin, get_args, Annotated, Any, Sequence)

from cuda.tile._stub import ConstantAnnotation, ArrayAnnotation, ScalarAnnotation, ListAnnotation
from cuda.tile._datatype import int64


@dataclass
class AnnotatedFunction:
    pyfunc: FunctionType
    pysig: inspect.Signature
    constant_parameter_mask: Sequence[bool]
    # array index dtype and scalar integer dtype can only be int64 or int32 now.
    int64_index_parameter_mask: Sequence[bool]
    int64_parameter_mask: Sequence[bool]


def get_annotated_function(pyfunc: FunctionType) -> AnnotatedFunction:
    sig = inspect.signature(pyfunc)
    constant_parameter_mask = tuple(_has_constant_annotation(param.annotation)
                                    for param in sig.parameters.values())
    int64_index_parameter_mask = tuple(_has_int64_index_annotation(param.annotation)
                                       for param in sig.parameters.values())
    int64_parameter_mask = tuple(_has_int64_annotation(param.annotation)
                                 for param in sig.parameters.values())
    return AnnotatedFunction(pyfunc=pyfunc,
                             pysig=sig,
                             constant_parameter_mask=constant_parameter_mask,
                             int64_index_parameter_mask=int64_index_parameter_mask,
                             int64_parameter_mask=int64_parameter_mask)


def _has_constant_annotation(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
        return any(isinstance(m, ConstantAnnotation) for m in metadata)
    return False


def _has_int64_index_annotation(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
        for m in metadata:
            if isinstance(m, ArrayAnnotation) and m.index_dtype is int64:
                return True
            if (isinstance(m, ListAnnotation)
                    and isinstance(m.element, ArrayAnnotation)
                    and m.element.index_dtype is int64):
                return True
        return False
    return False


def _has_int64_annotation(annotation: Any) -> bool:
    if get_origin(annotation) is Annotated:
        _, *metadata = get_args(annotation)
        return any(isinstance(m, ScalarAnnotation) and m.dtype is int64 for m in metadata)
    return False
