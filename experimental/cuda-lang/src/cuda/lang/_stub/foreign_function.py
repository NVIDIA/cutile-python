# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import simt_stub
import cuda.lang._datatype as datatype


@simt_stub
def _call_foreign_function(
    func: str, return_type: datatype.TypeSpec, parameters: tuple[datatype.TypeSpec]
):
    '''
    Gererate a call to a foreign function.

    Args:
        func: The name of the foreign function to call.
        return_type: The type of the return value or ``None`` if the function returns void.
        parameters: The parameters of the function or an empty tuple.

    Returns:
        None if the function returns void, otherwise the return value.

    .. note::

        The types of the parameters will be used directly.
        It is the caller's responsibility to ensure that the types are correct.

    Examples:

    .. testcode::
        :template: kernel_wrapper.py

        abs_value = _call_foreign_function(
            "__nv_abs",
            cl.int32,
            (-5,),
        )
        print(abs_value)

    .. testoutput::
        5
    '''
