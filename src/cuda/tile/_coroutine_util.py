# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Awaitable


# Run a coroutine using a software stack to bypass the Python's recursion limit.
# Use resume_after() to break the call chain and push a new frame to the software stack.
def run_coroutine(awaitable: Awaitable):
    ret = None
    exc_info = None
    stack = []
    with ExitStack() as es:
        try:
            stack.append(awaitable.__await__())
            while stack:
                top = stack[-1]
                try:
                    continuation = top.send(ret) if exc_info is None else top.throw(exc_info[1])
                except StopIteration as s:
                    ret = s.value
                    exc_info = None
                    stack.pop()
                except Exception:
                    ret = None
                    exc_info = sys.exc_info()
                    stack.pop()
                else:
                    ret = exc_info = None
                    stack.append(continuation.__await__())
            if exc_info is None:
                return ret
            else:
                raise exc_info[1]
        finally:
            for c in stack:
                es.callback(c.close)


# Replace `await foo()` with `await resume_after(foo())` to bypass the recursion limit.
@dataclass
class resume_after:
    awaitable: Awaitable

    def __await__(self):
        return (yield self.awaitable)
