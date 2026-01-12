#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

ignore_files=("src/cuda/tile/VERSION")
outputs=$(reuse lint --lines | grep -v ${ignore_files[@]/#/-e })
if [ -n "$outputs" ]; then
  echo -e "License check failed\n${outputs}"
  exit 1
fi

