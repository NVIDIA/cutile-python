<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Style Guide

- Avoid abbreviations, especially in user-facing APIs.
    Permitted abbreviations are included in the glossary section.

# Glossary

These terms and abbreviations are permitted in `cuda.lang` source code.

- TMA: Tensor Memory Accelerator
- CTA: cooperative thread array
- MMA: matrix multiply-accumulate
- SIMT: same instruction, multiple data
- API: application programming interface
- PTX: [Parallel Thread Execution](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) instruction set.
- IR: Intermediate representation.
- NVVM: [Intermediate representation based on llvm IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html)
- dtype: data type
- init: initialize
- tcgen05: 5th generation Tensor Core
- mbarrier: [A barrier created in shared memory.](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520bulk%2520tensor#parallel-synchronization-and-communication-instructions-mbarrier)
- src: source
- dst: destination
