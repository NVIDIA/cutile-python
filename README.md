<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# cuTile Python

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![PyPI version](https://img.shields.io/pypi/v/cuda-tile)]() [![License](https://img.shields.io/badge/license-Apache%202.0-blue)]() ![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

[Prerequisites](#prerequisites) • [Installation](#installation) • [Quickstart Example](#quickstart-example) • [Running Samples](#running-samples) • [Building from Source](#building-from-source) • [Community & Feedback](#community--feedback) • [License](#license)

<br />

cuTile Python makes it much easier to create fast programs for NVIDIA GPUs. Instead of managing complex hardware details like memory and individual processing threads, you can write code that works with arrays (groups of data), similar to how you write standard Python. This lets you focus on *what* needs to be calculated, rather than *how* the hardware does it.

It is part of the **CUDA Tile** ecosystem, which translates your code into a standard format that helps it run efficiently and enables new types of algorithms.

<br />

## Prerequisites

cuTile Python requires the following environment:

* **OS**: Linux x86_64, Linux aarch64, or Windows x86_64
* **GPU**: Compute capability 10.x or 12.x
* **Driver**: NVIDIA Driver r580 or later
* **Toolkit**: CUDA Toolkit 13.1 or later
* **Python**: Version 3.10, 3.11, 3.12, or 3.13

## Installation

With the [prerequisites](#prerequisites) met, install cuTile Python via pip:

```bash
pip install cuda-tile
```

### Optional Packages
To run the quickstart examples and samples, you will need additional packages:

```bash
# For the Quickstart example (requires CuPy for CUDA 13.x)
pip install cupy-cuda13x

# For the full samples suite
pip install pytest numpy
```

*For PyTorch installation instructions, see [pytorch.org](https://pytorch.org/get-started/locally/).*

## Quickstart Example

The following example shows **vector addition**, a typical first kernel for CUDA, but uses cuTile for tile-based programming.

**How it works:**
1.  **Load**: The kernel loads tiles from vectors `a` and `b` into `a_tile` and `b_tile`.
2.  **Compute**: These tiles are added element-wise to form `result`.
3.  **Store**: The `result` tile is written out to vector `c`.

```python
import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid, ), tile=result)

def test():
    # Create input data
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,  # 1D grid of processors
              vector_add,
              (a, b, c, tile_size))

    # Copy to host only to compare
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ vector_add_example passed!")

if __name__ == "__main__":
    test()
```

Run this script from the command line:
```bash
$ python3 VectorAdd_quickstart.py
✓ vector_add_example passed!
```

**[📚 See more details in the Official Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)**

<br>

For a visual guide on how to use cuTile, watch the deep dive below:

[![Deep Dive: How to Use cuTile Python](https://img.youtube.com/vi/YFrP03KuMZ8/0.jpg)](https://www.youtube.com/watch?v=YFrP03KuMZ8)

## Running Samples

The repository contains additional samples (FFT, Matrix Multiplication, etc.) in the `samples/` directory.

To run a specific sample:
```bash
python3 samples/FFT.py
```

To run the full suite of samples using `pytest`:
```bash
pytest samples
```

## Building from Source

If you want to contribute or require the latest bleeding-edge features, you can build cuTile from source.

**Build Requirements:**
* C++17-capable compiler (GNU C++ or MSVC)
* CMake 3.18+
* GNU Make (Linux) or msbuild (Windows)
* Python 3.10+ (with `python3-dev`)
* CUDA Toolkit 13.1+

**Steps:**

1.  **Install Dependencies (Ubuntu example):**
    ```bash
    sudo apt-get update && sudo apt-get install build-essential cmake python3-dev python3-venv
    ```
    *Note: The build script automatically downloads the [DLPack](https://github.com/dmlc/dlpack) dependency. To use your own, set `CUDA_TILE_CMAKE_DLPACK_PATH`.*

2.  **Set up Virtual Environment (Recommended):**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Build and Install:**
    Run the following in the source root to install in **editable mode**:
    ```bash
    pip install -e .
    ```
    *Note: Recompiling the C++ extension after changes can be done quickly via `make -C build`.*

## Community & Feedback

**This is just the beginning.** We are actively seeking feedback from developers, researchers, and compiler engineers.

* **Have a novel algorithm?** We want to see what you build with cuTile.
* **Building a compiler?** Tell us how you are targeting Tile IR.
* **Found a bug, feature request, or doc update?** Please file an issue using our **[GitHub Issue Templates](https://github.com/nvidia/cutile-python/issues/new/choose)**.

**Contributing**
We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit code, report bugs, and propose new features.

## License

Copyright © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

cuTile-Python is licensed under the **Apache 2.0** license. See the [LICENSES](LICENSES/Apache-2.0.txt) folder for the full license text.