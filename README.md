<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

cuTile Python
=============

cuTile Python is a programming language for NVIDIA GPUs. The official documentation can be found
on [docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python),
or built from source located in the [docs](docs/) folder.


Example
-------
```python
# This examples uses CuPy which can be installed via `pip install cupy-cuda13x`
# Make sure cuda toolkit 13.1+ is installed: https://developer.nvidia.com/cuda-downloads

import cuda.tile as ct
import cupy
import numpy as np

TILE_SIZE = 16

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# Generate input arrays
rng = cupy.random.default_rng()
a = rng.random(128)
b = rng.random(128)
expected = cupy.asnumpy(a) + cupy.asnumpy(b)

# Allocate an output array and launch the kernel
result = cupy.zeros_like(a)
grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))

# Verify the results
result_np = cupy.asnumpy(result)
np.testing.assert_array_almost_equal(result_np, expected)
```

More examples can be found at [Samples](samples/) and [TileGym](https://github.com/NVIDIA/TileGym).

System Requirements
-------------------
cuTile Python generates kernels based on [Tile IR](https://docs.nvidia.com/cuda/tile-ir/)
which requries NVIDIA Driver r580 or later to run.
Furthermore, the `tileiras` compiler only supports Blackwell GPU with 13.1 release, but the
restriction will be removed in the coming versions.
Checkout the [prerequisites](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites)
for full list of requirements.


Installing from PyPI
--------------------
cuTile Python is published on [PyPI](https://pypi.org/) under the
[cuda-tile](https://pypi.org/project/cuda-tile/) package name and can be installed with `pip`:
```
pip install cuda-tile
```
Currently, the [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads) is required
and needs to be installed separately. On a Debian-based system, use `apt-get install
cuda-tileiras-13.1 cuda-compiler-13.1` instead of `apt-get install cuda-toolkit-13.1`
if you wish to avoid installing the full CUDA Toolkit.


Building from Source
--------------------
cuTile is written mostly in Python, but includes a C++ extension which needs to be built.
You will need:
- A C++17-capable compiler, such as GNU C++ or MSVC;
- CMake 3.18+;
- GNU Make on Linux or msbuild on Windows;
- Python 3.10+ with development headers (`venv` module is recommended but optional);
- [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads)

On an Ubuntu system, the first four dependencies can be installed with APT:
```
sudo apt-get update && sudo apt-get install build-essential cmake python3-dev python3-venv
```

The CMakeLists.txt script will also automatically download
the [DLPack](https://github.com/dmlc/dlpack) dependency from GitHub.
If you wish to disable this behavior and provide your own copy of DLPack,
set the `CUDA_TILE_CMAKE_DLPACK_PATH` environment variable to a local path
to the DLPack source tree.

Unless you are already using a Python virtual environment, it is recommended to create one
in order to avoid installing cuTile globally:

```
python3 -m venv env
source env/bin/activate
```

Once the build dependencies are in place, the simplest way to build cuTile is to install it
in editable mode by running the following command in the source root directory:

```
pip install -e .
```

This will create the `build` directory and invoke the CMake-based build process.
In editable mode, the compiled extension module will be placed in the build directory,
and then a symbolic link to it will be created in the source directory.
This makes sure that the `pip install -e .` command above is needed only once, and recompiling
the extension after making changes to the C++ code can be done with `make -C build`
which is much faster. This logic is defined in [setup.py](./setup.py).

Experimental Features (Optional)
--------------------------------
cuTile now provides an experimental package containing APIs that are still under active development.
These are **not** part of the stable `cuda.tile` API and may change.

To enable the experimental features when working from a source checkout, install the experimental
package from the repository root:
```
pip install ./experimental
```

You can also install it directly from a GitHub repository subdirectory:
```
pip install \
  "git+https://github.com/NVIDIA/cutile-python.git#egg=cuda-tile-experimental&subdirectory=experimental"
```

For example, this will make the experimental namespace available for autotuner:
```
from cuda.tile_experimental import autotune_launch, clear_autotune_cache
```

Running Tests
-------------
cuTile uses the [pytest](https://pytest.org) framework for testing.
Tests have extra dependencies, such as PyTorch, which can be installed with
```
pip install -r test/requirements.txt
```

The tests are located in the [test/](test/) directory. To run a specific test file,
for example `test_copy.py`, use the following command:
```
pytest test/test_copy.py
```

Copyright and License Information
---------------------------------
Copyright © 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

cuTile-Python is licensed under the Apache 2.0 license. See the [LICENSES](LICENSES/) folder for the full license text.
