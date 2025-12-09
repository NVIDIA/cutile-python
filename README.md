<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

cuTile Python
=============

cuTile Python is a programming language for NVIDIA GPUs. The official documentation can be found
on [docs.nvidia.com](https://docs.nvidia.com/cuda/cutile-python),
or built from source located in the [docs](docs/) folder.

Installing from PyPI
--------------------
cuTile Python is published on [PyPI](https://pypi.org/) under the
[cuda-tile](https://pypi.org/project/cuda-tile/) package name and can be installed with `pip`:
```
pip install cuda-tile
```
Currently, the [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads) is required
and needs to be installed separately.

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
source venv/bin/activate
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

cuTile-Python is under Apache 2.0 license. See the [LICENSES](LICENSES/) folder for the full license text.
