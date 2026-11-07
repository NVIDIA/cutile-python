# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
import cuda.lang  # noqa: F401
import cuda.tile._datatype

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'CUDA Languages'
copyright = '2026, NVIDIA Corporation'
author = 'NVIDIA Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath('_ext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # for google style support
    'myst_parser',  # for markdown support
    'doctest_ext',
]

templates_path = ['_templates']
exclude_patterns = ['references.rst', 'stubs', 'generated/includes']

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'
toc_object_entries = False  # Don't include object entries in the TOC

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sphinx = False

# Configure sidebar depth and content
html_theme_options = {
    "navigation_depth": 2,
    "show_nav_level": 2,
    "show_toc_level": 4
}

# Set up the sidebar to use our custom global TOC
html_sidebars = {
    '**': ['globaltoc.html', 'searchbox.html'],
}

# Doc testing
doctest_test_doctest_blocks = ""
doctest_global_setup = """
import torch
from cuda.tile._cext import get_compute_capability

torch.cuda.set_device(0)


def is_blackwell_or_newer():
    return get_compute_capability()[0] >= 10
"""

# -- Generated content --------------------------------------------------------
# Make sure the generated includes directory exists
generated_includes_dir = os.path.join(os.path.dirname(__file__), 'generated', 'includes')
os.makedirs(generated_includes_dir, exist_ok=True)

for include_name in ("dtype_promotion_table", "numeric_dtypes"):
    if include_name == "numeric_dtypes":
        content = cuda.tile._datatype._generate_rst_numeric_dtypes(
            namespaces=("cuda.lang", "cuda.tile")
        )
    else:
        content = cuda.tile._datatype._generate_rst_dtype_promotion_table()
        content = re.sub(r"``(\w+)``", r"``cl.\1``", content)
    with open(os.path.join(generated_includes_dir, f"{include_name}.rst"), "w") as f:
        f.write(content)


# Include substitutions from references.rst in all documents (including docstrings)
with open(os.path.join(os.path.dirname(__file__), 'references.rst'), 'r') as f:
    rst_prolog = f.read()

# Don't expand type aliases. See https://github.com/sphinx-doc/sphinx/issues/10785
autodoc_type_aliases = {
    'Constant': 'Constant',
    'Shape': 'Shape',
}

_DTYPE_REPR_RE = re.compile(r"<DType '([^']+)'>")


def format_dtype_signature(app, what, name, obj, options, signature, return_annotation):
    """Render dtype singleton annotations as their public dtype names."""
    if signature is not None:
        signature = _DTYPE_REPR_RE.sub(r"\1", signature)
    if return_annotation is not None:
        return_annotation = _DTYPE_REPR_RE.sub(r"\1", return_annotation)
    return signature, return_annotation


def use_tile_doctest_templates(app, what, name, obj, options, lines):
    """Use cuTile's launch wrappers for examples in its API docstrings."""
    if name.startswith("cuda.tile."):
        for i, line in enumerate(lines):
            lines[i] = re.sub(r"(:template:\s+)(\w+\.py)", r"\1tile/\2", line)


def setup(app):
    app.connect("autodoc-process-docstring", use_tile_doctest_templates)
    app.connect("autodoc-process-signature", format_dtype_signature)
