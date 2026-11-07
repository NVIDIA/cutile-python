# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import shutil
import cuda.tile
import cuda.tile._datatype

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'cuTile Python'
copyright = '2026, NVIDIA Corporation'
author = 'NVIDIA Corporation'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath('_ext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # for google style support
    'doctest_ext',
    'myst_parser',  # for markdown support
]

myst_enable_extensions = [
    'attrs_block',
]

templates_path = ['_templates']
exclude_patterns = ['references.rst', 'stubs', 'generated/includes']

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
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

# Copy release notes from root
release_notes_src = os.path.join(os.path.dirname(__file__), '..', '..', "CHANGELOG.md")
release_notes_dst = os.path.join(os.path.dirname(__file__), 'generated', "release_notes.md")
shutil.copy(release_notes_src, release_notes_dst)

# List of RST include generation functions to run
rst_generated_includes = [
    "dtype_promotion_table",
    "numeric_dtypes"
]
# Generate RST include files
for include_name in rst_generated_includes:
    function_name = f"_generate_rst_{include_name}"
    file_path = os.path.join(generated_includes_dir, f"{include_name}.rst")
    content = getattr(cuda.tile._datatype, function_name)()
    with open(file_path, 'w') as f:
        f.write(content)

# Include substitutions from references.rst in all documents (including docstrings)
with open(os.path.join(os.path.dirname(__file__), 'references.rst'), 'r') as f:
    rst_prolog = f.read()

# Don't expand type aliases. See https://github.com/sphinx-doc/sphinx/issues/10785
autodoc_type_aliases = {
    'Constant': 'Constant',
    'Shape': 'Shape',
}


# Make links to type aliases actually work.
def resolve_type_aliases(app, env, node, contnode):
    """Resolve :class: references to our type aliases as :data: instead."""
    if (
        node["refdomain"] == "py"
        and node["reftype"] == "class"
        and node["reftarget"] in autodoc_type_aliases.keys()
    ):
        print("Resolving type alias", node["reftarget"])
        return app.env.get_domain("py").resolve_xref(
            env, node["refdoc"], app.builder, "data", node["reftarget"], node, contnode
        )


def setup(app):
    app.connect("missing-reference", resolve_type_aliases)
