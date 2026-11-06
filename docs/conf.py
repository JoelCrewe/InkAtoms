"""Sphinx configuration for the inkatoms documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
import inkatoms  # noqa: E402

project = "inkatoms"
author = "Joel Crewe"
copyright = "2026, Joel Crewe"
release = inkatoms.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]
autosummary_generate = True
autodoc_member_order = "bysource"
html_theme = "furo"
html_title = f"inkatoms {release}"
exclude_patterns = ["_build", "images"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples/gallery",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
}


def _plain_docstrings(app, what, name, obj, options, lines):
    """The docstrings are written as aligned plain text; show them preformatted."""
    if lines:
        body = ["    " + line for line in lines]
        lines[:] = ["::", ""] + body + [""]


def setup(app):
    app.connect("autodoc-process-docstring", _plain_docstrings)
