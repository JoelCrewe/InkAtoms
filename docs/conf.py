"""Sphinx configuration."""
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
html_static_path = ["_static"]
html_favicon = "_static/logo/favicon.ico"
html_theme_options = {
    # wordmark made by _static/logo/make_logo.py
    "light_logo": "logo/inkatoms_logo.svg",
    "dark_logo": "logo/inkatoms_logo_dark.svg",
    "sidebar_hide_name": True,
}
exclude_patterns = ["_build", "images"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples/gallery",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
}

