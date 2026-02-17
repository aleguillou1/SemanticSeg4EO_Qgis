# Configuration file for the Sphinx documentation builder.
# For SemanticSeg4EO QGIS Plugin

import os
import sys

# -- Project information -------------------------------------------------------
project = 'SemanticSeg4EO'
copyright = '2024, SemanticSeg4EO Team'
author = 'SemanticSeg4EO Team'
release = '1.1.0'
version = '1.1'

# -- General configuration -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
language = 'en'

# -- Options for HTML output ---------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    'display_github': True,
    'github_user': 'aleguillou1',
    'github_repo': 'SemanticSeg4EO',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# -- Options for todo extension ------------------------------------------------
todo_include_todos = True

# -- MyST configuration --------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
