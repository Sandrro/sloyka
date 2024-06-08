# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sloyka'
copyright = '2024, itmo_idu'
author = 'itmo_idu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    "sphinx_inline_tabs",
    "sphinx_design",
    "sphinx_issues",

    # For using CONTRIBUTING.md.
    "myst_parser",

    "notfound.extension",

    # These extensions require RTDs to work so they will not work locally.
    "hoverxref.extension",
    "sphinx_search.extension",
]

templates_path = ['_templates']
exclude_patterns = []


language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_show_sphinx = False

html_static_path = ['_static']
html_logo = '../logo/logo.png'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

html_css_files = ['_static/custom.css']

# html_js_files = [
#     'https://cdnjs.cloudflare.com/ajax/libs/medium-zoom/1.0.6/medium-zoom.min.js',
#     'https://p.trellocdn.com/embed.min.js',
#     'custom.js',
# ]
