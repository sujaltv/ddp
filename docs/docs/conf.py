import os
import sys


sys.path.insert(0, os.path.abspath('../..'))

project = 'DDPW'
author = 'Sujal'
templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

extensions = [
  "sphinx.ext.intersphinx",
  "sphinx.ext.autodoc",
  "sphinx.ext.doctest",
  "sphinx.ext.extlinks",
  "sphinx.ext.autosummary",
  "fluiddoc.mathmacro",
  "sphinx.ext.todo",
  "sphinx.ext.mathjax",
  # "sphinx.ext.viewcode",
  "sphinx_copybutton",
  "sphinxcontrib.bibtex",
]

html_theme = 'furo'
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False
html_title = 'DDPW'

html_context = {
  "display_github": True,  # 'Edit on Github' instead of 'View page source'
  "github_user": "sujaltv",
  "github_repo": "ddpw",
  "github_version": "master"
}

pygments_style = "friendly"
pygments_dark_style = "material"

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'super'
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

add_module_names = True # autocode to show only the final name
autodoc_preserve_defaults = True # True does not evaluate default values
autodoc_typehints_format = 'short' # short typehints for class/method arguments
autodoc_member_order = "bysource"
