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
  "sphinx_inline_tabs"
]

html_theme = 'furo'
html_show_sourcelink = False
html_show_copyright = True
copyright = f"Sujal"
html_favicon = './assets/favicon.ico'
html_show_sphinx = False
html_title = 'DDPW'
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sujaltv/ddpw",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "sidebar_hide_name": True
}
html_static_path = ['_static']
html_css_files = ['fonts.css', 'custom.css']

html_context = {
    "display_github": True,  # 'Edit on Github' instead of 'View page source'
    "github_user": "sujaltv",
    "github_repo": "ddpw",
    "github_version": "master",
    "copyright_duration": "2021-2024",
    "copyright_url": "https://sujal.tv"
}

pygments_style = "friendly"
pygments_dark_style = "material"

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'super'
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

# sphinx-autodoc
add_module_names = True # autocode to show only the final name
autodoc_member_order = "bysource"
autodoc_class_signature = 'mixed' # show init arguments without __init__
autodoc_preserve_defaults = True # True does not evaluate default values
autodoc_typehints = 'signature'
autodoc_typehints_format = 'short' # short typehints for class/method arguments

# sphinx-autodoc-typehints
typehints_fully_qualified = False # return type name resolution
typehints_use_signature = True
typehints_use_signature_return = True
typehints_document_rtype = False # show or hide return type
typehints_use_rtype = True

