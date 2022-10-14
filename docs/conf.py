from pathlib import Path
import os
import sys
from configparser import ConfigParser
from datetime import datetime
import importlib

import sphinx
import stsci_rtd_theme


def setup(app):
    try:
        app.add_css_file("stsci.css")
    except AttributeError:
        app.add_stylesheet("stsci.css")


REPO_ROOT = Path(__file__).parent.parent

# Modules that automodapi will document need to be available
# in the path:
sys.path.insert(0, str(REPO_ROOT/"romanisim"))

# Read the package's setup.cfg so that we can use relevant
# values here:
conf = ConfigParser()
conf.read(REPO_ROOT/"setup.cfg")
setup_metadata = dict(conf.items("metadata"))

project = setup_metadata["name"]
author = setup_metadata["author"]
copyright = f"{datetime.now().year}, {author}"

package = importlib.import_module(setup_metadata['name'])
try:
    version = package.__version__.split('-', 1)[0]
    release = package.__version__
except AttributeError:
    version = 'dev'
    release = 'dev'

extensions = [
    "numpydoc",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autoclass_content = "both"

graphviz_output_format = "svg"
graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": True
}
html_theme_path = [stsci_rtd_theme.get_html_theme_path()]
html_domain_indices = True
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_use_index = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'galsim': ('https://galsim-developers.github.io/GalSim/_build/html/', None),
    'coord': ('https://lsstdesc.org/Coord/_build/html/', None),
}

nitpicky = True
