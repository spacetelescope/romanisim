from pathlib import Path

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = True
Options.annotate = False

# importing these extension modules is tested in `.github/workflows/build.yml`;
# when adding new modules here, make sure to add them to the `test_command` entry there
extensions = [Extension('romanisim.ramp_fit_casertano',
                        ['romanisim/ramp_fit_casertano.pyx'],
                        include_dirs=[np.get_include()])]

scripts = [str(s) for s in Path('scripts/').iterdir()
           if s.is_file() and s.name != '__pycache__']

setup(scripts=scripts,
      ext_modules=cythonize(extensions),
      )
