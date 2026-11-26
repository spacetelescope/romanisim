import sysconfig
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.docstrings = True
Options.annotate = False

FREE_THREADED_PYTHON = sysconfig.get_config_var("Py_GIL_DISABLED") == 1

define_macros = [
    ("NUMPY", "1"),
]
if not FREE_THREADED_PYTHON:
    define_macros.append(("Py_LIMITED_API", 0x030B0000))  # PY_VERSION_HEX for 3.11

# importing these extension modules is tested in `.github/workflows/build.yml`;
# when adding new modules here, make sure to add them to the `test_command` entry there
extensions = [
    Extension(
        "romanisim.ramp_fit_casertano",
        ["romanisim/ramp_fit_casertano.pyx"],
        include_dirs=[np.get_include()],
        define_macros=define_macros,
        py_limited_api=not FREE_THREADED_PYTHON,
    )
]

scripts = [str(s) for s in Path('scripts/').iterdir()
           if s.is_file() and s.name != '__pycache__']

SETUPTOOLS_OPTIONS = {}
if not FREE_THREADED_PYTHON:
    SETUPTOOLS_OPTIONS["bdist_wheel"] = {"py_limited_api": "cp311"}

setup(
    scripts=scripts,
    ext_modules=cythonize(extensions),
    options=SETUPTOOLS_OPTIONS,
)
