#!/usr/bin/env python

from setuptools import setup, find_packages
from glob import glob
from os.path import basename

scripts = [s for s in glob('scripts/*') if basename(s) != '__pycache__']

setup(
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["examples"]),
    scripts=scripts,
    use_scm_version=True,
    include_package_data=True,)
