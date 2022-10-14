#!/usr/bin/env python
import os
import pkgutil
import subprocess
import sys
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.test import test as TestCommand
from glob import glob
from os.path import basename

scripts = [s for s in glob('scripts/*') if basename(s) != '__pycache__']

setup(
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["examples"]),
    scripts=scripts,
    use_scm_version=True,
    include_package_data=True,)
