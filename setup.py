from pathlib import Path

from setuptools import setup

scripts = [str(s) for s in Path('scripts/').iterdir() if s.is_file() and s.name != '__pycache__']

setup(scripts=scripts)
