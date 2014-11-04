#!/usr/bin/env python

import sys
import runpy

__module_name__ = 'py-seam-carving'
__version_str__ = runpy.run_path('seamcarving/version.py')['version']

if sys.version_info < (2, 7):
    raise RuntimeError('must use python 2.7 or greater')

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name=__module_name__,
      version=__version_str__,
      description='Improved seam carving for videos and images using graphcut.',
      author='Piero Dotti, Paolo Guglielmini',
      author_email='pnproductions.dev@gmail.com',
      url='http://github.com/PnProductions/py-seam-carving',
      license='MIT',
      packages=['seamcarving'],
      install_requires=requirements,
      setup_requires=requirements
      )
