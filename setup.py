#!/usr/bin/env python

__module_name__ = 'py-seam-merging'

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import runpy
from distutils.extension import Extension
import sys

if sys.version_info < (2, 7):
    raise RuntimeError('must use python 2.7 or greater')

__version_str__ = runpy.run_path("seammerging/version.py")["version"]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Lazy evaluate extension definition, to allow correct requirements install
class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

def extensions():
    import numpy
    from Cython.Build import cythonize
    include_path = [numpy.get_include()]
    ext = Extension('seammerging.native', ['seammerging/src/*.pyx'], include_dirs=include_path)
    return cythonize([ext])

setup(name=__module_name__,
      version=__version_str__,
      description='Seam merging for images implementation.',
      author='Piero Dotti, Paolo Guglielmini',
      author_email='pnproductions.dev@gmail.com',
      license='MIT',
      url='http://github.com/PnProductions/py-seam-merging',
      packages=['seammerging'],
      ext_modules=lazy_cythonize(extensions),
      install_requires=requirements,
      setup_requires=requirements
      )
