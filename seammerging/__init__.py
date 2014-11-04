from seammerging.reduction import SeamMergingWithDecomposition
from seammerging.enlargement import SeamMergingWithDecompositionEnlargement
import seammerging.utils as _utils


def progress_bar(value):
	_utils.PROGRESS_BAR = value


def seam_merging(image, cartoon, structure, enlarge_by, alpha=0.5, beta=0.5):
  instance = None
  if enlarge_by > 0:
    instance = SeamMergingWithDecompositionEnlargement(image, cartoon, structure, enlarge_by, 0, alpha, beta)
  else:
    instance = SeamMergingWithDecomposition(image, cartoon, structure, -enlarge_by, 0, alpha, beta)
  return instance.generate()

import version
import os
dirname = os.path.dirname(__file__)
__version__ = version.version
