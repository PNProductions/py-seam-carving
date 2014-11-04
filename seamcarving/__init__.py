from seamcarving.image_reduce import seam_carving_decomposition
from seamcarving.image_enlarge import seam_carving_decomposition_enlargement
from seamcarving.video_reduce import video_seam_carving_decomposition
import seamcarving.utils as _utils


def progress_bar(value):
  _utils.PROGRESS_BAR = value


def seam_carving(image, enlarge_by, use_integers=True):
  instance = None
  if image.ndim == 3:
    if enlarge_by > 0:
      instance = seam_carving_decomposition_enlargement(image, enlarge_by, 0)
    else:
      instance = seam_carving_decomposition(image, -enlarge_by, 0, use_integers)
  else:
    instance = video_seam_carving_decomposition(image, enlarge_by, 0, use_integers)

  return instance.generate()

import version
import os
dirname = os.path.dirname(__file__)
__version__ = version.version
