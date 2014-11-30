# coding=UTF-8
import numpy as np
from seamcarving.image_reduce import seam_carving_decomposition


class seam_carving_decomposition_enlargement(seam_carving_decomposition):
  def insert_indices(self, A, B, C, index_map):
    return np.insert(A.ravel(), index_map, B[xrange(B.shape[0]), C]).reshape(A.shape[0], A.shape[1] + 1)

  # Given A, B, C, transforms indices from 3d to flattern forms for A and B, given C
  def find_indices3(self, A, B, C):
    mi = np.ravel_multi_index([np.arange(A.shape[0]), C], A.shape[:2])
    mi2 = np.ravel_multi_index([np.arange(B.shape[0]), C], B.shape[:2])
    return mi, mi2

  # Given A, B, C, and multi-indexes, inserts elements from B in A according to the indexes
  def insert_indices3(self, A, B, mi, mi2):
    bvals = np.take(B.reshape(-1, B.shape[-1]), mi2, axis=0)
    return np.insert(A.reshape(-1, A.shape[2]), mi + 1, bvals, axis=0).reshape(A.shape[0], -1, A.shape[2])

  def apply_seam_carving(self, I, Simg, Z):
    I = I.astype(np.uint64)
    index_map = np.ravel_multi_index((xrange(Simg.shape[0]), I), Simg.shape) + 1
    SimgCopy = self.insert_indices(Simg, Simg, I, index_map)
    mi, m2 = self.find_indices3(Z, Z, I)
    ZCopy = self.insert_indices3(Z, Z, mi, m2)

    return SimgCopy, ZCopy
