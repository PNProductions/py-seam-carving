# coding=UTF-8
import numpy as np
from seammerging.reduction import SeamMergingWithDecomposition


class SeamMergingWithDecompositionEnlargement(SeamMergingWithDecomposition):
  def insert_indices(self, A, B, C, index_map):
    return np.insert(A.ravel(), index_map, B[xrange(B.shape[0]), C]).reshape(A.shape[0], A.shape[1] + 1)

  # Given A, B, C, transforms indices from 3d to flattern forms for A and B, given C
  def find_indices3(self, A, B, C):
    mi = np.ravel_multi_index([np.arange(A.shape[0]), C], A.shape[:2])
    mi2 = np.ravel_multi_index([np.arange(B.shape[0]), C], B.shape[:2])
    return mi, mi2

  # Given A, B, C, and multi-indexes, inserts elements from B in A according to the indexes
  def insert_indices3(self, A, B, C, mi, mi2):
    bvals = np.take(B.reshape(-1, B.shape[-1]), mi2, axis=0)
    return np.insert(A.reshape(-1, A.shape[2]), mi + 1, bvals, axis=0).reshape(A.shape[0], -1, A.shape[2])

  def insert_indices_sum(self, A, B, C, mi, mi2):
    bvals = np.take(B.reshape(-1, B.shape[-1]), mi2, axis=0)
    bvals2 = np.take(B.reshape(-1, B.shape[-1]), mi2 + 1, axis=0)
    return np.insert(A.reshape(-1, A.shape[2]), mi + 1, bvals + bvals2, axis=0).reshape(A.shape[0], -1, A.shape[2])

  def apply_seam_merging(self, I, q11, upQ11, q12, upQ12, p12, upP12, p22, upP22, Simg, v, Z):
    I = I.astype(np.uint64)

    index_map = np.ravel_multi_index((xrange(q11.shape[0]), I), q11.shape) + 1

    q11Copy = self.insert_indices(q11, upQ11, I, index_map)
    q12Copy = self.insert_indices(q12, upQ12, I, index_map)

    SimgCopy = self.insert_indices(Simg, v, I, index_map)

    mi, m2 = self.find_indices3(p12, upP12, I)
    p12Copy = self.insert_indices3(p12, upP12, I, mi, m2)
    p22Copy = self.insert_indices3(p22, upP22, I, mi, m2)

    mi, m2 = self.find_indices3(Z, Z, I)
    ZCopy = self.insert_indices_sum(Z, Z, I, mi, m2)

    return q11Copy, q12Copy, p12Copy, p22Copy, SimgCopy, ZCopy

  def generateNorthEnergy(self, Simg, v, northA, northB, northC):
    square = self.square
    DD = self.initD(Simg)
    DD[1:, :] = v[1:, :] - v[0:-1, :]  # Dovrebbe essere c_0(q, n)
    CNcc = square(DD, northA, northB, northC)  # Dovrebbe essere ||c_k(q, n) - c_0(q, n)||^2

    # Upper-left connection
    DD = self.initD(Simg)
    DD[1:, 1:] = v[1:, 1:] - Simg[0:-1, 1:-1]
    CNcnCL = square(DD, northA, northB, northC)

    # Upper-right connection
    DD = self.initD(Simg)
    DD[1:, 0:-1] = v[1:, 0:-1] - Simg[0:-1, 1:-1]
    CNcnCR = square(DD, northA, northB, northC)
    return CNcc, CNcnCL, CNcnCR

  def generateSouthEnergy(self, Simg, v, southA, southB, southC):
    square = self.square
    # Lower connection
    DD = self.initD(Simg)
    DD[0:-1, :] = v[0:-1, :] - v[1:, :]
    CScc = square(DD, southA, southB, southC)

    # Lower-left connection
    DD = self.initD(Simg)
    DD[0:-1, 0:-1] = v[0:-1, 0:-1] - Simg[1:, 1:-1]
    CScnCL = square(DD, southA, southB, southC)

    # Lower-right connection
    DD = self.initD(Simg)
    DD[0:-1, 1:] = v[0:-1, 1:] - Simg[1:, 1:-1]
    CScnCR = square(DD, southA, southB, southC)

    return CScc, CScnCL, CScnCR

  def generateEastEnergy(self, Simg, v, eastA, eastB, eastC):
    DD = self.initD(Simg)
    DD[:, 0:-1] = v[:, 0:-1] - Simg[:, 1:-1]
    return self.square(DD, eastA, eastB, eastC)

  def generateWestEnergy(self, Simg, v, westA, westB, westC):
    DD = self.initD(Simg)
    DD[:, 1:] = v[:, 1:] - Simg[:, 1:-1]
    return self.square(DD, westA, westB, westC)
