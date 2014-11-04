# coding=UTF-8
from numpy import concatenate, size, ones, zeros, amax, where, empty
import numpy as np
from random import random
import maxflow
import cv2
from seamcarving.utils import cli_progress_bar, cli_progress_bar_end


class seam_carving_decomposition(object):
  #
  # X: input image
  # deleteNumberW  : Number of columns to be deleted
  # deleteNumberH  : Number of rows to be deleted
  #
  def __init__(self, X, deleteNumberW, deleteNumberH, use_integers=True):
    self.X = X
    self.deleteNumberW = deleteNumberW
    self.deleteNumberH = deleteNumberH
    self.use_integers = use_integers

  def initD(self, Simg):
    return zeros((size(Simg, 0), size(Simg, 1) - 1))

  def find_neighborhood(self, image, node):
    index = np.unravel_index((node), image.shape)
    unraveled = ((index[0] + 1, index[1] - 1), (index[0] + 1, index[1]), (index[0] + 1, index[1] + 1))
    return unraveled

  def find_node(self, index, image):
    if index[0] < 0 or index[0] >= image.shape[0] or index[1] >= image.shape[1] or index[1] < 0:
      return None
    else:
      return np.ravel_multi_index(index, image.shape)

  def generate_graph(self, I):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    i_mult = 1

    if self.use_integers:
      g = maxflow.Graph[int]()
      i_inf = 10000000
      i_mult = 10000

    nodeids = g.add_grid_nodes(I.shape)
    links = zeros((I.shape[0], I.shape[1], 4))

    # SU
    # LR I(i,j+1)- I(i,j-1) (SU)
    links[:, 1:-1, 0] = np.abs(I[:, 2:] - I[:, 0:-2])

    links[:, -2, 0] = i_inf
    links[:, 0, 0] = i_inf

    # -LU I(i+1,j)- I(i,j-1) (DESTRA)
    links[0:-1, 1:, 1] = np.abs(I[1:, 1:] - I[0:-1, 0:-1])

    # LU (SINISTRA)
    # I(i-1,j)-I(i,j-1)
    links[1:, 1:, 2] = np.abs(I[0:-1, 1:] - I[1:, 0:-1])

    # GIU
    links[:, :, 3] = i_inf

    links = links * i_mult

    structure = np.array([[i_inf, 0, 0],
                          [i_inf, 0, 0],
                          [i_inf, 0, 0]
                          ])
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)

    # From Left to Right
    weights = links[:, :, 0]
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # GIU = destra
    weights = links[:, :, 1]
    structure = np.zeros((3, 3))
    structure[2, 1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # SU = sinistra
    weights = links[:, :, 2]
    structure = np.zeros((3, 3))
    structure[0, 1] = 1
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    left_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), zeros((1, I.shape[0])))).astype(np.uint64)
    left_most = np.ravel_multi_index(left_most, I.shape)
    g.add_grid_tedges(left_most, i_inf, 0)

    right_most = concatenate((np.arange(I.shape[0]).reshape(1, I.shape[0]), ones((1, I.shape[0])) * (size(I, 1) - 1))).astype(np.uint64)
    right_most = np.ravel_multi_index(right_most, I.shape)
    g.add_grid_tedges(right_most, 0, i_inf)
    return g, nodeids

  def graph_cut(self, I):
    g, nodeids = self.generate_graph(I)
    g.maxflow()
    I = g.get_grid_segments(nodeids)
    I = (I == False).sum(1) - 1
    I = I.reshape(I.shape[0], 1)
    return I

  ## Given the actual state matrixes of the algorithm, it applies the seam merging to each of them.
  # @I A vector that maps a certain row with a certain column, and represents which pixel of each row should be merged with the right neighbour
  # @q11 A matrix for mean pixel value calculation
  # @upQ11 Look-forward version of q11 (representing the value of every pixel merged with its right neighbour)
  # @q12 The actual inverse value of the skeletal image, without applying the mean value
  # @upQ12 Look-forward version of q12
  # @p12 A 4-components (that represents the four directions) structure of the image. See initialization for more details.
  # @upP12 Look-forward version of p12
  # @p22 The square value of p12 (p12**2), precomputed.
  # @upP22 Look-forward version of p22
  # @Simg The actual skeletal image value (with the mean applied). It's equivalent to -q12/q11
  # @v The look-forward version of Simg
  # @Z A matrix that contains the original image, the structure image and a matrix of ones.
  #
  # Returns:
  # All the updated matrixes ready for the next iteration
  #
  # This method applies the merge in two steps:
  # * Deletion: For each row, deletes a value according to I.
  # * Merge/substitution: For each row, it replaces the actual value of the seam with it's look-forwarded version, according to I
  # The only exception is Z, that is not precomputed and should be calculated in real time.
  def apply_seam_carving(self, I, Simg, Z):
    reduced_size_1, reduced_size_2 = size(Simg, 0), size(Simg, 1) - 1

    ## Deletion:
    # Generating a deletion mask n x m. It's a binary matrix that contains True if the pixel should be keeped, False if they should be deleted.
    # The total number of Falses and Trues at each like should be the same.
    # Applying that matrix to a standard numpy array, it efficiently generates a clone matrix with the deleted values
    mask = np.arange(size(Z, 1)) != np.vstack(I)
    SimgCopy = Simg[mask].reshape(reduced_size_1, reduced_size_2)
    ZCopy = Z[mask].reshape(reduced_size_1, reduced_size_2, Z.shape[2])
    return SimgCopy, ZCopy

  ## Starting from the energy map and the path map, it generates vector pix, a vector that maps, for each row, the column of the seam to be merged.
  # @Pot The energy map. The position of minimum value of the last row of Pot represents the starting pixel of the seam (with a bottom-up strategy)
  # @pathMap A matrix that maps, for each position, the best direction to be taken to find the lower energy seam.
  #
  # Returns:
  # @pix the seam coordinates map.
  #
  # Example:
  # pix = [3, 4, 5, 5, 4, 5]
  # That maps this list of coordinates:
  # (0, 3), (1, 4), (2, 5), (3, 5), (4, 5), (5, 5)
  def generateSeamPath(self, Pot, pathMap):
    s_Pot_1 = Pot.shape[0]

    pix = empty((s_Pot_1, 1))
    Pot_last_line = Pot[-1, :]

    # mn, pix[-1] = Pot_last_line.min(axis=0), Pot_last_line.argmin(axis=0)
    # Finding the minimum value from Pot's last line's values.
    mn = Pot_last_line.min(axis=0)

    # Searching the list of indexes that have the minimum energy
    pp = where(Pot_last_line == mn)[0]

    # If there's more than one, it's random choosen
    pix[-1] = pp[int(random() * amax(pp.shape))]
    # Starting from the bottom
    for ii in reversed(xrange(0, s_Pot_1 - 1)):  # xrange(s_Pot_1 - 2, -1, -1):
      # Directions expressed in pathMap uses this rule: 0 => upper-left, 1 => upper, 2 => upper-right
      # They are remapped to be like that: -1 => upper-left, 0 => upper, 1 => upper-right
      # To calculate the coordinate at step ii, you should map with: coordinate(ii + 1) + remapped direction
      pix[ii] = pix[ii + 1] + pathMap[ii + 1, int(pix[ii + 1])] - 1
    return pix

  def generate(self):
    X = self.X
    S = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY).astype(np.float64)

    Z = np.copy(X)

    # Cloning S [To be fixed]
    Simg = np.copy(S)

    # For each seam I want to merge
    num_seams = self.deleteNumberW + self.deleteNumberH
    for i in xrange(num_seams):
      cli_progress_bar(i, num_seams)
      # pathmap is a matrix that, for each position, specifies the best direction
      # to be taken to minimize the cost.
      # Pot, pathMap = self.dynamic_programming(Pot, CU, CL, CR, zeros(Pot.shape))
      pix = self.graph_cut(Simg)

      # pix = self.generateSeamPath(Pot, pathMap)

      Simg, Z = self.apply_seam_carving(pix.transpose()[0], Simg, Z)

    cli_progress_bar_end()
    return Z
