# coding=UTF-8
from numpy import size, amax, where, empty
import numpy as np
from random import random
import maxflow
from seamcarving.utils import cli_progress_bar, cli_progress_bar_end

DEBUG = True


class video_seam_carving_decomposition(object):
  #
  # X: An fxnxmxc matrix (f = frame, n = rows, m = columns, c = components)
  # deleteNumberW  : Number of columns to be deleted
  # deleteNumberH  : Number of rows to be deleted
  #
  def __init__(self, X, deleteNumberW, deleteNumberH, use_integers=True):
    self.X = X
    self.deleteNumberW = deleteNumberW
    self.deleteNumberH = deleteNumberH
    self.use_integers = use_integers

  def generate_up_down_edges(self, I, nodeids, i_inf, i_mult):
    structure = np.zeros((3, 3, 3))
    structure[1, 2, 1] = 1
    links = np.zeros(I.shape)
    links[:, 0:-1, 1:] = np.abs(I[:, 0:-1, 1:] - I[:, 1:, 0:-1])
    links = links * i_mult
    return links, structure

  def generate_down_up_edges(self, I, nodeids, i_inf, i_mult):
    structure = np.zeros((3, 3, 3))
    structure[1, 0, 1] = 1
    links = np.zeros(I.shape)
    links[:, 1:, 1:] = np.abs(I[:, 1:, 1:] - I[:, 0:-1, 0:-1])
    links = links * i_mult
    return links, structure

  def generate_left_right_edges(self, I, nodeids, i_inf, i_mult):
    structure = np.zeros((3, 3, 3))
    structure[1, 1, 2] = 1
    links = np.zeros(I.shape)
    links[:, :, 1:-1] = np.abs(I[:, :, 2:] - I[:, :, 0:-2])
    links = links * i_mult
    links[:, :, -2] = i_inf
    links[:, :, 0] = i_inf
    return links, structure

  def generate_backward_forward_edges(self, I, nodeids, i_inf, i_mult):
    structure = np.zeros((3, 3, 3))
    structure[2, 1, 1] = 1
    links = np.zeros(I.shape)
    links[0:-1, :, 1:] = np.abs(I[0:-1, :, 1:] - I[1:, :, 0:-1])
    links = links * i_mult
    return links, structure

  def generate_forward_backward_edges(self, I, nodeids, i_inf, i_mult):
    structure = np.zeros((3, 3, 3))
    structure[0, 1, 1] = 1
    links = np.zeros(I.shape)
    links[1:, :, 0:-1] = np.abs(I[0:-1, :, 0:-1] - I[1:, :, 1:])
    links = links * i_mult
    return links, structure

  def generate_graph(self, I):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    i_mult = 1

    if self.use_integers:
      g = maxflow.Graph[int]()
      i_inf = 10000000
      i_mult = 10000

    nodeids = g.add_grid_nodes(I.shape)

    links, structure = self.generate_left_right_edges(I, nodeids, i_inf, i_mult)
    g.add_grid_edges(nodeids, structure=structure, weights=links, symmetric=False)
    links, structure = self.generate_up_down_edges(I, nodeids, i_inf, i_mult)
    g.add_grid_edges(nodeids, structure=structure, weights=links, symmetric=False)
    links, structure = self.generate_down_up_edges(I, nodeids, i_inf, i_mult)
    g.add_grid_edges(nodeids, structure=structure, weights=links, symmetric=False)

   # Diagonali su singola immagine
    structure = np.zeros((3, 3, 3))
    structure[1, :, 0] = i_inf
    structure[2, 1, 0] = i_inf
    structure[0, 1, 0] = i_inf
    g.add_grid_edges(nodeids, structure=structure)

    links, structure = self.generate_backward_forward_edges(I, nodeids, i_inf, i_mult)
    g.add_grid_edges(nodeids, structure=structure, weights=links, symmetric=False)
    links, structure = self.generate_forward_backward_edges(I, nodeids, i_inf, i_mult)
    g.add_grid_edges(nodeids, structure=structure, weights=links, symmetric=False)

    g.add_grid_tedges(nodeids[:, :, 0], i_inf, 0)
    g.add_grid_tedges(nodeids[:, :, -1], 0, i_inf)
    return g, nodeids

  def graph_cut(self, I):
    g, nodeids = self.generate_graph(I)
    g.maxflow()
    pathMap = g.get_grid_segments(nodeids)
    I = (pathMap == False).sum(2) - 1
    del g
    return I, pathMap

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
  def apply_seam_carving(self, I, mask, Simg, Z):
    reduced_size_1, reduced_size_2, reduced_size_3 = size(Simg, 0), size(Simg, 1), size(Simg, 2) - 1
    SimgCopy = Simg[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3)
    ZCopy = Z[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3, Z.shape[3])
    return SimgCopy, ZCopy

  # Starting from the energy map and the path map, it generates vector pix, a vector that maps, for each row, the column of the seam to be merged.
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

  def makeEdge(self, A):
    X = np.ones_like(A)
    X[:, :, 0:-1] = A[:, :, 1:]
    return np.invert(A ^ X)

  def generate(self):
    X = self.X
    # S = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY).astype(np.float64)
    S = X.astype(np.float64).sum(axis=3) / 3

    # Precomputed sizes
    s_X_1, s_X_2, s_X_3, s_X_4 = X.shape
    s_S_1, s_S_2, s_S_3 = S.shape

    Z = np.copy(X)

    # Cloning S
    Simg = np.copy(S)

    num_seams = abs(self.deleteNumberW + self.deleteNumberH)
    self.seams = np.empty((num_seams, X.shape[0], X.shape[1]))
    # For each seam I want to merge
    for i in xrange(num_seams):
      cli_progress_bar(i, num_seams)

      I, pathMap = self.graph_cut(Simg)

      self.seams[i] = I

      mask = self.makeEdge(pathMap)
      Simg, Z = self.apply_seam_carving(I, mask, Simg, Z)

    cli_progress_bar_end()
    return Z
