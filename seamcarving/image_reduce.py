# coding=UTF-8
from numpy import max, concatenate, size, ones, r_, zeros, inf, amax, where, shape, copy, append, empty
import numpy as np
from random import random
import numexpr as ne
from seamcarving.native import improvedSumShifted
import maxflow
import cv2
from seamcarving.utils import cli_progress_bar, cli_progress_bar_end


class seam_carving_decomposition(object):
  #
  # X: input image
  # S: skeletal(cartoon) image of the input image
  # T: importance map
  # deleteNumberW  : Number of columns to be deleted
  # deleteNumberH  : Number of rows to be deleted
  #
  def __init__(self, X, T, deleteNumberW, deleteNumberH, alpha, beta, use_integers=True):
    self.X = X
    self.T = T / max(max(T))
    self.deleteNumberW = deleteNumberW
    self.deleteNumberH = deleteNumberH
    self.alpha = alpha
    self.beta = beta
    self.gamma = 1 - alpha
    self.use_integers = use_integers

  def initD(self, Simg):
    return zeros((size(Simg, 0), size(Simg, 1) - 1))

  # Normalizzazione dei pesi: calcolo il risultato della programmazione
  # dinamica al primo passo e ne prendo il massimo invece del minimo.
  def initializeParameters(self, imp):
    # Maximum path of importance
    Pot = copy(imp)
    for ii in xrange(1, size(Pot, 0)):  # =2:size(Pot, 1)
      pp = Pot[ii - 1, :]  # one row in front of ii
      energy3 = zeros((size(Pot, 1), 3))
      # Energy in the case of a seam that binds to L
      energy3[:, 0] = concatenate(([0], pp[0:-1]))  # The left side of the screen is not calculated
      # Energy in the case of a seam that binds to U
      energy3[:, 1] = pp
      # Energy in the case of a seam that binds to the R
      energy3[:, 2] = append(pp[1:], 0)  # The right edge of the screen is not calculated
      Pot[ii, :] = Pot[ii, :] + energy3.max(axis=1)

    impMax = Pot[-1, :].max()

    iteMax = size(imp, 0)

    return self.alpha, self.gamma / impMax, self.beta / iteMax

  def find_neighborhood(self, image, node):
    index = np.unravel_index((node), image.shape)
    unraveled = ((index[0] + 1, index[1] - 1), (index[0] + 1, index[1]), (index[0] + 1, index[1] + 1))
    return unraveled

  def find_node(self, index, image):
    if index[0] < 0 or index[0] >= image.shape[0] or index[1] >= image.shape[1] or index[1] < 0:
      return None
    else:
      return np.ravel_multi_index(index, image.shape)

  def graph_cut(self, I, Imp, ite):
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
  def apply_seam_carving(self, I, q11, upQ11, q12, upQ12, p12, upP12, p22, upP22, Simg, v, Z):
    reduced_size_1, reduced_size_2 = size(Simg, 0), size(Simg, 1) - 1

    ## Deletion:
    # Generating a deletion mask n x m. It's a binary matrix that contains True if the pixel should be keeped, False if they should be deleted.
    # The total number of Falses and Trues at each like should be the same.
    # Applying that matrix to a standard numpy array, it efficiently generates a clone matrix with the deleted values
    mask = np.arange(size(Z, 1)) != np.vstack(I)
    # After applying the mask, the new vector generated is flattened, so you should reshape it.
    q11Copy = q11[mask].reshape(reduced_size_1, reduced_size_2)
    q12Copy = q12[mask].reshape(reduced_size_1, reduced_size_2)

    SimgCopy = Simg[mask].reshape(reduced_size_1, reduced_size_2)

    p12Copy = p12[mask].reshape(reduced_size_1, reduced_size_2, p12.shape[2])
    p22Copy = p22[mask].reshape(reduced_size_1, reduced_size_2, p22.shape[2])
    ZCopy = Z[mask].reshape(reduced_size_1, reduced_size_2, Z.shape[2])

    return q11Copy, q12Copy, p12Copy, p22Copy, SimgCopy, ZCopy

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

  def divide(self, a, b):
    return ne.evaluate('- a / b')

  def sumShifted(self, a):
    return a[:, 0:-1] + a[:, 1:]

  def generate(self):
    sumShifted = self.sumShifted
    X, T = self.X, self.T
    S = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Precomputed sizes
    s_X_1, s_X_2, s_X_3 = size(X, 0), size(X, 1), size(X, 2)
    s_T_1, s_T_2, s_T_3 = size(T, 0), size(T, 1), 1# size(T, 2) TO BE FIXED!!!
    s_S_1, s_S_2 = size(S, 0), size(S, 1)

    # Z is a matrix that contains both the image S, the matrix T (Importance Map),
    # and a matrix of ones. Each component looks like: [X_0, X_1, X_2, T_0, 1]
    Z = concatenate((X, T.reshape(s_T_1, s_T_2, s_T_3), ones((s_X_1, s_X_2, 1))), axis=2)
    s_Z_1, s_Z_2, s_Z_3 = size(Z, 0), size(Z, 1), size(Z, 2)


    # A fast way to index all components of Z that contains the components of X
    ZIindex = r_[0:s_X_3]
    # A fast way to index all components of Z that contains the components of T
    ZTindex = s_X_3 + r_[0:s_T_3]
    # A fast way to index all components of Z that contains the ones
    ZUindex = s_Z_3 - 1
    # Both T and ones indexes together
    ZTUindex = append(ZTindex, ZUindex)

    # q11 is a matrix of ones. It's used as to calculate the correct mean value of
    # pixel values of the merged image. Ad each position contains a value so that
    # p12 / that_value = mean value of the image in that position.
    q11 = ones(shape(S), order='C')
    # Precomputed value of -S (skeleton image). It's updated every frame and
    # represents the actual value of the skeleton seam-merged skeleton image
    q12 = np.ascontiguousarray(-S)

    # list of indexes for easy access to p12 matrix's components
    up, down, right, left = 0, 1, 2, 3

    # p12 is a matrix with four components, one for each direction. It represents
    # the structure value of each pixel of the image as a difference between
    # the gamma value of that pixel and it's neighbour.
    # On the paper it's called c(r)
    p12 = zeros((s_S_1, s_S_2, 4))
    # Upper connection
    # C(r, 0)
    p12[:, :, up] = concatenate((zeros((1, s_S_2)), S[1:, :] - S[0:-1, :]))  # [[zeros(1, size(S, 2))], [S[1:, :] - S[-2, :]]]
    # Lower connection
    # C(r, 1)
    p12[:, :, down] = concatenate((S[0:-1, :] - S[1:, :], zeros((1, s_S_2))))  # [[S[-2, :] - S[1:, :]], [zeros(1, size(S, 2))]]
    # Right connection
    # C(r, 2)
    p12[:, :, right] = concatenate((S[:, 0:-1] - S[:, 1:], zeros((s_S_1, 1))), axis=1)
    # Left connection
    # C(r, 3)
    p12[:, :, left] = concatenate((zeros((s_S_1, 1)), S[:, 1:] - S[:, 0:-1]), axis=1)

    # Precomputing inverse value for mean-square calculation
    p12 = -p12
    # Precomputing square value of p12 for mean-square calculation
    p22 = p12 ** 2

    # Cloning S [To be fixed]
    Simg = np.copy(S)

    alphaN, gammaN, betaN = 0, 0, 0

    # upQ11, upQ12, upP12, upP22 = None, None, None, None

    # For each seam I want to merge
    num_seams = self.deleteNumberW + self.deleteNumberH
    for i in xrange(0, num_seams):
      cli_progress_bar(i, num_seams)

      # Improved sum shifted = summing each column of the pixel with the
      # one in the right. It's the look-forward value for each matrix.
      # matrix
      upQ11, upQ12, upP12, upP22 = improvedSumShifted(q11, q12, p12, p22)

      # v is the mean look-forward value of S, for each pixel
      v = self.divide(upQ12, upQ11)

      # Upper connection
      # Temporary matrixes that represents differences between a pixel
      # and its northen neighbour.
      # CNcc, CNcnCL, CNcnCR = self.generateNorthEnergy(Simg, v, upQ11, upP12[:, :, up], upP22[:, :, up])

      # Lower connection
      # The same with the southern neighbour.
      # CScc, CScnCL, CScnCR = self.generateSouthEnergy(Simg, v, upQ11, upP12[:, :, down], upP22[:, :, down])

      # Right connection
      # CE = self.generateEastEnergy(Simg, v, upQ11, upP12[:, :, right], upP22[:, :, right])

      # Left connection
      # CW = self.generateWestEnergy(Simg, v, upQ11, upP12[:, :, left], upP22[:, :, left])

      # Error when binding a row on was just above
      # CU, CL, CR = self.generateEnergyUpLeftRight(CScc, CNcc, CScnCL, CNcnCL, CScnCR, CNcnCR)

      # Calculating future-value for both importance map and ones, that is the sum
      # of a pixel with its right-most neighbour.
      Z_T = Z[:, :, ZTUindex]
      temp = sumShifted(Z_T)
      imp = temp[:, :, 0]
      # imp = self.sumShifted(Z[:, :, ZTindex], Z[:, :, ZTindex])

      ite = temp[:, :, 1]
      # ite = self.sumShifted(Z[:, :, ZUindex], Z[:, :, ZUindex])  # Z[:, 0:-1, ZUindex] + Z[:, 1:, ZUindex]
      # This step is quite useless if importance map and ones map is a single component matrix
      if size(ZTUindex) > 2:
        imp = imp.sum(axis=2)
        ite = ite.sum(axis=2)

      # Calculating the maximum possible value for alpha, beta, gamma, dividing their
      # values by the maximum value that can be obtained by dynamic programming results.
      if i == 0:
        alphaN, gammaN, betaN = self.initializeParameters(imp)

      # Calcolo i valori iniziali di E(r).
      # Pot is initialized with E values for each pixel
      # Pot is M in the paper, and it's defined as:
      # Pot(i+1) = E(i) + min { ???? }
      # Pot = self.calculatePot(CW, CE, alphaN, imp, gammaN, ite, betaN)

      # Weighing CU CR and CL with input weight.
      # CU, CR, CL = CU * alphaN, CR * alphaN, CL * alphaN

      # pathmap is a matrix that, for each position, specifies the best direction
      # to be taken to minimize the cost.
      # Pot, pathMap = self.dynamic_programming(Pot, CU, CL, CR, zeros(Pot.shape))
      pix = self.graph_cut(Simg, imp * gammaN, ite * betaN)

      # pix = self.generateSeamPath(Pot, pathMap)

      q11, q12, p12, p22, Simg, Z = self.apply_seam_carving(pix.transpose()[0], q11, upQ11, q12, upQ12, p12, upP12, p22, upP22, Simg, v, Z)

    cli_progress_bar_end()
    img = Z[:, :, ZIindex]
    return img
