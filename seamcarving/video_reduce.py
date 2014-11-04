# coding=UTF-8
from numpy import max, concatenate, size, ones, r_, zeros, amax, where, append, empty
import numpy as np
from random import random
import numexpr as ne
import maxflow
from seamcarving.utils import cli_progress_bar, cli_progress_bar_end

DEBUG = True


class video_seam_carving_decomposition(object):
  #
  # X: An fxnxmxc matrix (f = frame, n = rows, m = columns, c = components)
  # S: skeletal(cartoon) image of the input image
  # T: importance map
  # deleteNumberW  : Number of columns to be deleted
  # deleteNumberH  : Number of rows to be deleted
  #
  def __init__(self, X, S, T, deleteNumberW, deleteNumberH, alpha, beta, use_integers=True):
    self.X = X
    self.S = S
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
    # Pot = copy(imp)
    # for ii in xrange(1, size(Pot, 0)):  # =2:size(Pot, 1)
    #   pp = Pot[:, ii - 1, :]  # one row in front of ii
    #   energy3 = zeros((Pot.size[0], Pot.size[2], 3))
    #   # Energy in the case of a seam that binds to L
    #   energy3[:, 0] = concatenate(([0], pp[:, 0:-1]))  # The left side of the screen is not calculated
    #   # Energy in the case of a seam that binds to U
    #   energy3[:, 1] = pp
    #   # Energy in the case of a seam that binds to the R
    #   energy3[:, 2] = append(pp[:, 1:], 0, axis=1)  # The right edge of the screen is not calculated
    #   Pot[:, ii, :] = Pot[:, ii, :] + energy3.max(axis=2)

    impMax = 255 # Pot[:, -1, :].max()

    iteMax = size(imp, 1)

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

  def generate_graph(self, I, Imp, ite):
    g = maxflow.Graph[float]()
    i_inf = np.inf
    i_mult = 1

    if self.use_integers:
      g = maxflow.Graph[int]()
      i_inf = 10000000
      i_mult = 10000

    nodeids = g.add_grid_nodes(I.shape)

    # SX
    structure = np.zeros((3, 3, 3))
    structure[1, 1, 2] = 1

    pastleft = I[:, :, 1:] - I[:, :, 0:-1]
    futureleft = ((I[:, :, 1:-1] + I[:, :, 2:]) * 0.5) - I[:, :, 0:-2]

    pastright = -pastleft # I[:, 0:-1] - I[:, 1:] = IO - sx
    futureright = ((I[:, :, 0:-2] + I[:, :, 1:-1]) * 0.5) - I[:, :, 2:]

    left = (pastleft[:, :, 0:-1] - futureleft) ** 2
    right = (pastright[:, :, 0:-1] - futureright) ** 2

    weights = np.zeros(I.shape)
    weights[:, :, 1:-2] = (left[:, :, 0:-1] + right[:, :, 1:]) / 510 * self.alpha
    weights = weights * i_mult

    weights[:, :, 0:-1] = weights[:, :, 0:-1] + Imp + ite

    weights[:, :, -2] = i_inf
    weights[:, :, 0] = i_inf

    # weights = np.zeros(I.shape)
    # weights[:, :, 1:-1] = np.abs(I[:, :, 0:-2] - ((I[:, :, 1:-1] + I[:, :, 2:]) * 0.5))
    # weights = weights * i_mult
    # weights[:, :, 0] = i_inf

    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # # GIU
    # structure = np.zeros((3, 3, 3))
    # structure[1, 2, 1] = 1

    # weights = np.zeros(I.shape)
    # weights[:, 0:-1, 0:-1] = np.abs(I[:, 1:, 0:-1] - ((I[:, 0:-1, 0:-1] + I[:, 0:-1, 1:]) * 0.5))
    # weights = weights * i_mult
    # g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # # SU
    # structure = np.zeros((3, 3, 3))
    # structure[1, 0, 1] = 1

    # weights = np.zeros(I.shape)
    # weights[:, 0:-1, 0:-1] = np.abs(I[:, 1:, 0:-1] - ((I[:, 0:-1, 0:-1] + I[:, 0:-1, 1:]) * 0.5))
    # weights = weights * i_mult
    # g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # Diagonali su singola immagine
    structure = np.zeros((3, 3, 3))
    structure[1, :, 0] = i_inf
    structure[0, 0, 1] = i_inf
    structure[0, 2, 1] = i_inf

    g.add_grid_edges(nodeids, structure=structure)

    # In profondità sx
    structure = np.zeros((3, 3, 3))
    structure[2, 1, 1] = 1

    weights[0:-1, :, 1:-1] = np.abs(((I[0:-1, :, 1:-1] + I[0:-1, :, 2:]) * 0.5) - ((I[1:, :, 0:-2] + I[1:, :, 1:-1]) * 0.5))
    weights = weights * i_mult
    # g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # In profondità dx
    structure = np.zeros((3, 3, 3))
    structure[0, 1, 1] = 1

    weights = np.zeros(I.shape)
    weights[1:, :, 1:-1] = np.abs(((I[0:-1, :, 0:-2] + I[0:-1, :, 1:-1]) * 0.5) - ((I[1:, :, 1:-1] + I[1:, :, 2:]) * 0.5))
    weights = weights * i_mult
    # g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    g.add_grid_tedges(nodeids[:, :, 0], i_inf, 0)
    g.add_grid_tedges(nodeids[:, :, -1], 0, i_inf)
    return g, nodeids

  def graph_cut(self, I, Imp, ite):
    g, nodeids = self.generate_graph(I, Imp, ite)
    g.maxflow()
    pathMap = g.get_grid_segments(nodeids)
    I = (pathMap == False).sum(2) - 1
    del g
    return I, pathMap

  def graph_cut_slow(self, I, Imp, ite):

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(I.shape)

    # SX
    structure = np.zeros((3, 3, 3))
    structure[1, 1, 2] = 1

    weights = np.zeros(I.shape)
    weights[:, :, 1:-1] = np.abs(I[:, :, 0:-2] - ((I[:, :, 1:-1] + I[:, :, 2:]) * 0.5))
    weights[:, :, 0] = np.inf

    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # GIU
    structure = np.zeros((3, 3, 3))
    structure[1, 2, 1] = 1

    weights = np.zeros(I.shape)
    weights[:, 0:-1, 0:-1] = np.abs(I[:, 1:, 0:-1] - ((I[:, 0:-1, 0:-1] + I[:, 0:-1, 1:]) * 0.5))
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # SU
    structure = np.zeros((3, 3, 3))
    structure[1, 0, 1] = 1

    weights = np.zeros(I.shape)
    weights[:, 0:-1, 0:-1] = np.abs(I[:, 1:, 0:-1] - ((I[:, 0:-1, 0:-1] + I[:, 0:-1, 1:]) * 0.5))
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # Diagonali su singola immagine
    structure = np.zeros((3, 3, 3))
    structure[1, :, 0] = np.inf
    structure[0, 0, 1] = np.inf
    structure[0, 2, 1] = np.inf

    g.add_grid_edges(nodeids, structure=structure)

    # In profondità sx
    structure = np.zeros((3, 3, 3))
    structure[2, 1, 1] = 1

    weights = np.zeros(I.shape)
    weights[0:-1, :, 1:-1] = np.abs(((I[0:-1, :, 1:-1] + I[0:-1, :, 2:]) * 0.5) - ((I[1:, :, 0:-2] + I[1:, :, 1:-1]) * 0.5))
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

    # In profondità dx
    structure = np.zeros((3, 3, 3))
    structure[0, 1, 1] = 1

    weights = np.zeros(I.shape)
    weights[1:, :, 1:-1] = np.abs(((I[0:-1, :, 0:-2] + I[0:-1, :, 1:-1]) * 0.5) - ((I[1:, :, 1:-1] + I[1:, :, 2:]) * 0.5))
    g.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)


    # X, Y = np.mgrid[:I.shape[0], :I.shape[1]]
    # X, Y = X.reshape(1, np.prod(X.shape)), Y.reshape(1, np.prod(Y.shape))
    # left_most = concatenate((X, Y, np.zeros_like(X))).astype(np.uint64)
    # left_most = np.ravel_multi_index(left_most, I.shape)

    g.add_grid_tedges(nodeids[:, :, 0], np.inf, 0)

    # right_most = left_most + I.shape[2] - 1
    g.add_grid_tedges(nodeids[:, :, -1], 0, np.inf)

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
  def apply_seam_carving(self, I, mask, q11, upQ11, q12, upQ12, p12, upP12, p22, upP22, Simg, v, Z):
    reduced_size_1, reduced_size_2, reduced_size_3 = size(Simg, 0), size(Simg, 1), size(Simg, 2) - 1

    ## Deletion:
    # Generating a deletion mask n x m. It's a binary matrix that contains True if the pixel should be keeped, False if they should be deleted.
    # The total number of Falses and Trues at each like should be the same.
    # Applying that matrix to a standard numpy array, it efficiently generates a clone matrix with the deleted values
    # After applying the mask, the new vector generated is flattened, so you should reshape it.
    q11Copy = q11[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3)
    q12Copy = q12[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3)

    SimgCopy = Simg[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3)

    p12Copy = p12[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3, p12.shape[3])
    p22Copy = p22[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3, p22.shape[3])
    ZCopy = Z[mask].reshape(reduced_size_1, reduced_size_2, reduced_size_3, Z.shape[3])

    ## Merge:
    # I is converted to an integer matrix, in order to be used as an index map.
    # This can achieve a non-aligned multirow substitution very efficiently
    # Every indexed value of the seam is replaced with it's look-forward version.
    I = I.astype(np.uint32)
    # r = r_[0:size(I)]
    X, Y = np.mgrid[:q11Copy.shape[0], :q11Copy.shape[1]]
    q11Copy[X, Y, I] = upQ11[X, Y, I]
    q12Copy[X, Y, I] = upQ12[X, Y, I]

    p12Copy[X, Y, I, :] = upP12[X, Y, I]
    p22Copy[X, Y, I, :] = upP22[X, Y, I]

    SimgCopy[X, Y, I] = v[X, Y, I]
    # Z lookforward version is not precomputed, so you have to do it in real time
    ZCopy[X, Y, I, :] = Z[X, Y, I, :] + Z[X, Y, I + 1, :]

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

  def generateNorthEnergy(self, Simg, v, northA, northB, northC):
    square = self.square
    DD = self.initD(Simg)
    DD[1:, :] = v[1:, :] - v[0:-1, :]  # Dovrebbe essere c_0(q, n)
    CNcc = square(DD, northA, northB, northC)  # Dovrebbe essere ||c_k(q, n) - c_0(q, n)||^2

    # Upper-left connection
    DD = self.initD(Simg)
    DD[1:, 1:] = v[1:, 1:] - Simg[0:-1, 2:]
    CNcnCL = square(DD, northA, northB, northC)

    # Upper-right connection
    DD = self.initD(Simg)
    DD[1:, 0:-1] = v[1:, 0:-1] - Simg[0:-1, 0:-2]
    CNcnCR = square(DD, northA, northB, northC)
    return CNcc, CNcnCL, CNcnCR

  def generateSouthEnergy(self, Simg, v, southA, southB, southC):
    square = self.square
    # Lower connection
    # CScc = || Structure_k+1 - Structure_k ||_2 in south direction
    # Structure_k+1 = Skel(r) - Skel(r-1) # (r = pixel)
    DD = self.initD(Simg)
    DD[0:-1, :] = v[0:-1, :] - v[1:, :]
    CScc = square(DD, southA, southB, southC)

    # Lower-left connection
    DD = self.initD(Simg)
    DD[0:-1, 0:-1] = v[0:-1, 0:-1] - Simg[1:, 0:-2]
    CScnCL = square(DD, southA, southB, southC)

    # Lower-right connection
    DD = self.initD(Simg)
    DD[0:-1, 1:] = v[0:-1, 1:] - Simg[1:, 2:]
    CScnCR = square(DD, southA, southB, southC)

    return CScc, CScnCL, CScnCR

  def generateEastEnergy(self, Simg, v, eastA, eastB, eastC):
    DD = self.initD(Simg)
    DD[:, 0:-1] = v[:, 0:-1] - Simg[:, 2:]
    return self.square(DD, eastA, eastB, eastC)

  def generateWestEnergy(self, Simg, v, westA, westB, westC):
    DD = self.initD(Simg)
    DD[:, 1:] = v[:, 1:] - Simg[:, 0:-2]
    return self.square(DD, westA, westB, westC)

  def generateEnergyUpLeftRight(self, CScc, CNcc, CScnCL, CNcnCL, CScnCR, CNcnCR):
    CU = zeros(CScc.shape)
    # Qui non è niente di che: CS e CN sono disallineate, perché sono una rispetto al nord e una rispetto al sud
    # e devo riallinearle per poterle sommare correttamente.
    CU[1:, :] = CScc[0:-1, :] + CNcc[1:, :]

    CL = zeros(CScc.shape)
    CL[1:, 1:] = CScnCL[0:-1, 0:-1] + CNcnCL[1:, 1:]

    CR = zeros(CScc.shape)
    CR[1:, 0:-1] = CScnCR[0:-1, 1:] + CNcnCR[1:, 0:-1]
    return CU, CL, CR

  def divide(self, a, b):
    return ne.evaluate('- a / b')

  def square(self, DD, a, b, c):
    return ne.evaluate('a * (DD ** 2) + 2 * b * DD + c')

  def calculatePot(self, CW, CE, alphaN, imp, gammaN, ite, betaN):
    return ne.evaluate('(CW + CE) * alphaN + imp * gammaN + betaN * ite')

  def sumShifted(self, a):
    return a[:, :, 0:-1] + a[:, :, 1:]

  def makeEdge(self, A):
    print A.shape
    X = np.ones_like(A)
    X[:, :, 0:-1] = A[:, :, 1:]
    return np.invert(A ^ X)

  def generate(self):
    sumShifted = self.sumShifted
    X, S, T = self.X, self.S, self.T

    # Precomputed sizes
    s_X_1, s_X_2, s_X_3, s_X_4 = X.shape
    s_T_1, s_T_2, s_T_3, s_T_4 = size(T, 0), size(T, 1), size(T, 2), 1
    s_S_1, s_S_2, s_S_3 = S.shape

    # Z is a matrix that contains both the image S, the matrix T (Importance Map),
    # and a matrix of ones. Each component looks like: [X_0, X_1, X_2, T_0, 1]
    Z = concatenate((X, T.reshape(s_T_1, s_T_2, s_T_3, s_T_4), ones((s_X_1, s_X_2, s_X_3, 1))), axis=3)
    s_Z_1, s_Z_2, s_Z_3, s_Z_4 = Z.shape


    # A fast way to index all components of Z that contains the components of X
    ZIindex = r_[0:s_X_4]
    # A fast way to index all components of Z that contains the components of T
    ZTindex = s_X_4 + r_[0:s_T_4]
    # A fast way to index all components of Z that contains the ones
    ZUindex = s_Z_4 - 1
    # Both T and ones indexes together
    ZTUindex = append(ZTindex, ZUindex)

    # q11 is a matrix of ones. It's used as to calculate the correct mean value of
    # pixel values of the merged image. Ad each position contains a value so that
    # p12 / that_value = mean value of the image in that position.
    q11 = ones(S.shape, order='C')
    # Precomputed value of -S (skeleton image). It's updated every frame and
    # represents the actual value of the skeleton seam-merged skeleton image
    q12 = np.ascontiguousarray(-S)

    # list of indexes for easy access to p12 matrix's components
    up, down, right, left = 0, 1, 2, 3

    # p12 is a matrix with four components, one for each direction. It represents
    # the structure value of each pixel of the image as a difference between
    # the gamma value of that pixel and it's neighbour.
    # On the paper it's called c(r)
    p12 = zeros((s_S_1, s_S_2, s_S_3, 4))
    # Upper connection
    # C(r, 0)
    p12[:, 1:, :, up] = S[:, 1:] - S[:, 0:-1]  # [[zeros(1, size(S, 2))], [S[1:, :] - S[-2, :]]]
    # Lower connection
    # C(r, 1)
    p12[:, 0:-1, :, down] = S[:, 0:-1] - S[:, 1:]
    # Right connection
    # C(r, 2)
    p12[:, :, 0:-1, right] = S[:, :, 0:-1] - S[:, :, 1:]
    # Left connection
    # C(r, 3)
    p12[:, :, 1:, left] = S[:, :, 1:] - S[:, :, 0:-1]

    # Precomputing inverse value for mean-square calculation
    p12 = -p12
    # Precomputing square value of p12 for mean-square calculation
    p22 = p12 ** 2

    # Cloning S
    Simg = np.copy(S)

    alphaN, gammaN, betaN = 0, 0, 0

    seams = np.empty((self.deleteNumberW + self.deleteNumberH, X.shape[0], X.shape[1]))
    # For each seam I want to merge
    num_seams = self.deleteNumberW + self.deleteNumberH
    for i in xrange(num_seams):
      cli_progress_bar(i, num_seams)

      # Improved sum shifted = summing each column of the pixel with the
      # one in the right. It's the look-forward value for each matrix.
      # matrix
      # upQ11, upQ12, upP12, upP22 = improvedSumShifted(q11, q12, p12, p22)
      upQ11 = sumShifted(q11)
      upQ12 = sumShifted(q12)
      upP12 = sumShifted(p12)
      upP22 = sumShifted(p22)

      # v is the mean look-forward value of S, for each pixel
      v = self.divide(upQ12, upQ11)

      # Calculating future-value for both importance map and ones, that is the sum
      # of a pixel with its right-most neighbour.
      Z_T = Z[:, :, :, ZTUindex]
      temp = sumShifted(Z_T)
      imp = temp[:, :, :, 0]
      # imp = self.sumShifted(Z[:, :, ZTindex], Z[:, :, ZTindex])

      ite = temp[:, :, :, 1]
      # ite = self.sumShifted(Z[:, :, ZUindex], Z[:, :, ZUindex])  # Z[:, 0:-1, ZUindex] + Z[:, 1:, ZUindex]
      # This step is quite useless if importance map and ones map is a single component matrix
      if size(ZTUindex) > 2:
        imp = imp.sum(axis=3)
        ite = ite.sum(axis=3)

      # Calculating the maximum possible value for alpha, beta, gamma, dividing their
      # values by the maximum value that can be obtained by dynamic programming results.
      if i == 0:
        alphaN, gammaN, betaN = self.initializeParameters(imp)

      I, pathMap = self.graph_cut(Simg, imp * gammaN, ite * betaN)

      if DEBUG:
        seams[i] = I

      mask = self.makeEdge(pathMap)

      q11, q12, p12, p22, Simg, Z = self.apply_seam_carving(I, mask, q11, upQ11, q12, upQ12, p12, upP12, p22, upP22, Simg, v, Z)

    cli_progress_bar_end()
    img = Z[:, :, :, ZIindex]
    img = img / Z[:, :, :, [ZUindex, ZUindex, ZUindex]]  # ???
    return (img, seams) if DEBUG else img
