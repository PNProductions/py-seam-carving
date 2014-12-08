# coding=UTF-8
import numpy as np
import videoseam as vs


class weights_delegate(object):
  """Delegate class to manage the weightning for the graph construction"""
  def __init__(self, parent, fill_with=np.inf, ndim=3):
    super(weights_delegate, self).__init__()
    self.parent = parent
    self.fill_with = fill_with
    self.ndim = ndim

  # Given a list of vectors and a list of links, creates an appropriate dictionary
  # @vectors a list of tuples, that represents a structure type
  # @links a list (or numpy array) of numpy arrays, that contains weights to be assigned to a specific structure
  #
  # returns: a dictionary
  #
  # Example:
  # @vectors [(2, 1, 1), (1, 0, 1)]
  # @links [[[1, 2], [1, 0]],  [[0, 1], [1, 1]]]
  # returns: {(2, 1, 1): [[1, 2], [1, 0]], (1, 0, 1): [[0, 1], [1, 1]]}
  def to_hash(self, vectors, links):
    return {k: v for k, v in zip(vectors, links)}

  # Given an n-dimensional tuple, resizes its dimensions to fit the class settings (self.ndim)
  # @tupleval a tuple
  # returns: another tuple with the correct dimension
  #
  # Example:
  # @listval (2, 1, 1)
  # returns (assuming self.ndim = 2): (1, 1)
  def adjust_dim(self, tupleval):
    if len(tupleval) == self.ndim:
      return tupleval
    resize = len(tupleval) - self.ndim
    return tupleval[resize:]

  # Given a list of n-dimensional tuple, resizes the dimension of each tuple to fit the class settings (self.ndim)
  # @listval a list a tuple
  # returns: a list of tuple with correct dimensions
  #
  # Example:
  # @listval [(2, 1, 1), (1, 0, 1)]
  # returns (assuming self.ndim = 2): [(1, 1), (0, 1)]
  def adjust_list(self, listval):
    return [self.adjust_dim(t) for t in listval]

  # Given I, it creates look-forward energies for that I, associated to the correct structure (1, 1, 2)
  # @I: An image skeleton (or a list of them)
  # returns: a dictionary that associates a structure key to an array of weights
  #
  # Example:
  # @I [[2, 1, 0, 3], [1, 0, 2, 4], [5, 2, 1, 3], [6, 2, 4, 3]]
  # returns {(1, 1, 2): [[inf, ?, ?, inf], [inf, ?, ?, inf], [inf, ?, ?, inf], [inf, ?, ?, inf]]}
  def weights_structure(self, I):
    vectors = self.adjust_list([(1, 1, 2)])  # left to right
    links = np.zeros((1,) + I.shape)

    # Formula: ((past_left - future_left)^2 + (past_right - future_right)^2) / 2
    pastleft = I[..., 1:] - I[..., 0:-1]
    futureleft = ((I[..., 1:-1] + I[..., 2:]) * 0.5) - I[..., 0:-2]

    pastright = -pastleft  # I[:, 0:-1] - I[:, 1:] = ME - sx
    futureright = ((I[..., 0:-2] + I[..., 1:-1]) * 0.5) - I[..., 2:]

    left = (pastleft[..., 0:-1] - futureleft) ** 2
    right = (pastright[..., 0:-1] - futureright) ** 2

    links[0, ..., 1:-2] = (left[..., 0:-1] + right[..., 1:]) * 0.5

    links = links * self.parent.alpha
    links[0, ..., -2] = self.fill_with
    links[0, ..., 0] = self.fill_with

    return self.to_hash(vectors, links)

  # Given I, it creates look-forward energies for that I, associated to the correct structure (2, 1, 1)
  # @I: An image skeleton (or a list of them)
  # returns: a dictionary that associates a structure key to an array of weights
  #
  # Example:
  # @I [[2, 1, 0, 3], [1, 0, 2, 4], [5, 2, 1, 3], [6, 2, 4, 3]]
  # returns {(1, 1, 2): [[inf, ?, ?, inf], [inf, ?, ?, inf], [inf, ?, ?, inf], [inf, ?, ?, inf]]}
  def weights_structure_time(self, I):
    vectors = [(2, 1, 1)]
    links = np.zeros((1,) + I.shape)

    pastleft = I[1:, :, :] - I[0:-1, :, :]
    futureleft = ((I[1:-1, :, :] + I[2:, :, :]) * 0.5) - I[0:-2, :, :]

    pastright = -pastleft  # I[:, 0:-1] - I[:, 1:] = ME - sx
    futureright = ((I[0:-2, :, :] + I[1:-1, :, :]) * 0.5) - I[2:, :, :]

    left = (pastleft[0:-1, :, :] - futureleft) ** 2
    right = (pastright[0:-1, :, :] - futureright) ** 2

    links[0, 1:-2, :, :] = (left[0:-1, :, :] + right[1:, :, :]) * 0.5
    links = links
    links[0, -2, :, :] = self.fill_with
    links[0, 0, :, :] = self.fill_with
    return self.to_hash(vectors, links)

  # A generic method to apply an energy function to a certain structure key
  # @I: The referring image
  # @A: The energy function
  # returns: a dictionary that associates a structure key to an array of weights
  #
  # Example:
  # @I [[2, 1, 3], [1, 0, 5], [5, 2, 3]]
  # @A [[2, 1], [1, 0], [5, 2]]
  # returns {(1, 1, 2): [[2, 1, 0], [1, 0, 0], [5, 2, 0]]}
  def weights_standard(self, I, A):
    vectors = self.adjust_list([(1, 1, 2)])  # left to right
    links = np.zeros((1,) + I.shape)
    links[0, ..., 0:-1] = A
    return self.to_hash(vectors, links)

  # Applies the importance map to a structure key. It applies also it's appropriate multiplier
  # @I: The referring image
  # @imp: importance map
  # returns: a dictionary that associates a structure key to an array of weights
  def weights_importance(self, I, imp):
    return self.weights_standard(I, imp * self.parent.gamma)

  # Applies the iterations count to a structure key. It applies also it's appropriate multiplier
  # @I: The referring image
  # @imp: The iteration counter energy function
  # returns: a dictionary that associates a structure key to an array of weights
  def weights_iterations(self, I, ite):
    return self.weights_standard(I, ite * self.parent.beta)

  # Applies the vector map to a structure key. It applies also it's appropriate multiplier
  # @I: The referring image
  # @vector: The vector tracking enegy function
  # returns: a dictionary that associates a structure key to an array of weights
  def weights_vector(self, I, V):
    return self.weights_standard(I, V * self.parent.delta)

  def weights_frame_iterations(self, I, ite):
    vectors = [(2, 1, 1)]  # left to right
    links = np.zeros((1,) + I.shape)
    links[0, 0:-1, :, :] = ite * self.parent.beta
    return self.to_hash(vectors, links)

  def weights_deepness(self, I):
    vectors = [(2, 1, 1), (0, 1, 1)]
    links = np.zeros((2,) + I.shape)
    # In profondità sx
    links[0, 0:-1, :, 1:-1] = np.abs(((I[0:-1, :, 1:-1] + I[0:-1, :, 2:]) * 0.5) - ((I[1:, :, 0:-2] + I[1:, :, 1:-1]) * 0.5))
    # In profondità dx
    links[1, 1:, :, 1:-1] = np.abs(((I[0:-1, :, 0:-2] + I[0:-1, :, 1:-1]) * 0.5) - ((I[1:, :, 1:-1] + I[1:, :, 2:]) * 0.5))
    return self.to_hash(vectors, links)

  def weights_diagonal(self, I):
    vectors = [(0, 1, 2), (2, 1, 2)]
    energy = (I[:, :, 0:-1] - (I[:, :, 0:-1] + I[:, :, 1:]) * 0.5) ** 2
    links = np.zeros((2,) + I.shape)

    links[0, 1:, :, 0:-1] = energy[0:-1]
    links[1, :, :, 0:-1] = energy

    links = links / self.parent.alpha
    return self.to_hash(vectors, links)

  # Given a bitmask list of methods and all the useful energy functions, generates a tuple of dictionaries,
  # that create an associations between a structure key and it's own energy function
  # @I: The referring image (skeleton)
  # @Imp: The importance map
  # @ite: The iteration counter energy function
  # @V: The vector tracking enegy function
  # @methods: A bit mask to identify which method should be actived
  # returns: a dictionary that associates a structure key to an array of weights
  #
  # Example:
  # @I [[2, 1, 0], [0, 1, 3], [2, 2, 2]]
  # @Imp [[2, 2], [1, 3], [0, 0]]
  # @ite [[2, 1], [1, 1], [2, 4]]
  # @V [[0, 0], [0, 0], [0, 0]]
  # @methods vs.IMP | vs.ITE
  # returns  ({(1, 1, 2): [[2, 2, 0], [1, 3, 0], [0, 0, 0]]}, {(1, 1, 2): [[2, 1, 0], [1, 1, 0], [2, 4, 0]]})
  def select_methods(self, I, Imp, ite, V, methods):
    all_weights = ()
    if (vs.STR & methods) != 0:
      all_weights += (self.weights_structure(I),)
    if (vs.IMP & methods) != 0:
      all_weights += (self.weights_importance(I, Imp),)
    if (vs.ITE & methods) != 0:
      all_weights += (self.weights_iterations(I, ite),)
    if (vs.FIT & methods) != 0:
      all_weights += (self.weights_frame_iterations(I, ite),)
    if (vs.DEE & methods) != 0:
      all_weights += (self.weights_deepness(I),)
    if (vs.DIA & methods) != 0:
      all_weights += (self.weights_diagonal(I),)
    if (vs.VEC & methods) != 0:
      all_weights += (self.weights_vector(I, V),)
    if (vs.TIM & methods) != 0:
      all_weights += (self.weights_structure_time(I),)
    return all_weights
