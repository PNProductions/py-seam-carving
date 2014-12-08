import numpy as np


class structure_delegate(object):
  """Given a shape, generates any kind of structure matrix with that shape"""
  def __init__(self, shape):
    super(structure_delegate, self).__init__()
    self.shape = shape

  ## Given a position and a value, generates a matrix of zeros with the value at the given position
  # @position: a tuple of values (indexes)
  # @value: value to be set in that position
  def one_directional_structure(self, position, value=1):
    result = np.zeros(self.shape)
    result[position] = value
    return result

  ## Creates an associative array that gives, that associates every structure to it's
  # own representation with a matrix
  # Example:
  # @structure_list [(2, 1, 0), (0, 0, 0)]
  # @value 1
  # Result:
  # {
  #   (0, 0, 0) => [[[1, 0, 0],
  #                  [0, 0, 0],
  #                  [0, 0, 0]
  #                 ],
  #                 [[0, 0, 0],
  #                  [0, 0, 0],
  #                  [0, 0, 0]
  #                 ],
  #                 [[0, 0, 0],
  #                  [0, 0, 0],
  #                  [0, 0, 0]
  #                 ]
  #                ],
  #   (2, 1, 0) => [[[0, 0, 0],
  #                  [0, 0, 0],
  #                  [0, 0, 0]
  #                 ],
  #                 [[0, 0, 0],
  #                  [0, 0, 0],
  #                  [0, 0, 0]
  #                 ],
  #                 [[0, 0, 0],
  #                  [1, 0, 0],
  #                  [0, 0, 0]
  #                 ]
  #                ]
  # }
  def batch_structures(self, structure_list, value=1):
    structures = {}
    for structure in structure_list:
      structures[structure] = self.one_directional_structure(structure, value)
    return structures

  ## Creates a matrix that is the combination of multiple structure matrixes, given a
  # structure list
  # Example:
  # @structure_list [(2, 1, 0), (0, 0, 0)]
  # @value 1
  # Result:
  #
  # [
  #  [[1, 0, 0],
  #   [0, 0, 0],
  #   [0, 0, 0]
  #  ],
  #  [[0, 0, 0],
  #   [0, 0, 0],
  #   [0, 0, 0]
  #  ],
  #  [[0, 0, 0],
  #   [1, 0, 0],
  #   [0, 0, 0]
  #  ]
  # ]
  def merged_structures(self, structure_list, value=1):
    result = np.zeros(self.shape)
    for structure in structure_list:
      result += self.one_directional_structure(structure, value)
    return result

  # Legend:
  # ((left), (right)) or ((up), (down)) or ((forward), (backward))
  # * = background nodes, o = foreground nodes
  #
  #
  # ((1, 1, 0), (1, 1, 2)):
  #     * ------ *
  #
  #   o ----- o
  #
  #     * ----- *
  #
  #   o ----- o
  #
  # ((2, 1, 1), (0, 1, 1))
  #     *       *
  #    /       /
  #   o       o
  #
  #     *       *
  #    /       /
  #   o       o
  #
  # ((0, 1, 2), (2, 1, 2))
  #     *-_  _->*
  #      _ X_
  #   o -    ->o
  #
  #     *-_  _->*
  #      _ X_
  #   o -    ->o
  # ((0, 1, 0), (2, 1, 0))
  #    *<-_  _-*
  #      _ X_
  #   o<-    -o
  #
  #    *<-_  _-*
  #      _ X_
  #   o<-    -o
  # ((2, 1, 0), (2, 1, 2))
  #     *-_  _- *
  #      _ X_
  #   o-     -o
  #
  #     *-_  _- *
  #      _ X_
  #   o-     -o
  #
  # ((1, :, 0), (1, :, 0))
  #     *       *
  #
  #   o <----- o
  #      \  /
  #     * \/     *
  #       /\
  #      /  \
  #    o <---- o
  #
  #  ((1, 0, 1), (1, 2, 1)):
  #     *       *
  #     |       |
  #   o |     o |
  #   | |     | |
  #   | *     | *
  #   |       |
  #   o       o
