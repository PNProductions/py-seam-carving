from videoseam.delegate.structure import structure_delegate as sd
from videoseam.delegate.weights import weights_delegate as wd

import maxflow
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseGraph(object):
  __metaclass__ = ABCMeta

  def __init__(self, parent, methods, I, Imp=None, ite=None, V=None, dtype=float, reverse=False):
    super(BaseGraph, self).__init__()
    self.parent = parent
    self.methods = methods
    self.I = I
    self.Imp = Imp
    self.ite = ite
    self.V = V
    if dtype == float:
      self.inf, self.multiplier = np.inf, 1
    elif dtype == int:
      self.inf, self.multiplier = 1000000000, 100000
    self.reverse = reverse
    self.graph = maxflow.Graph[dtype]()
    self.init_structure()

  @abstractmethod
  def init_structure(self):
    pass

  def build_tedges(self, nodeids):
    self.graph.add_grid_tedges(nodeids[..., 0], self.inf, 0)
    self.graph.add_grid_tedges(nodeids[..., -1], 0, self.inf)

  def weights_correction(self, weights):
    weights = np.copy(weights)
    if self.multiplier != 1:
      weights *= self.multiplier
      weights[np.where(weights == np.inf)] = self.inf
      weights[np.where(weights > self.inf)] = self.inf
    return weights.max() - weights if self.reverse else weights

  def build_edges(self, nodeids, methods_weights):
    for key, structure in self.structures.iteritems():
      for weights in methods_weights:
        if key in weights:
          weights = self.weights_correction(weights[key])
          self.graph.add_grid_edges(nodeids, structure=structure, weights=weights, symmetric=False)

  def build_infinite_edges(self, nodeids):
    self.graph.add_grid_edges(nodeids, structure=self.structures_inf)

  def build(self):
    nodeids = self.graph.add_grid_nodes(self.I.shape)
    self.methods_weights = self.wd.select_methods(self.I, self.Imp, self.ite, self.V, self.methods)
    self.build_edges(nodeids, self.methods_weights)
    self.build_infinite_edges(nodeids)
    self.build_tedges(nodeids)
    return self.graph, nodeids


class Graph2D(BaseGraph):
  def init_structure(self):
    self.sd = sd((3, 3))
    self.wd = wd(self.parent, 0, ndim=2) if self.reverse else wd(self.parent, ndim=2)
    self.structures = self.sd.batch_structures([(1, 2), (2, 1), (0, 1)])
    self.structures_inf = self.sd.merged_structures([(Ellipsis, 0)], self.inf)


class Graph3D(BaseGraph):
  def init_structure(self):
    self.sd = sd((3, 3, 3))
    self.wd = wd(self.parent, 0) if self.reverse else wd(self.parent)
    self.structures = self.sd.batch_structures([(1, 1, 2), (0, 1, 2), (2, 1, 2), (2, 1, 1), (0, 1, 1)])
    self.structures_inf = self.sd.merged_structures([(1, Ellipsis, 0), (0, 1, 0), (2, 1, 0)], self.inf)


class ReversedGraph3D(Graph3D):
  def build_tedges(self, nodeids):
    self.graph.add_grid_tedges(nodeids[0], self.inf, 0)
    self.graph.add_grid_tedges(nodeids[-1], 0, self.inf)

  def init_structure(self):
    super(ReversedGraph3D, self).init_structure()
    self.structures_inf = self.sd.merged_structures([(0, 1, Ellipsis), (0, 0, 1), (0, 2, 1)], self.inf)
