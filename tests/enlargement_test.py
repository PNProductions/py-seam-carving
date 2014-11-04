import unittest
from numpy import array, r_, zeros
import scipy.io
from numpy.testing import assert_array_equal
from seammerging.enlargement import SeamMergingWithDecompositionEnlargement

X = zeros((5, 5, 3))
X[:, :, 0] = array([r_[1:6], r_[6:11], r_[11:16], r_[16: 21], r_[21:26]])
X[:, :, 1] = X[:, :, 0] * 2
X[:, :, 2] = X[:, :, 0] * 3
S = X[:, :, 0] + 25
T = X[:, :, 0] + 50
subject = SeamMergingWithDecompositionEnlargement(X, S, T, 2, 0, 0.5, 0.5)


class SeamMergingWithDecompositionTest(unittest.TestCase):
  def test_applySeamMerging(self):
    p = scipy.io.loadmat('./tests/mats/enlarge/applySeamMergingEnlarg.mat')
    I = p['I'].T[0] - 1
    q11, q12, p12, p22, Simg, Z = subject.apply_seam_merging(I, p['q11'], p['upQ11'], p['q12'], p['upQ12'], p['p12'], p['upP12'], p['p22'], p['upP22'], p['Simg'], p['v'], p['Z'])
    assert_array_equal(q11, p['q11Copy'])
    assert_array_equal(q12, p['q12Copy'])
    assert_array_equal(p22, p['p22Copy'])
    assert_array_equal(Simg, p['SimgCopy'])
    assert_array_equal(Z, p['ZCopy'])

  def test_generateEastEnergy(self):
    p = scipy.io.loadmat('./tests/mats/enlarge/generateEastConnectionsEnlarg.mat')
    CE = subject.generateEastEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 6], p['upP22'][:, :, 6])
    assert_array_equal(CE, p['CE'])

  def test_generateNorthEnergy(self):
    p = scipy.io.loadmat('./tests/mats/enlarge/generateNorthConnectionsEnlarg.mat')
    CNcc, CNcnCL, CNcnCR = subject.generateNorthEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 4], p['upP22'][:, :, 4])
    assert_array_equal(CNcc, p['CNcc'])
    assert_array_equal(CNcnCL, p['CNcnCL'])
    assert_array_equal(CNcnCR, p['CNcnCR'])

  def test_generateSouthEnergy(self):
    p = scipy.io.loadmat('./tests/mats/enlarge/generateSouthConnectionsEnlarg.mat')
    CScc, CScnCL, CScnCR = subject.generateSouthEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 5], p['upP22'][:, :, 5])
    assert_array_equal(CScc, p['CScc'])
    assert_array_equal(CScnCL, p['CScnCL'])
    assert_array_equal(CScnCR, p['CScnCR'])

  def test_generateWestEnergy(self):
    p = scipy.io.loadmat('./tests/mats/enlarge/generateWestConnectionsEnlarg.mat')
    CW = subject.generateWestEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 7], p['upP22'][:, :, 7])
    assert_array_equal(CW, p['CW'])
