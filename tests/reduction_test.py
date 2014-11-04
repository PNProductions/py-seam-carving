import unittest
from numpy import array, r_, zeros
import scipy.io
from numpy.testing import assert_array_equal
from seammerging.reduction import SeamMergingWithDecomposition

X = zeros((5, 5, 3))
X[:, :, 0] = array([r_[1:6], r_[6:11], r_[11:16], r_[16: 21], r_[21:26]])
X[:, :, 1] = X[:, :, 0] * 2
X[:, :, 2] = X[:, :, 0] * 3
S = X[:, :, 0] + 25
T = X[:, :, 0] + 50
subject = SeamMergingWithDecomposition(X, S, T, 2, 0, 0.5, 0.5)


class SeamMergingWithDecompositionTest(unittest.TestCase):
  def test_applySeamMerging(self):
    p = scipy.io.loadmat('./tests/mats/applySeamMerging.mat')
    I = p['I'].T[0] - 1
    q11, q12, p12, p22, Simg, Z = subject.apply_seam_merging(I, p['q11'], p['upQ11'], p['q12'], p['upQ12'], p['p12'], p['upP12'], p['p22'], p['upP22'], p['Simg'], p['v'], p['Z'])
    assert_array_equal(q11, p['q11Copy'])
    assert_array_equal(q12, p['q12Copy'])
    assert_array_equal(p22, p['p22Copy'])
    assert_array_equal(Simg, p['SimgCopy'])
    assert_array_equal(Z, p['ZCopy'])

  def test_calculatePot(self):
    p = scipy.io.loadmat('./tests/mats/calculatePot.mat')
    Pot = subject.calculatePot(p['CW'], p['CE'], p['alphaN'], p['imp'], p['gammaN'], p['ite'], p['betaN'])
    assert_array_equal(Pot, p['Pot'])

  def test_dynamic_programming(self):
    p = scipy.io.loadmat('./tests/mats/dynamic_programming.mat')
    pathMap = zeros(p['Pot_old'].shape)
    Pot = subject.dynamic_programming(p['Pot_old'], p['CU'], p['CL'], p['CR'], pathMap)
    assert_array_equal(Pot, p['Pot'])
    pathMap[1:] = pathMap[1:] + 1
    assert_array_equal(pathMap, p['pathMap'])

  def test_generateEastEnergy(self):
    p = scipy.io.loadmat('./tests/mats/generateEastConnections.mat')
    CE = subject.generateEastEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 6], p['upP22'][:, :, 6])
    assert_array_equal(CE, p['CE'])

  def test_generateEnergyUpLeftRight(self):
    p = scipy.io.loadmat('./tests/mats/generateEnergyUpLeftRight.mat')
    CU, CL, CR = subject.generateEnergyUpLeftRight(p['CScc'], p['CNcc'], p['CScnCL'], p['CNcnCL'], p['CScnCR'], p['CNcnCR'])
    assert_array_equal(CU, p['CU'])
    assert_array_equal(CL, p['CL'])
    assert_array_equal(CR, p['CR'])

  def test_generateNorthEnergy(self):
    p = scipy.io.loadmat('./tests/mats/generateNorthConnections.mat')
    CNcc, CNcnCL, CNcnCR = subject.generateNorthEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 4], p['upP22'][:, :, 4])
    assert_array_equal(CNcc, p['CNcc'])
    assert_array_equal(CNcnCL, p['CNcnCL'])
    assert_array_equal(CNcnCR, p['CNcnCR'])

  def test_generateSouthEnergy(self):
    p = scipy.io.loadmat('./tests/mats/generateSouthConnections.mat')
    CScc, CScnCL, CScnCR = subject.generateSouthEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 5], p['upP22'][:, :, 5])
    assert_array_equal(CScc, p['CScc'])
    assert_array_equal(CScnCL, p['CScnCL'])
    assert_array_equal(CScnCR, p['CScnCR'])

  def test_generateWestEnergy(self):
    p = scipy.io.loadmat('./tests/mats/generateWestConnections.mat')
    CW = subject.generateWestEnergy(p['Simg'], p['v'], p['upQ11'], p['upP12'][:, :, 7], p['upP22'][:, :, 7])
    assert_array_equal(CW, p['CW'])

  def test_initializeParameters(self):
    p = scipy.io.loadmat('./tests/mats/initializeParameters.mat')
    alphaN, gammaN, betaN = subject.initializeParameters(p['imp'], p['CU'], p['CW'], p['CE'], p['CL'], p['CR'])
    self.assertEqual(alphaN, p['alphaN'])
    self.assertEqual(gammaN, p['gammaN'])
    self.assertEqual(betaN, p['betaN'])

  def test_sumShifted(self):
    p = scipy.io.loadmat('./tests/mats/sumShifted.mat')
    upP22 = subject.sumShifted(p['p22'])
    assert_array_equal(upP22, p['upP22'])
