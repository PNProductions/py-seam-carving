import unittest
from numpy import arange, array
from numpy.testing import assert_array_equal
from seamcarving.video_reduce import video_seam_carving_decomposition

i_inf = 10000000
i_mult = 1
Image = arange(18).reshape(2, 3, 3)
subject = video_seam_carving_decomposition(Image, 0, 0, False)
g, nodeids = subject.generate_graph(Image)


class seam_carving_decompositionTest(unittest.TestCase):
  def test_generate_up_down_edges(self):
    links, structure = subject.generate_up_down_edges(Image, nodeids, i_inf, i_mult)
    print links
    assert_array_equal(links, array(([[[0, 2, 2], [0, 2, 2.], [0, 0, 0.]], [0, 2, 2.], [0, 2, 2.], [0, 0, 0.]])))
