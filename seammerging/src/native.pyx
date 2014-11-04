import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython


def improved_sum_shifted(dtype_t[:, ::1] a, dtype_t[:, ::1] b, dtype_t[:, :, ::1] c, dtype_t[:, :, ::1] d):
  cdef unsigned int i, j, k
  cdef unsigned int w = c.shape[0], h = c.shape[1] - 1, z = c.shape[2]
  cdef dtype_t[:, ::1] aa = np.empty((w, h), 'f8')
  cdef dtype_t[:, ::1] bb = np.empty((w, h), 'f8')
  cdef dtype_t[:, :, ::1] cc = np.empty((w, h, z), 'f8')
  cdef dtype_t[:, :, ::1] dd = np.empty((w, h, z), 'f8')
  with cython.boundscheck(False), cython.wraparound(False):
    for i in range(w):
      for j in range(h):
        aa[i, j] = a[i, j] + a[i, j + 1]
        bb[i, j] = b[i, j] + b[i, j + 1]
        for k in range(z):
          cc[i, j, k] = c[i, j, k] + c[i, j + 1, k]
          dd[i, j, k] = d[i, j, k] + d[i, j + 1, k]
  return np.asarray(aa), np.asarray(bb), np.asarray(cc), np.asarray(dd)
