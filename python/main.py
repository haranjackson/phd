from numpy import array, int32

import GPRpy

import test

from plot import *


T = test.barton(1)

DEBUG = False
N = 3
cfl = 0.5
SPLIT = True
FLUX = 0
contorted_tol = 1.


assert(GPRpy.options.N() == N)

u0, MPs, tf, dX, bcs = T()

ndim = u0.ndim - 1
nX = array(u0.shape[:-1], dtype=int32)

BOUNDARIES = {'transitive': array([0] * 2 * ndim, dtype=int32),
              'periodic':   array([1] * 2 * ndim, dtype=int32),
              'slip':       array([2] * 2 * ndim, dtype=int32),
              'stick':      array([3] * 2 * ndim, dtype=int32),
              'lid_driven': array([3, 3, 3, 4], dtype=int32),
              'symmetric':  array([5, 5, 5, 5], dtype=int32)
              }


sol = GPRpy.solvers.iterator(u0.ravel(), tf, nX, array(dX), cfl,
                             BOUNDARIES[bcs], SPLIT, True, False, FLUX, MPs,
                             contorted_tol)

uList = [s.reshape(u0.shape) for s in sol]
