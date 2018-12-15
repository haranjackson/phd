from numpy import array, int32

import GPRpy

import test

from plot import *
from store import save_results


T = test.aluminium_plates

N = 2
cfl = 0.5
SPLIT = False
FLUX = 0
contorted_tol = 1.
HALF_STEP = True


assert(GPRpy.options.N() == N)

u0, MPs, tf, dX, bcs = T()

ndim = u0.ndim - 1
nX = array(u0.shape[:-1], dtype=int32)

BOUNDARIES = {'transitive': array([0] * 2 * ndim, dtype=int32),
              'periodic':   array([1] * 2 * ndim, dtype=int32),
              'slip':       array([2] * 2 * ndim, dtype=int32),
              'stick':      array([3] * 2 * ndim, dtype=int32),
              'lid_driven': array([3, 3, 3, 4], dtype=int32),
              'symmetric':  array([5, 5, 5, 5], dtype=int32),
              'half': array([5, 0, 0, 0], dtype=int32),
              }


sol = GPRpy.solvers.iterator(u0.ravel(), tf, nX, array(dX), cfl,
                             BOUNDARIES[bcs], SPLIT, HALF_STEP, False, FLUX,
                             MPs, contorted_tol)

for i in range(100):
    try:
        uList = [s.reshape(u0.shape) for s in sol[:100-i]]
        break
    except:
        pass

save_results(uList, T, N, cfl, SPLIT, FLUX)
