from itertools import product

from numpy import arange, array, zeros

from solvers.basis import WGHTS
from options import N, SPLIT, NDIM


TWGHTS = array([1]) if SPLIT else WGHTS
TN = len(TWGHTS)


def quad_weights():

    ret = zeros([TN] + [N]*NDIM)
    coordList = [arange(N)] * NDIM

    for t in range(TN):
        for coords in product(*coordList):

            wght = TWGHTS[t]
            for d in range(NDIM):
                wght *= WGHTS[coords[d]]

            ret[(t,) + coords] = wght

    return ret


# quadrature weights for integration over a spacetime cell
# (the cartesian product of the temporal and spatial quadrature weights)
WGHT = quad_weights()

# quadrature weights for integration over the hypersurface in time and
# space, normal to a particular spatial direction. factor of 0.5 comes from
# factor of 1/2 in fluxes - applied here for numerical reason
WGHT_END = 0.5 * WGHT.sum(axis=-1).ravel()
