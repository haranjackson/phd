from itertools import product

from numpy import arange, array, zeros

from solvers.basis import WGHTS


def quad_weights(N, NDIM, time_rec=True):
    """ WGHT contains quadrature weights for integration over a spacetime cell
        (the cartesian product of the temporal and spatial quadrature weights)

        WGHT_END contains quadrature weights for integration over the
        spacetime hypersurface normal to a particular spatial direction.
        Th factor of 0.5 comes from the factor of 1/2 in fluxes - applied here
        for numerical reasona
    """
    TWGHTS = WGHTS if time_rec else array([1])
    TN = len(TWGHTS)

    WGHT = zeros([TN] + [N] * NDIM)
    coordList = [arange(N)] * NDIM

    for t in range(TN):
        for coords in product(*coordList):

            wght = TWGHTS[t]
            for d in range(NDIM):
                wght *= WGHTS[coords[d]]

            WGHT[(t,) + coords] = wght

    WGHT_END = 0.5 * WGHT.sum(axis=-1).ravel()

    return TN, WGHT, WGHT_END
