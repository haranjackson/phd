from numpy import arange, array, concatenate, prod

from etc.grids import extend_grid
from options import NDIM, N, NV


def standard_BC(u, wall=[0]*NDIM):
    """ Extends the grid u in all dimensions. If wall[d]=1 then the
        boundaries in dimension d are no-slip, else they are transmissive.
    """
    ret = u.copy()
    endCells = concatenate([arange(N), arange(-N,0)])
    reflectVars = array([2,3,4,14,15,16])

    for d in range(NDIM):

        wall_ = wall[d] != 0
        ret = extend_grid(ret, N, d, wall_)

        if wall_:
            shape = ret.shape
            n1 = prod(shape[:d])
            n2 = shape[d]
            n3 = prod(shape[d + 1 : NDIM])

            ret.reshape([n1, n2, n3, NV])[:, endCells, :, reflectVars] *= -1

    return ret


def periodic_BC(u):

    ret = u.copy()

    for d in range(NDIM):
        ret = extend_grid(ret, N, d, 2)

    return ret
