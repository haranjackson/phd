from numpy import arange, array, concatenate, prod

from etc.grids import extend_grid


def standard_BC(u, N, NDIM, wall=None):
    """ Extends the grid u in all dimensions. If wall[d]=1 then the
        boundaries in dimension d are no-slip, else they are transmissive.
    """
    if wall is None:
        wall = [0]*NDIM

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

            ret.reshape(n1, n2, n3, -1)[:, endCells, :, reflectVars] *= -1

    return ret


def periodic_BC(u, N, NDIM):

    ret = u.copy()

    for d in range(NDIM):
        ret = extend_grid(ret, N, d, 2)

    return ret
