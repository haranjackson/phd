from numpy import arange, array, concatenate, dot, flip, int64, ones, prod, zeros
from numpy.linalg import det, svd

from options import NDIM, N, NV


def extend_grid(arr, ext, d, kind):
    """ Extends the arr by ext cells on each surface in direction d.
        kind=0: transmissive, kind=1: no-slip, kind=2: periodic
    """
    n = arr.shape[d]

    if kind == 0:
        reps = concatenate((zeros(ext),
                            arange(n),
                            (n - 1) * ones(ext))).astype(int64)
    elif kind == 1:
        reps = concatenate((flip(arange(ext), 0),
                            arange(n),
                            flip(arange(n - ext, n), 0))).astype(int64)

    elif kind == 2:
        reps = concatenate((arange(n - ext, n),
                            arange(n),
                            arange(ext))).astype(int64)

    return arr.take(reps, axis=d)


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


def destress(Q, MP):
    """ Removes the stress and associated energy from state Q
    """
    A = Q[5:14].reshape([3,3])
    detA = det(A)
    U, _, Vh = svd(A)
    Q[5:14] = detA**(1 / 3) * dot(U, Vh).ravel()
