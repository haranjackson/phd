from numpy import arange, concatenate, dot, flip, int64, ones, zeros
from numpy.linalg import det, svd

from options import NDIM, N


def extend(arr, ext, d, kind):
    """ Extends the input array by M cells on each surface
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

    return arr.take(reps, axis=d)


def standard_BC(u, reflect=0):
    ret = extend(u, N, 0, reflect)
    if reflect:
        ret[:N, :, :, 2:5] *= -1
        ret[-N:, :, :, 2:5] *= -1
        ret[:N, :, :, 14:17] *= -1
        ret[-N:, :, :, 14:17] *= -1
    if NDIM > 1:
        ret = extend(ret, N, 1, reflect)
        if reflect:
            ret[:, :N, :, 2:5] *= -1
            ret[:, -N:, :, 2:5] *= -1
            ret[:, :N, :, 14:17] *= -1
            ret[:, -N:, :, 14:17] *= -1
    if NDIM > 2:
        ret = extend(ret, 1, 2, reflect)
        if reflect:
            ret[:, :, :N, 2:5] *= -1
            ret[:, :, -N:, 2:5] *= -1
            ret[:, :, :N, 14:17] *= -1
            ret[:, :, -N:, 14:17] *= -1
    return ret


def periodic_BC(u):
    """ NEED TO UPDATE """
    ret = extend(u, 1, 0)
    ret[0] = ret[-2]
    ret[-1] = ret[1]
    if NDIM > 1:
        ret = extend(ret, 1, 1)
        ret[:, 0] = ret[:, -2]
        ret[:, -1] = ret[:, 1]
    if NDIM > 2:
        ret = extend(ret, 1, 2)
        ret[:, :, 0] = ret[:, :, -2]
        ret[:, :, -1] = ret[:, :, 1]
    return ret


def destress(Q, MP):
    """ Removes the stress and associated energy from state Q
    """
    A = Q[5:14].reshape([3,3])
    detA = det(A)
    U, _, Vh = svd(A)
    Q[5:14] = detA**(1 / 3) * dot(U, Vh).ravel()
