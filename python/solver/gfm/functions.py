from numpy import stack
from numpy.linalg import norm


def sign(x):
    if x <= 0:
        return -1
    else:
        return 1


def finite_difference(φ, dX):
    """ ret[i,j,..][d] is the derivative in the dth direction in cell (i,j,...)
    """
    NDIM = φ.ndim

    ret = []
    for d in range(NDIM):
        n = φ.shape[d]
        tmp = φ.take(range(2, n), axis=d) - φ.take(range(n - 2), axis=d)
        inds = [0] + list(range(n-2)) + [n-3]
        ret.append(tmp.take(inds, axis=d) / (2 * dX[d]))

    return stack(ret, axis=-1)


def normal(Δφ):
    return Δφ / norm(Δφ)
