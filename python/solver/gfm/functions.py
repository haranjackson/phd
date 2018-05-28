from numpy import array, maximum, stack
from numpy.linalg import norm
from skfmm import distance


def boundary_inds(ind, φ, n, dX):
    """ Calculates indexes of the boundary states at position given by ind
    """
    xp = (array(ind) + 0.5) * dX

    d = 1.

    xi = xp - φ[ind] * n    # interface position
    xL = xi - d * dX * n    # probe on left side
    xR = xi + d * dX * n    # probe on right side

    # TODO: replace with interpolated values
    ii = array(xi / dX, dtype=int)
    iL = array(xL / dX, dtype=int)
    iR = array(xR / dX, dtype=int)

    return ii, iL, iR


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


def renormalize_levelsets(u, nmat, dX, ncells):

    for i in range(nmat - 1):
        ind = i - (nmat - 1)
        φ = u.take(ind, axis=-1)
        u.reshape([ncells, -1])[:, ind] = distance(φ, dx=dX, order=1).ravel()


def material_indicator(u, mat, nmat, dX):

    φs = [-u.take(i - (nmat-1), axis=-1) for i in range(mat)] + \
        [u.take(i - (nmat-1), axis=-1) for i in range(mat, nmat-1)]

    if len(φs) > 1:
        φ = maximum(*φs)
        return distance(φ, dx=dX)
    else:
        return φs[0]
