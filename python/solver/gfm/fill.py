from itertools import product

from numpy import array, logical_or, zeros
from skfmm import distance

from gpr.multi.riemann import star_states
from solver.gfm.functions import finite_difference, normal, sign


def find_interface_cells(u, i, m):
    """ Finds the cells lying on the ith interface,
        given that there are m materials
    """
    NDIM = u.ndim - 1
    ii = i - (m - 1)
    shape = u.shape[:-1]
    mask = zeros(shape)

    for indsL in product(*[range(s) for s in shape]):
        for d in range(NDIM):

            if indsL[d] < shape[d] - 1:
                indsR = indsL[:d] + (indsL[d] + 1,) + indsL[d + 1:]
                φL = u[indsL][ii]
                φR = u[indsR][ii]

                if φL * φR <= 0:
                    mask[indsL] = sign(φL)
                    mask[indsR] = sign(φR)

    return mask


def boundary_inds(ind, φ, Δφ, dx):
    """ Calculates indexes of the boundary states at position given by ind
    """
    xp = (array(ind) + 0.5) * dx
    n = normal(Δφ[ind])
    absφ = abs(φ[ind])

    xip = xp + absφ * n
    xL = xip - 1.5 * dx * n
    xR = xip + 1.5 * dx * n
    xp_ = xp + 2 * absφ * n

    # TODO: replace with interpolated values
    iL = array(xL / dx, dtype=int)
    iR = array(xR / dx, dtype=int)
    i_ = array(xp_ / dx, dtype=int)

    return iL, iR, i_


def fill_boundary_cells(u, ret, intMask, i, φ, Δφ, dx, indList, MPL, MPR):

    for ind in product(*indList):

        if intMask[ind] != 0:
            iL, iR, i_ = boundary_inds(ind, φ, Δφ, dx)

            # TODO: rotate vector quantities towards the normal
            QL = u[tuple(iL)].take(range(17))
            QR = u[tuple(iR)].take(range(17))
            QL_, QR_ = star_states(QL, QR, MPL, MPR)

        if intMask[ind] == -1:
            ret[i][ind][:17] = QR_

        elif intMask[ind] == 1:
            ret[i+1][ind][:17] = QL_


def fill_from_neighbor(u, Δφ, ind, dx):
    """ makes the value of cell ind equal to the value of its neighbor in the
        direction of the interface
        TODO: replace with interpolated values
    """
    n = normal(Δφ[ind])
    x = (array(ind) + 0.5) * dx
    xn = x + dx * n
    u[ind] = u[array(xn / dx, dtype=int)]


def fill_neighbor_cells(ret, Δφ, dx, N, NDIM, indList, intMask):

    for N0 in range(1, N-1):
        for ind in product(*indList):

            if intMask[ind] == 0:

                neighbors = [intMask[ind[:d] + (ind[d] - 1,) + ind[d + 1:]]
                             for d in range(NDIM)] + \
                            [intMask[ind[:d] + (ind[d] + 1,) + ind[d + 1:]]
                             for d in range(NDIM)]

                if N0 in neighbors:
                    intMask[ind] = N0 + 1
                    fill_from_neighbor(ret[i+1], Δφ, ind, dx)

                if -N0 in neighbors:
                    intMask[ind] = -(N0 + 1)
                    fill_from_neighbor(ret[i], Δφ, ind, dx)


def fill_ghost_cells(u, m, N, dX, MPs):

    NDIM = u.ndim - 1
    shape = u.shape[:-1]
    ret = [u.copy() for i in range(m)]
    masks = [ones(shape, dtype=bool) for i in range(m)]
    dx = dX[0]

    for i in range(m - 1):

        MPL = MPs[i]
        MPR = MPs[i+1]

        intMask = find_interface_cells(u, i, m)
        φ = distance(u.take(i - (m-1), axis=-1), dx=dx)
        Δφ = finite_difference(φ, dX)
        indList = [range(s) for s in intMask.shape]

        fill_boundary_cells(u, ret, intMask, i, φ, Δφ, dx, indList, MPL, MPR)
        fill_neighbor_cells(ret, Δφ, dx, N, NDIM, indList, intMask)

        maskL = logical_or((φ <= 0), (intMask == 1))
        maskR = logical_or((φ >= 0), (intMask == -1))

        masks[i] *= maskL
        masks[i+1] *= maskR

    return ret, masks
