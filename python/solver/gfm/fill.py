from itertools import product

from numpy import array, logical_or, ones, prod, zeros
from skfmm import distance

from ader.etc.boundaries import neighbor_cells

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

    d = 1.5

    xip = xp - φ[ind] * n
    xL = xip - d * dx * n
    xR = xip + d * dx * n
    xp_ = xp - 2 * φ[ind] * n

    # TODO: replace with interpolated values
    iL = array(xL / dx, dtype=int)
    iR = array(xR / dx, dtype=int)
    i_ = array(xp_ / dx, dtype=int)

    return iL, iR, i_


def fill_boundary_cells(u, grids, intMask, i, φ, Δφ, dx, MPL, MPR):

    for ind in product(*[range(s) for s in intMask.shape]):

        if intMask[ind] != 0:
            iL, iR, i_ = boundary_inds(ind, φ, Δφ, dx)

            # TODO: rotate vector quantities towards the normal
            QL = u[tuple(iL)][:17]
            QR = u[tuple(iR)][:17]
            QL_, QR_ = star_states(QL, QR, MPL, MPR)

        # TODO: investigate where QR_, QL_ should be reversed, and if the
        # inside cell should be filled
        if intMask[ind] == -1:
            grids[i][ind][:17] = QL_
            grids[i][tuple(i_)][:17] = QL_

        elif intMask[ind] == 1:
            grids[i+1][ind][:17] = QR_
            grids[i+1][tuple(i_)][:17] = QR_


def fill_from_neighbor(grid, Δφ, ind, dx, sgn):
    """ makes the value of cell ind equal to the value of its neighbor in the
        direction of the interface
        TODO: replace with interpolated values
    """
    n = normal(Δφ[ind])
    x = (array(ind) + 0.5) * dx
    xn = x + sgn * dx * n
    grid[ind] = grid[array(xn / dx, dtype=int)]


def fill_neighbor_cells(grids, intMask, i, Δφ, dx, N, NDIM):

    shape = intMask.shape
    inds = [range(s) for s in shape]

    for N0 in range(1, N + 1):
        for ind in product(*inds):

            if intMask[ind] == 0:

                neighbors = neighbor_cells(intMask, ind)

                if N0 in neighbors:
                    intMask[ind] = N0 + 1
                    fill_from_neighbor(grids[i], Δφ, ind, dx, -1)

                if -N0 in neighbors:
                    intMask[ind] = -(N0 + 1)
                    fill_from_neighbor(grids[i+1], Δφ, ind, dx, 1)


def fill_ghost_cells(u, m, N, dX, MPs):

    NDIM = u.ndim - 1
    shape = u.shape[:-1]
    ncells = prod(shape)

    grids = [u.copy() for i in range(m)]
    masks = [ones(shape, dtype=bool) for i in range(m)]
    dx = dX[0]

    for i in range(m - 1):

        MPL = MPs[i]
        MPR = MPs[i+1]

        intMask = find_interface_cells(u, i, m)
        φ = distance(u.take(i - (m-1), axis=-1), dx=dx)
        Δφ = finite_difference(φ, dX)

        fill_boundary_cells(u, grids, intMask, i, φ, Δφ, dx, MPL, MPR)
        fill_neighbor_cells(grids, intMask, i, Δφ, dx, N, NDIM)

        masks[i] *= logical_or((φ <= 0), (intMask == 1))
        masks[i+1] *= logical_or((φ >= 0), (intMask == -1))

        for j in range(m):
            grids[j].reshape([ncells, -1])[:, i - (m-1)] = φ

    return grids, masks
