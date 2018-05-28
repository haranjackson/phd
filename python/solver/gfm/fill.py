from itertools import product

from numpy import array, logical_or, prod, zeros

from ader.etc.boundaries import neighbor_cells

from gpr.opts import LSET
from gpr.multi import get_material_index
from gpr.multi.riemann import star_states
from solver.gfm.functions import finite_difference, normal, sign, boundary_inds, \
    renormalize_levelsets, material_indicator


def neighbors(inds, nX, ndim):

    CORNERS = False  # whether to consider corners as neighbors

    if ndim == 1:
        if inds[0] > 0:
            return [(inds[0] - 1,)]
        else:
            return []

    elif ndim ==2:
        ret = []

        if inds[0] > 0:
            ret.append((inds[0] -1, inds[1]))

            if CORNERS:
                if inds[1] > 0:
                    ret.append((inds[0] - 1, inds[1] - 1))
                if inds[1] < nX[1] - 1:
                    ret.append((inds[0] -1, inds[1] + 1))

        if inds[1] > 0:
            ret.append((inds[0], inds[1] - 1))

        return ret


def find_interface_cells(φ):
    """ Finds the cells lying on the interface of material i of m
        intMask = -1 on the inside and intMask = 1 on the outside
    """
    shape = φ.shape
    ndim = φ.ndim
    intMask = zeros(shape)

    for indsL in product(*[range(s) for s in shape]):

        for indsR in neighbors(indsL, shape, ndim):

            φL = φ[indsL]
            φR = φ[indsR]

            if φL * φR <= 0:
                intMask[indsL] = sign(φL)
                intMask[indsR] = sign(φR)

    return intMask


def fill_boundary_cells(u, grid, intMask, mat, φ, Δφ, dX, MPs, dt):

    MPL = MPs[mat]

    for ind in product(*[range(s) for s in intMask.shape]):

        if intMask[ind] == 1:

            n = normal(Δφ[ind])
            ii, iL, iR = boundary_inds(ind, φ, n, dX)

            QL = u[tuple(ii)]
            QR = u[tuple(iR)]

            # in case of thin strip of the inside material
            miL = get_material_index(QL, len(MPs))
            miR = get_material_index(QR, len(MPs))
            if miL == miR:
                QL = u[tuple(iL)]

            MPR = MPs[miR]
            QL_, QR_ = star_states(QL, QR, MPL, MPR, dt, n)

            grid[ind][:-LSET] = QL_[:-LSET]
            grid[tuple(ii)][:-LSET] = QL_[:-LSET]


def fill_neighbor_cells(grid, intMask, Δφ, dX, N):
    """ makes the value of cell ind equal to the value of its neighbor in the
        direction of the interface
        TODO: replace with interpolated values
    """
    shape = intMask.shape
    inds = [range(s) for s in shape]

    for N0 in range(1, N + 1):
        for ind in product(*inds):

            if intMask[ind] == 0:

                neighbors = neighbor_cells(intMask, ind)

                if N0 in neighbors:
                    intMask[ind] = N0 + 1
                    n = normal(Δφ[ind])
                    x = (array(ind) + 0.5) * dX
                    xn = x - dX * n
                    indn = tuple(array(xn / dX, dtype=int))
                    grid[ind][:-LSET] = grid[indn][:-LSET]


def fill_ghost_cells(grids, masks, u, nmat, N, dX, MPs, dt):

    ncells = prod(u.shape[:-1])
    renormalize_levelsets(u, nmat, dX, ncells)

    for mat in range(nmat):

        if MPs[mat].EOS > -1:  # not a vacuum

            φ = material_indicator(u, mat, nmat, dX)
            Δφ = finite_difference(φ, dX)
            intMask = find_interface_cells(φ)

            grids[mat] = u.copy()
            grid = grids[mat]
            fill_boundary_cells(u, grid, intMask, mat, φ, Δφ, dX, MPs, dt)
            fill_neighbor_cells(grid, intMask, Δφ, dX, N)

            masks[mat] = logical_or((φ <= 0), (intMask == 1))

            grid.reshape([ncells, -1])[:, - (nmat - 1)
                         :] = u.reshape([ncells, -1])[:, - (nmat - 1):]

        else:
            masks[mat] *= False
