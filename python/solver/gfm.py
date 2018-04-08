from itertools import product

from numpy import sum, zeros

from gpr.multi.riemann import star_states


def make_u(mats):
    """ Builds u across the domain, from the different material grids
    """
    u = zeros(mats[0].shape)
    NDIM = len(u.shape) - 1
    av = sum(mats, axis=0) / len(mats)

    for coords in product(*[range(s) for s in av.shape[:NDIM]]):

        materialIndex = get_material_index(av[coords])
        u[coords] = mats[materialIndex][coords]

    return u


def get_levelset_root(u, i, m):
    """ return the location of interface i
    """
    φ = u[:, :, :, i - (m - 1)]
    n = len(φ)
    for j in range(n):
        if φ[j] >= 0:
            return j
    return n


def add_ghost_cells(mats, dt, MPs, ISO_FIX=0):

    m = len(MPs)
    for i in range(m - 1):
        uL = mats[i]
        uR = mats[i + 1]
        ind = get_levelset_root(uL, i, m)
        MPL = MPs[i]
        MPR = MPs[i + 1]

        QL = uL[ind - 1 - ISO_FIX, 0, 0, :-(m - 1)]
        QR = uR[ind + ISO_FIX, 0, 0, :-(m - 1)]

        print(QL, QR)
        QL_, QR_ = star_states(QL, QR, dt, MPL, MPR)

        for j in range(ind, len(uL)):
            uL[j, 0, 0, :-(m - 1)] = QL_
        for j in range(ind):
            uR[j, 0, 0, :-(m - 1)] = QR_
