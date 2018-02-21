from itertools import product

from numpy import sum

from multi.gfm import get_levelset_root

from options import CFL, NDIM, NV
from system import max_eig


def make_u(mats):
    """ Builds u across the domain, from the different material grids
    """
    m = len(mats)
    u = mats[0]
    N = NV - (m - 1)
    u[:, :, :, N:] = sum([mat[:, :, :, N:] for mat in mats], axis=0) / m

    for i in range(1, m):
        ind = get_levelset_root(u, i - 1, m)
        u[ind:, :, :, :N] = mats[i][ind:, :, :, :N]

    return u


def timestep(mats, count, t, tf, dX, MPs):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(mats)
    MAX = 0
    for ind in range(m):

        u = mats[ind]
        MP = MPs[ind]
        nx, ny, nz = u.shape[:3]

        for i, j, k in product(range(nx), range(ny), range(nz)):

            Q = u[i, j, k]
            MAX = max(MAX, max_eig(Q, 0, MP) / dX[0])
            if NDIM > 1:
                MAX = max(MAX, max_eig(Q, 1, MP) / dX[1])
                if NDIM > 2:
                    MAX = max(MAX, max_eig(Q, 2, MP) / dX[2])

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt
