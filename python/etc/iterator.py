from itertools import product

from numpy import sum

from multi.gfm import get_levelset_root
from system.eigenvalues import max_abs_eigs
from options import nx, ny, nz, CFL, dx, dy, dz, ndim, nV


def make_u(mats):
    """ Builds u across the domain, from the different material grids
    """
    m = len(mats)
    u = mats[0]
    N = nV - (m-1)
    u[:,:,:,N:] = sum([mat[:,:,:,N:] for mat in mats], axis=0) / m

    for i in range(1, m):
        ind = get_levelset_root(u, i-1, m)
        u[ind:,:,:,:N] = mats[i][ind:,:,:,:N]

    return u

def timestep(mats, count, t, tf, MPs):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(mats)
    MAX = 0
    for ind in range(m):

        u = mats[ind]
        MP = MPs[ind]

        for i,j,k in product(range(nx), range(ny), range(nz)):

            Q = u[i,j,k]
            MAX = max(MAX, max_abs_eigs(Q, 0, MP) / dx)
            if ndim > 1:
                MAX = max(MAX, max_abs_eigs(Q, 1, MP) / dy)
                if ndim > 2:
                    MAX = max(MAX, max_abs_eigs(Q, 2, MP) / dz)

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt
