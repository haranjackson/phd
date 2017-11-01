from itertools import product

from system.eigenvalues import max_abs_eigs
from options import nx, ny, nz, CFL, dx, dy, dz, ndim


def timestep(fluids, count, t, tf, PARs):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(fluids)
    MAX = 0
    for ind in range(m):

        u = fluids[ind]
        PAR = PARs[ind]

        for i,j,k in product(range(nx), range(ny), range(nz)):

            Q = u[i,j,k]
            MAX = max(MAX, max_abs_eigs(Q, 0, PAR) / dx)
            if ndim > 1:
                MAX = max(MAX, max_abs_eigs(Q, 1, PAR) / dy)
                if ndim > 2:
                    MAX = max(MAX, max_abs_eigs(Q, 2, PAR) / dz)

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt
