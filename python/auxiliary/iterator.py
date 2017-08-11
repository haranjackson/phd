from itertools import product

from gpr.eig import max_abs_eigs
from gpr.variables.vectors import Cvec_to_Pvec
from options import CFL, dx, dy, dz, ndim


def timestep(fluids, count, t, tf, PARs):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    m = len(fluids)
    MAX = 0
    for ind in range(m):
        u = fluids[ind]
        PAR = PARs[ind]
        nx, ny, nz = u.shape[:3]
        for i,j,k in product(range(nx), range(ny), range(nz)):
            P = Cvec_to_Pvec(u[i,j,k], PAR)
            MAX = max(MAX, max_abs_eigs(P, 0, PAR) / dx)
            if ndim > 1:
                MAX = max(MAX, max_abs_eigs(P, 1, PAR) / dy)
                if ndim > 2:
                    MAX = max(MAX, max_abs_eigs(P, 2, PAR) / dz)

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt
