from itertools import product

from numpy import sum, zeros

from multi.gfm import get_material_index
from options import CFL, NDIM
from system import max_eig


def make_u(mats):
    """ Builds u across the domain, from the different material grids
    """
    u = zeros(mats[0].shape)
    av = sum(mats, axis=0) / len(mats)

    for coords in product(*[range(s) for s in av.shape[:NDIM]]):

        materialIndex = get_material_index(av[coords])
        u[coords] = mats[materialIndex][coords]

    return u


def timestep(u, count, t, tf, dX, MPs):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    MAX = 0
    for coords in product(*[range(s) for s in u.shape[:NDIM]]):

        Q = u[coords]
        materialIndex = get_material_index(Q)
        MP = MPs[materialIndex]

        for d in range(NDIM):
            MAX = max(MAX, max_eig(Q, d, MP) / dX[d])

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    return min(tf - t, dt)
