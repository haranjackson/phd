from itertools import product

from numpy import sum, zeros

from solver.gfm.fill import fill_ghost_cells


def get_material_index(Q, m):
    NV = len(Q)
    LSET = m - 1
    return sum(Q[NV-LSET:] >= 0)


def make_u(mats):
    """ Builds u across the domain, from the different material grids
    """
    u = zeros(mats[0].shape)
    NDIM = len(u.shape) - 1
    av = sum(mats, axis=0) / len(mats)

    for coords in product(*[range(s) for s in av.shape[:NDIM]]):

        materialIndex = get_material_index(av[coords], len(mats))
        u[coords] = mats[materialIndex][coords]

    return u
