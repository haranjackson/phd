from functools import product

from numpy import array, dot, zeros

from solvers.basis import PSI
from solvers.weno.weno import weno
from models.gpr.misc.structures import Cvec_to_Pvec


# The values of the basis polynomials at x=0.5
MIDVALS = array([ψ(0.5) for ψ in PSI])


def weno_primitive(q, MP):
    """ Returns a WENO reconstruction in primitive variables, given the grid of
        conserved values. A reconstruction in conserved variables is performed.
        The midpoints of this reconstruction are then taken as the conserved
        cell averages (required for >2nd order). A primitive reconstruction is
        then performed with these averages.
    """
    qr = weno(q)
    qav = dot(MIDVALS, qr)
    pav = zeros(qav.shape)
    nx, ny, nz = qav.shape[:3]
    for i, j, k in product(range(nx), range(ny), range(nz)):
        pav[i, j, k] = Cvec_to_Pvec(qav[i, j, k], MP)
    return weno(pav)


def expand_weno(wh):
    nx, ny, _, N, _, nvar = wh.shape
    ret = zeros([nx * N, ny * N, nvar])
    for i in range(nx):
        for j in range(ny):
            for ii in range(N):
                for jj in range(N):
                    indi = i * N + ii
                    indj = j * N + jj
                    ret[indi, indj] = wh[i, j, 0, ii, jj]
    return ret
