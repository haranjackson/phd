from functools import product

from numpy import dot, zeros

from solvers.basis import MIDVALS
from solvers.weno.weno import weno
from gpr.misc.structures import Cvec_to_Pvec


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
        pav[i,j,k] = Cvec_to_Pvec(qav[i,j,k], MP)
    return weno(pav)

def expand_weno(wh):
    nx, ny, _, N1, _, nvar = wh.shape
    ret = zeros([nx*N1, ny*N1, nvar])
    for i in range(nx):
        for j in range(ny):
            for ii in range(N1):
                for jj in range(N1):
                    indi = i*N1+ii
                    indj = j*N1+jj
                    ret[indi,indj] = wh[i,j,0,ii,jj]
    return ret