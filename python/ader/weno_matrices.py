from numpy import polyint, zeros, floor, ceil

from ader.basis import basis_polys
from options import N, N1


_, psiDer, psiInt = basis_polys()


if N%2:
    nStencils = 4
else:
    nStencils = 3


def coefficient_matrices():
    """ Generate linear systems governing the coefficients of the basis polynomials
    """
    floorHalfN = floor(N/2)
    ceilHalfN = ceil(N/2)
    Mc = zeros([nStencils, N1, N1])
    for e in range(N1):
        for p in range(N1):
            psipInt = psiInt[p]
            if nStencils==3:
                Mc[0,e,p] = psipInt(e-floorHalfN+1) - psipInt(e-floorHalfN)
                Mc[1,e,p] = psipInt(e-N1+2) - psipInt(e-N1+1)
                Mc[2,e,p] = psipInt(e+1) - psipInt(e)
            else:
                Mc[0,e,p] = psipInt(e-floorHalfN+1) - psipInt(e-floorHalfN)
                Mc[1,e,p] = psipInt(e-ceilHalfN+1) - psipInt(e-ceilHalfN)
                Mc[2,e,p] = psipInt(e-N1+2) - psipInt(e-N1+1)
                Mc[3,e,p] = psipInt(e+1) - psipInt(e)
    return Mc

def oscillation_indicator():
    """ Generate the oscillation indicator matrix
    """
    SIGMA = zeros([N1, N1])
    for p in range(N1):
        for m in range(N1):
            for a in range(1,N1):
                psiDera = psiDer[a]
                antiderivative = polyint(psiDera[p] * psiDera[m])
                SIGMA[p,m] += antiderivative(1) - antiderivative(0)
    return SIGMA