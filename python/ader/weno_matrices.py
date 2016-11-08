from numpy import array, ceil, floor, polyint, sqrt, zeros

from ader.basis import basis_polys
from options import N, N1


_, ψDer, ψInt = basis_polys()


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
            ψpInt = ψInt[p]
            if nStencils==3:
                Mc[0,e,p] = ψpInt(e-floorHalfN+1) - ψpInt(e-floorHalfN)
                Mc[1,e,p] = ψpInt(e-N1+2) - ψpInt(e-N1+1)
                Mc[2,e,p] = ψpInt(e+1) - ψpInt(e)
            else:
                Mc[0,e,p] = ψpInt(e-floorHalfN+1) - ψpInt(e-floorHalfN)
                Mc[1,e,p] = ψpInt(e-ceilHalfN+1) - ψpInt(e-ceilHalfN)
                Mc[2,e,p] = ψpInt(e-N1+2) - ψpInt(e-N1+1)
                Mc[3,e,p] = ψpInt(e+1) - ψpInt(e)
    return Mc

def inv_coeff_mats_1():
    r3 = sqrt(3)
    ret1 = 1/r3 * array([[r3+1/2, -1/2],[r3-1/2, 1/2]])
    ret2 = 1/r3 * array([[1/2, r3-1/2],[-1/2, r3+1/2]])
    return ret1, ret2

def oscillation_indicator():
    """ Generate the oscillation indicator matrix
    """
    Σ = zeros([N1, N1])
    for p in range(N1):
        for m in range(N1):
            for a in range(1,N1):
                ψDera = ψDer[a]
                antiderivative = polyint(ψDera[p] * ψDera[m])
                Σ[p,m] += antiderivative(1) - antiderivative(0)
    return Σ
