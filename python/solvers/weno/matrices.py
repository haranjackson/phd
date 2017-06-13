from numpy import array, ceil, floor, polyint, sqrt, zeros

from solvers.basis import basis_polys
from options import N, N1


_, ψDer, ψInt = basis_polys()


def coefficient_matrices():
    """ Generate linear systems governing the coefficients of the basis polynomials
    """
    floorHalfN = floor(N/2)
    ceilHalfN = ceil(N/2)
    Mc = zeros([4, N1, N1])
    for e in range(N1):
        for p in range(N1):
            ψpInt = ψInt[p]
            Mc[0,e,p] = ψpInt(e-N1+2) - ψpInt(e-N1+1)
            Mc[1,e,p] = ψpInt(e+1) - ψpInt(e)
            Mc[2,e,p] = ψpInt(e-ceilHalfN+1) - ψpInt(e-ceilHalfN)
            Mc[3,e,p] = ψpInt(e-floorHalfN+1) - ψpInt(e-floorHalfN)
    return Mc

def inv_coeff_mats_1():
    r3 = sqrt(3)
    invML = 1/r3 * array([[1/2, r3-1/2],[-1/2, r3+1/2]])
    invMR = 1/r3 * array([[r3+1/2, -1/2],[r3-1/2, 1/2]])
    return invML, invMR

def inv_coeff_mats_2():
    invML = array([[(2-3*sqrt(15))/60, -1/15+sqrt(3/5), (62-9*sqrt(15))/60],
                  [-1/24, 1/12, 23/24],
                  [(2+3*sqrt(15))/60, -1/15-sqrt(3/5), (62+9*sqrt(15))/60]])
    invMR = array([[(62+9*sqrt(15))/60, -1/15-sqrt(3/5), (2+3*sqrt(15))/60],
                  [23/24, 1/12, -1/24],
                  [(62-9*sqrt(15))/60, -1/15+sqrt(3/5), (2-3*sqrt(15))/60]])
    invMC = array([[(2+3*sqrt(15))/60, 14/15, (2-3*sqrt(15))/60],
                  [-1/24, 13/12, -1/24],
                  [(2-3*sqrt(15))/60, 14/15, (2+3*sqrt(15))/60]])

    return invML, invMR, invMC

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
