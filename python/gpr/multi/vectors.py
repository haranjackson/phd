from numpy import zeros

from gpr.vars.eos import total_energy
from gpr.opts import THERMAL


def Pvec(P):
    """ Vector of primitive variables
        NOTE: Uses atypical ordering
    """
    if THERMAL:
        ret = zeros(17)
        ret[14:17] = P.J
    else:
        ret = zeros(14)

    ret[0] = P.ρ
    ret[1] = P.p()
    ret[2:11] = P.A.ravel(order='F')
    ret[11:14] = P.v

    return ret


def Pvec_to_Cvec(P, MP):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3, 3])

    λ = 0

    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], λ, MP)
    Q[2:5] *= ρ
    Q[14:] *= ρ

    return Q
