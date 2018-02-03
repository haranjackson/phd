from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import Bdot, system, max_eig
from options import N, nV


RUSANOV = 0
ROE = 1
OSHER = 2


def Bint(qL, qR, d, MP):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(nV)
    Δq = qR - qL
    for i in range(N):
        q = qL + NODES[i] * Δq
        tmp = zeros(nV)
        Bdot(tmp, Δq, q, d, MP)
        ret += WGHTS[i] * tmp
    return ret


def D_OSH(qL, qR, d, MP):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(nV, dtype=complex128)
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M = system(q, d, MP)
        λ, R = eig(M, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += WGHTS[i] * dot(R, abs(λ) * b)

    return ret.real


def D_ROE(qL, qR, d, MP):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([nV, nV])
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M += WGHTS[i] * system(q, d, MP)

    λ, R = eig(M, overwrite_a=1, check_finite=0)
    b = solve(R, Δq, check_finite=0)
    return dot(R, abs(λ) * b).real


def D_RUS(qL, qR, d, MP):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = max_eig(qL, d, MP)
    max2 = max_eig(qR, d, MP)
    return max(max1, max2) * (qR - qL)
