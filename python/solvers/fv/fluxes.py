from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import Bdot, system, max_eig
from options import N, NV


RUSANOV = 0
ROE = 1
OSHER = 2


def B_INT(qL, qR, d, MP):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(NV)
    Δq = qR - qL
    for i in range(N):
        q = qL + NODES[i] * Δq
        tmp = zeros(NV)
        Bdot(tmp, Δq, q, d, MP)
        ret += WGHTS[i] * tmp
    return ret


def D_OSH(qL, qR, d, MP):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(NV, dtype=complex128)
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M = system(q, d, MP)
        λ, R = eig(M, overwrite_a=1)
        b = solve(R, Δq)
        ret += WGHTS[i] * dot(R, abs(λ) * b)

    return ret.real


def D_ROE(qL, qR, d, MP):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([NV, NV])
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M += WGHTS[i] * system(q, d, MP)

    λ, R = eig(M, overwrite_a=1)
    b = solve(R, Δq)
    return dot(R, abs(λ) * b).real


def D_RUS(qL, qR, d, MP):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = max_eig(qL, d, MP)
    max2 = max_eig(qR, d, MP)
    return max(max1, max2) * (qR - qL)
