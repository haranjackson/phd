from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import nonconservative_matrix, system_matrix, max_eig
from options import N, NV


RUSANOV = 0
ROE = 1
OSHER = 2


def B_INT(qL, qR, d, *args):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(NV)
    Δq = qR - qL
    for i in range(N):
        q = qL + NODES[i] * Δq
        B = nonconservative_matrix(q, d, *args)
        ret += WGHTS[i] * dot(B, Δq)
    return ret


def D_OSH(qL, qR, d, *args):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(NV, dtype=complex128)
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M = system_matrix(q, d, *args)
        λ, R = eig(M, overwrite_a=1)
        b = solve(R, Δq)
        ret += WGHTS[i] * dot(R, abs(λ) * b)

    return ret.real


def D_ROE(qL, qR, d, *args):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([NV, NV])
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M += WGHTS[i] * system_matrix(q, d, *args)

    λ, R = eig(M, overwrite_a=1)
    b = solve(R, Δq)
    return dot(R, abs(λ) * b).real


def D_RUS(qL, qR, d, *args):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = max_eig(qL, d, *args)
    max2 = max_eig(qR, d, *args)
    return max(max1, max2) * (qR - qL)
