from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import Bdot, system, max_eig
from options import N1, nV


def Bint(qL, qR, d, MP):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(nV)
    qJump = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * qJump
        tmp  = zeros(nV)
        Bdot(tmp, qJump, q, d, MP)
        ret += WGHTS[i] * tmp
    return ret

def Aint(qL, qR, d, MP):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(nV, dtype=complex128)
    Δq = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * Δq
        J = system(q, d, MP)
        λ, R = eig(J, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += WGHTS[i] * dot(R, abs(λ)*b)
    return ret.real

def Smax(qL, qR, d, MP):
    max1 = max_eig(qL, d, MP)
    max2 = max_eig(qR, d, MP)
    return max(max1, max2) * (qR - qL)
