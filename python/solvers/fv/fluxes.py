from numba import jit
from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import quad
from system.eigenvalues import max_abs_eigs
from system.system import Bdot, system
from options import N1, nV


NODES, _, WGHTS = quad()


@jit
def Bint(qL, qR, d, PAR):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(nV)
    qJump = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * qJump
        tmp  = zeros(nV)
        Bdot(tmp, qJump, q, d, PAR)
        ret += WGHTS[i] * tmp
    return ret

def Aint(qL, qR, d, PAR):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(nV, dtype=complex128)
    Δq = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * Δq
        J = system(q, d, PAR)
        λ, R = eig(J, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += WGHTS[i] * dot(R, abs(λ)*b)
    return ret.real

def Smax(qL, qR, d, PAR):
    max1 = max_abs_eigs(qL, d, PAR)
    max2 = max_abs_eigs(qR, d, PAR)
    return max(max1, max2) * (qR - qL)
