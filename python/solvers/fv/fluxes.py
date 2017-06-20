from numba import jit
from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import quad
from gpr.eig import max_abs_eigs, perron_frobenius
from gpr.matrices.conserved import Bdot, system_conserved
from gpr.variables.vectors import Pvec_to_Cvec, Cvec_to_Pvec
from options import N1, reconstructPrim, perronFrob


nodes, _, weights = quad()


@jit
def Bint(qL, qR, d, viscous):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(18)
    v = zeros(3)
    qJump = qR - qL
    for i in range(N1):
        q = qL + nodes[i] * qJump
        v += weights[i] * q[2:5] / q[0]
    Bdot(ret, qJump, v, d)
    return ret

def Aint(pL, pR, qL, qR, d, PAR, SYS):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(18, dtype=complex128)
    Δq = qR - qL
    for i in range(N1):
        q = qL + nodes[i] * Δq
        J = system_conserved(q, d, PAR, SYS)
        λ, R = eig(J, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += weights[i] * dot(R, abs(λ)*b)
    return ret.real

def Smax(pL, pR, qL, qR, d, PAR, SYS):

    if perronFrob:
        max1 = perron_frobenius(pL, d, PAR, SYS)
        max2 = perron_frobenius(pR, d, PAR, SYS)
    else:
        max1 = max_abs_eigs(pL, d, PAR, SYS)
        max2 = max_abs_eigs(pR, d, PAR, SYS)

    return max(max1, max2) * (qR - qL)

def input_vectors(xL, xR, PAR, SYS):

    if reconstructPrim:
        pL = xL
        pR = xR
        qL = Pvec_to_Cvec(pL, PAR, SYS)
        qR = Pvec_to_Cvec(pR, PAR, SYS)
    else:
        qL = xL
        qR = xR
        pL = Cvec_to_Pvec(qL, PAR, SYS)
        pR = Cvec_to_Pvec(qR, PAR, SYS)

    return pL, pR, qL, qR
