from numba import jit
from numpy import complex128, diag, dot, imag, zeros
from scipy.linalg import eig, solve

from ader.basis import quad, end_values, derivative_values
from gpr.eig import max_abs_eigs
from gpr.matrices.conserved import flux, Bdot, system_conserved
from options import DEBUG, N1

nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()


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

def Aint(qL, qR, d, params, subsystems):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(18, dtype=complex128)
    qJump = qR - qL
    for i in range(N1):
        q = qL + nodes[i] * qJump
        J = system_conserved(q, d, params, subsystems)
        eigs, R = eig(J, overwrite_a=1, check_finite=0)
        if DEBUG:
            if (abs(imag(R)) > 1e-15).any():
                print("////WARNING//// COMPLEX VALUES IN JACOBIAN")
        L = diag(abs(eigs))
        b = solve(R, qJump, overwrite_b=1, check_finite=0)
        ret += weights[i] * dot(R, dot(L, b))
    return ret.real

def s_max(qL, qR, d, params, subsystems):
    max1 = max_abs_eigs(qL, d, params, subsystems)
    max2 = max_abs_eigs(qR, d, params, subsystems)
    return max(max1, max2) * (qR - qL)

def Drus(qL, qR, d, pos, params, subsystems):
    """ Returns the Rusanov jump term at the dth boundary
    """
    if pos:
        ret = flux(qR, d, params, subsystems)
        ret += flux(qL, d, params, subsystems)
        ret += Bint(qL, qR, d, subsystems.viscous)
    else:
        ret = -flux(qR, d, params, subsystems)
        ret -= flux(qL, d, params, subsystems)
        ret -= Bint(qL, qR, d, subsystems.viscous)
    ret -= s_max(qL, qR, d, params, subsystems)
    return ret

def Dos(qL, qR, d, pos, params, subsystems):
    """ Returns the Osher-Solomon jump term at the dth boundary
    """
    if pos:
        ret = flux(qR, d, params, subsystems)
        ret += flux(qL, d, params, subsystems)
        ret += Bint(qL, qR, d, subsystems.viscous)
    else:
        ret = -flux(qR, d, params, subsystems)
        ret -= flux(qL, d, params, subsystems)
        ret -= Bint(qL, qR, d, subsystems.viscous)
    ret -= Aint(qL, qR, d, params, subsystems)
    return ret
