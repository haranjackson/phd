from numba import jit
from numpy import complex128, diag, dot, imag, zeros
from scipy.linalg import eig, solve

from ader.basis import quad, end_values, derivative_values
from gpr.eig import max_abs_eigs
from gpr.matrices.conserved import flux_ref, Bdot, system_conserved
from gpr.variables.vectors import Pvec_to_Cvec, Cvec_to_Pvec
from options import DEBUG, N1, reconstructPrim

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

def Aint(qL, qR, d, PAR, SYS):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(18, dtype=complex128)
    qJump = qR - qL
    for i in range(N1):
        q = qL + nodes[i] * qJump
        J = system_conserved(q, d, PAR, SYS)
        eigs, R = eig(J, overwrite_a=1, check_finite=0)
        if DEBUG:
            if (abs(imag(R)) > 1e-15).any():
                print("////WARNING//// COMPLEX VALUES IN JACOBIAN")
        L = diag(abs(eigs))
        b = solve(R, qJump, overwrite_b=1, check_finite=0)
        ret += weights[i] * dot(R, dot(L, b))
    return ret.real

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

def Drus(xL, xR, d, pos, PAR, SYS):
    """ Returns the Rusanov jump term at the dth boundary
    """
    pL, pR, qL, qR = input_vectors(xL, xR, PAR, SYS)

    max1 = max_abs_eigs(pL, d, PAR, SYS)
    max2 = max_abs_eigs(pR, d, PAR, SYS)
    if pos:
        ret = - max(max1, max2) * (qR - qL)
    else:
        ret = max(max1, max2) * (qR - qL)

    flux_ref(ret, pR, d, PAR, SYS)
    flux_ref(ret, pL, d, PAR, SYS)
    ret += Bint(qL, qR, d, SYS.viscous)

    if pos:
        return ret
    else:
        return -ret

def Dos(xL, xR, d, pos, PAR, SYS):
    """ Returns the Osher-Solomon jump term at the dth boundary
    """
    pL, pR, qL, qR = input_vectors(xL, xR, PAR, SYS)

    if pos:
        ret = - Aint(qL, qR, d, PAR, SYS)
    else:
        ret = Aint(qL, qR, d, PAR, SYS)

    flux_ref(ret, pR, d, PAR, SYS)
    flux_ref(ret, pL, d, PAR, SYS)
    ret += Bint(qL, qR, d, SYS.viscous)

    if pos:
        return ret
    else:
        return -ret
