from numba import jit
from numpy import complex128, diag, dot, imag, zeros
from scipy.linalg import eig, solve

from ader.basis import quad, end_values, derivative_values
from experimental.ader_prim.matrices_prim import flux_ref
from gpr.eig import max_abs_eigs_prim
from gpr.matrices.conserved import block, system_conserved
from options import DEBUG, N1

nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()


@jit
def Bint(pL, pR, d, viscous):
    """ Returns the jump matrix for B, in the dth direction
    """
    v = zeros(3)
    for i in range(N1):
        p = pL + nodes[i] * (pR - pL)
        v += weights[i] * p[2:5] / p[0]
    return dot(block(v, d, viscous), pR - pL)

def Aint(pL, pR, d, params, subsystems):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(18, dtype=complex128)
    for i in range(N1):
        p = pL + nodes[i] * (pR - pL)
        J = system_conserved(p, d, params, subsystems)
        eigs, R = eig(J, overwrite_a=1, check_finite=0
        L = diag(abs(eigs))
        b = solve(R, pR-pL, overwrite_b=1, check_finite=0)
        ret += weights[i] * dot(R, dot(L, b))
    return ret.real

def Drus(pL, pR, d, pos, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive):
    """ Returns the Rusanov jump term at the dth boundary
    """
    max1 = max_abs_eigs_prim(pL, d, γ, pINF, cv, cs2, α2, viscous, thermal)
    max2 = max_abs_eigs_prim(pR, d, γ, pINF, cv, cs2, α2, viscous, thermal)
    if pos:
        ret = - max(max1, max2) * (pR - pL)
    else:
        ret = max(max1, max2) * (pR - pL)

    flux_ref(ret, pR, d, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive)
    flux_ref(ret, pL, d, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive)
    ret += Bint(pL, pR, d, viscous)

    if pos:
        return ret
    else:
        return -ret

def Dos(pL, pR, d, pos, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive):
    """ Returns the Osher-Solomon jump term at the dth boundary
    """
    if pos:
        ret = - Aint(pL, pR, d, params, subsystems)
    else:
        ret = Aint(pL, pR, d, params, subsystems)

    flux_ref(ret, pR, d, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive)
    flux_ref(ret, pL, d, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive)
    ret += Bint(pL, pR, d, viscous)

    if pos:
        return ret
    else:
        return -ret
