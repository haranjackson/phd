from numpy import array, dot, eye, zeros

from auxiliary.funcs import L2_1D, L2_2D
from gpr.variables.eos import E_A, E_J
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.state import sigma, sigma_A, temperature
from gpr.variables.vectors import Cvec_to_Pclass, Cvec_to_Pvec
from options import VISCOUS, THERMAL, REACTIVE


def system_primitive(Q, d, PAR):
    """ Returns the system matrix in the dth direction for the system of
        primitive variables, calculated directly
    """
    P = Cvec_to_Pclass(Q, PAR)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    γ = PAR.γ; pINF = PAR.pINF; cs2 = PAR.cs2

    sig = sigma(ρ, A, cs2)
    dσdA = sigma_A(ρ, A, cs2)

    ret = v[d] * eye(18)
    ret[0, 2+d] = ρ
    ret[1, 2+d] = γ * p
    ret[1, 14+d] = (γ-1) * PAR.α2 * T
    ret[2+d, 1] = 1 / ρ

    ret[2:5, 0] = -sig[d] / ρ**2
    ret[2:5, 5:14] = -1 / ρ * dσdA[d].reshape([3,9])

    ret[5+d, 2:5] = A[0]
    ret[8+d, 2:5] = A[1]
    ret[11+d, 2:5] = A[2]

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not REACTIVE:
        ret[17, 17] = 0

    return ret

def system_primitive_reordered(Q, d, PAR):
    """ Returns the system matrix in the dth direction for the system of
        primitive variables, calculated directly.
        NOTE: Currently in column-major form.
    """
    ret = system_primitive(Q, d, PAR)
    perm = array([0,1,5,8,11,6,9,12,7,10,13,2,3,4,14,15,16,17])
    return ret[:,perm][perm,:]

def source_primitive_ref(ret, P, PAR):

    ρ = P[0]
    γ = PAR.γ

    if VISCOUS:
        A = P[5:14].reshape([3,3])
        ψ = E_A(A, PAR.cs2)
        θ1 = theta_1(A, PAR.cs2, PAR.τ1)

        ret[1] = (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if THERMAL:
        J = P[14:17]
        T = temperature(ρ, P[1], γ, PAR.pINF, PAR.cv)
        H = E_J(J, PAR.α2)
        θ2 = theta_2(ρ, T, PAR.ρ0, PAR.T0, PAR.α2, PAR.τ2)

        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

def source_primitive(P, PAR):

    ret = zeros(18)
    source_primitive_ref(ret, P, PAR)
    return ret

def source_primitive_reordered(Q, PAR):

    P = Cvec_to_Pvec(Q, PAR)
    ret = source_primitive(P, PAR)
    perm = array([0,1,5,8,11,6,9,12,7,10,13,2,3,4,14,15,16,17])
    return ret[perm]

def Mdot_ref(ret, P, x, d, PAR):
    """ Returns M(P).x
    """
    ρ = P[0]
    p = P[1]
    γ = PAR.γ
    pINF = PAR.pINF

    ret += P[2+d] * x            # v[d] * x
    ret[0] += ρ * x[2+d]
    ret[1] += γ * p * x[2+d]
    ret[2+d] += x[1] / ρ

    if VISCOUS:
        A = P[5:14].reshape([3,3])
        cs2 = PAR.cs2
        σ = sigma(ρ, A, cs2)
        dσdAd = sigma_A(ρ, A, cs2)[d].reshape([3,9])

        ret[2:5] -= x[0] * σ[d] / ρ**2 + dot(dσdAd, x[5:14]) / ρ
        xv = x[2:5]
        ret[5+d] += dot(A[0],xv)
        ret[8+d] += dot(A[1],xv)
        ret[11+d] += dot(A[2],xv)

    if THERMAL:
        T = temperature(ρ, p, γ, pINF, PAR.cv)
        ret[1] += (γ-1) * PAR.α2 * T * x[14+d]
        ret[14+d] += T / ρ * (x[1]/(p+pINF) - x[0]/ρ)
