from numpy import dot, eye, zeros

from auxiliary.funcs import L2_1D, L2_2D
from gpr.variables.eos import E_A, E_J
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.state import sigma, sigma_A, temperature
from gpr.variables.vectors import primitive, Cvec_to_Pvec


def system_primitive(Q, d, PAR, SYS):
    """ Returns the system matrix in the dth direction for the system of primitive variables,
        calculated directly
    """
    P = primitive(Q, PAR, SYS)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    y = PAR.y; pINF = PAR.pINF
    sig = sigma(ρ, A)
    dσdA = sigma_A(ρ, A)

    ret = v[d] * eye(18)
    ret[0, 2+d] = ρ
    ret[1, 2+d] = y * p
    ret[1, 14+d] = (y-1) * PAR.α2 * T
    ret[2+d, 1] = 1 / ρ

    ret[2:5, 0] = -sig[d] / ρ**2
    ret[2:5, 5:14] = -1 / ρ * dσdA[d].reshape([3,9])

    ret[5+d, 2:5] = A[0]
    ret[8+d, 2:5] = A[1]
    ret[11+d, 2:5] = A[2]

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not SYS.reactive:
        ret[17, 17] = 0

    return ret

def system_primitive_reordered(Q, d, PAR, SYS):
    """ Returns the system matrix in the dth direction for the system of primitive variables,
        calculated directly.
        NOTE: Currently in column-major form.
    """
    P = primitive(Q, PAR, SYS)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    γ = PAR.γ; pINF = PAR.pINF
    sig = sigma(ρ, A)
    dsdA = sigma_A(ρ, A)

    ret = v[d] * eye(18)
    ret[0, 11+d] = ρ
    ret[1, 11+d] = γ * p
    ret[1, 14+d] = (γ-1) * PAR.α2 * T
    ret[2+3*d:5+3*d, 11:14] = A
    ret[11:14, 0] = -sig[d] / ρ**2
    ret[11+d, 1] = 1 / ρ

    for i in range(3):
        ret[11+i, 2:11] = -1 / ρ * dsdA[i,d].ravel(order='F')

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not SYS.reactive:
        ret[17, 17] = 0

    return ret


def source_primitive_ref(ret, P, PAR, SYS):

    ρ = P[0]
    γ = PAR.γ

    if SYS.viscous:
        A = P[5:14].reshape([3,3])
        ψ = E_A(A, PAR.cs2)
        θ1 = theta_1(A, PAR.cs2, PAR.τ1)

        ret[1] = (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if SYS.thermal:
        J = P[14:17]
        T = temperature(ρ, P.p, γ, PAR.pINF, PAR.cv)
        H = E_J(J, PAR.α2)
        θ2 = theta_2(ρ, T, PAR.ρ0, PAR.T0, PAR.α2, PAR.τ2)

        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

def source_primitive(Q, PAR, SYS):

    ret = zeros(18)
    P = Cvec_to_Pvec(Q, PAR, SYS)
    source_primitive_ref(ret, P, PAR, SYS)
    return ret

def source_primitive_reordered(Q, PAR, SYS):

    ret = zeros(18)
    P = primitive(Q, PAR)
    ρ = P.ρ; A = P.A
    γ = PAR.γ

    if SYS.viscous:
        ψ = E_A(A)
        θ1 = theta_1(A, PAR.cs2, PAR.τ1)
        ret[1] += (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[2:11] = -ψ.ravel() / θ1

    if SYS.thermal:
        H = E_J(P.J)
        θ2 = theta_2(ρ, P.T, PAR.ρ0, PAR.T0, PAR.α2, PAR.τ2)
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

    return ret


def Mdot_ref(ret, P, x, d, PAR, SYS):
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

    if SYS.viscous:
        A = P[5:14].reshape([3,3])
        cs2 = PAR.cs2
        σ = sigma(ρ, A, cs2)
        dσdAd = sigma_A(ρ, A, cs2)[d].reshape([3,9])

        ret[2:5] -= x[0] * σ[d] / ρ**2 + dot(dσdAd, x[5:14]) / ρ
        xv = x[2:5]
        ret[5+d] += dot(A[0],xv)
        ret[8+d] += dot(A[1],xv)
        ret[11+d] += dot(A[2],xv)

    if SYS.thermal:
        T = temperature(ρ, p, γ, pINF, PAR.cv)
        ret[1] += (γ-1) * PAR.α2 * T * x[14+d]
        ret[14+d] += T / ρ * (x[1]/(p+pINF) - x[0]/ρ)
