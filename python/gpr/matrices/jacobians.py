from numpy import dot, eye, outer, tensordot, zeros

from auxiliary.funcs import L2_1D
from gpr.functions import primitive
from gpr.matrices.conserved import block
from gpr.variables import E_1, E_A, heat_flux, sigma, sigma_A
from options import reactiveEOS


class jacobian_variables():

    def __init__(self, prims, params):
        ρ = prims.ρ; p = prims.p; A = prims.A; J = prims.J; v = prims.v; E = prims.E
        γ = params.γ; pINF = params.pINF; cs2 = params.cs2; α2 = params.α2

        q = heat_flux(prims.T, J, α2)
        σ = sigma(ρ, A, cs2)
        dσdA = sigma_A(ρ, A, cs2)
        ψ = E_A(A, cs2)

        self.Γ = params.γ - 1
        self.Ψ = ρ * outer(v, v) - σ
        self.Ω = (E - E_1(ρ, p, γ, pINF)) * v - (dot(σ, v) + q) / ρ
        self.Υ = self.Γ * (L2_1D(v) + α2 * L2_1D(J) + E_1(ρ, p, γ, pINF) - E)
        self.Φ = ρ * outer(v, ψ).reshape([3,3,3])
        self.Φ -= tensordot(v, dσdA, axes=(0,0))

def dQdP(P, params, jacVars, subsystems):
    """ Returns the Jacobian of the conserved variables with respect to the primitive variables
    """
    ρ = P.ρ; p = P.p; A = P.A; J = P.J; v = P.v; λ = P.λ; E = P.E
    ψ = E_A(A)
    ret = eye(18)
    Γ = jacVars.Γ

    ret[1, 0] = E - E_1(ρ, p, params.y, params.pINF)
    ret[1, 1] /= Γ
    ret[1, 2:5] = ρ * v
    ret[2:5, 0] = v
    ret[2:5, 2:5] *= ρ

    if subsystems.viscous:
        ret[1, 5:14] = ρ * ψ.ravel(order='F')

    if subsystems.thermal:
        ret[1, 14:17] = params.α2 * ρ * J
        ret[14:17, 0] = J
        ret[14:17, 14:17] *= ρ

    if subsystems.reactive:
        ret[1, 17] = params.Qc * ρ
        ret[17, 0] = λ
        ret[17, 17] *= ρ

    return ret

def dPdQ(P, params, jacVars, subsystems):
    """ Returns the Jacobian of the primitive variables with respect to the conserved variables
    """
    ρ = P.ρ; A = P.A; J = P.J; v = P.v; λ = P.λ
    ρ_1 = 1 / ρ
    ψ = E_A(A, params.cs2)
    ret = eye(18)
    Γ, Υ = jacVars.Γ, jacVars.Υ

    ret[1, 0] = Υ
    ret[1, 1] = Γ
    ret[1, 2:5] = -Γ * v
    ret[2:5, 0] = -v / ρ
    for i in range(2,5):
        ret[i, i] = ρ_1

    if subsystems.viscous:
        ret[1, 5:14] = -Γ * ρ * ψ.ravel(order='F')

    if subsystems.thermal:
        ret[1, 14:17] = -Γ * params.α2 * J
        ret[14:17, 0] = -J / ρ
        for i in range(14,17):
            ret[i, i] = ρ_1

    if subsystems.reactive:
        ret[17, 0] = -λ / ρ
        ret[17, 17] /= ρ

        if reactiveEOS:
            ret[1, 0] += Γ * params.Qc * λ
            ret[1, 17] -= Γ * params.Qc

    return ret

def dFdP(P, d, params, jacVars, subsystems):
    """ Returns the Jacobian of the flux vector with respect to the primitive variables
    """
    ρ = P.ρ; p = P.p; A = P.A; J = P.J; v = P.v; λ = P.λ; E = P.E; T = P.T
    γ = params.γ; pINF = params.pINF; cs2 = params.cs2; α2 = params.α2
    ρvd = ρ * v[d]

    q = heat_flux(T, J, α2)
    dσdA = sigma_A(ρ, A, cs2)
    Ψ, Φ, Ω, Γ = jacVars.Ψ, jacVars.Φ, jacVars.Ω, jacVars.Γ

    ret = zeros([18, 18])
    ret[0, 0] = v[d]
    ret[0, 2+d] = ρ
    ret[1, 0] = Ω[d]
    ret[1, 1] = γ * v[d] / Γ + q[d] / (p + pINF)
    ret[1, 2:5] = Ψ[d]
    ret[1, 2+d] += ρ * E + p
    ret[2:5, 0] = Ψ[d] / ρ
    for i in range(2,5):
        ret[i, i] = ρvd
    ret[2:5, 2+d] += ρ * v
    ret[2+d, 1] = 1

    if subsystems.viscous:
        ret[1, 5:14] = Φ[d].ravel(order='F')
        ret[2:5, 5:14] = -dσdA[d].reshape([3,9], order='F')
        k1 = 5+3*d
        ret[k1:k1+3, 2:5] = A
        for i in range(3):
            vi = v[i]
            k2 = 5+3*i
            for j in range(3):
                ret[k1+j, k2+j] = vi

    if subsystems.thermal:
        ret[1, 14:17] = α2 * ρvd * J
        ret[1, 14+d] += α2 * T
        ret[14:17, 0] = v[d] * J
        ret[14+d, 0] -= T / ρ
        ret[14+d, 1] = T / (p+pINF)
        ret[14:17, 2+d] = ρ * J
        for i in range(14,17):
            ret[i, i] = ρvd

    if subsystems.reactive:
        ret[17, 0] = v[d] * λ
        ret[17, 2+d] = ρ * λ
        ret[17, 17] = ρvd

        if reactiveEOS:
            ret[1, 17] += params.Qc * ρvd

    return ret

def jacobian(Q, d, params, subsystems):
    """ Returns the Jacobian in the dth direction
    """
    P = primitive(Q, params, subsystems)
    jacVars = jacobian_variables(P, params)
    DFDP = dFdP(P, d, params, jacVars, subsystems)
    DPDQ = dPdQ(P, params, jacVars, subsystems)
    return dot(DFDP, DPDQ) + block(P.v, d, subsystems.viscous)
