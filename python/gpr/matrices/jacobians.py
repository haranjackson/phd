from numpy import dot, eye, outer, tensordot, zeros

from auxiliary.funcs import L2_1D
from gpr.variables.eos import E_1, E_A, total_energy
from gpr.variables.state import heat_flux, sigma, sigma_A


class jacobian_variables():

    def __init__(self, prims, PAR):
        ρ = prims.ρ; p = prims.p; A = prims.A; J = prims.J; v = prims.v; E = prims.E
        γ = PAR.γ; pINF = PAR.pINF; cs2 = PAR.cs2; α2 = PAR.α2

        q = heat_flux(prims.T, J, α2)
        σ = sigma(ρ, A, cs2)
        dσdA = sigma_A(ρ, A, cs2)
        ψ = E_A(A, cs2)

        self.Γ = γ - 1
        self.Ψ = ρ * outer(v, v) - σ
        self.Ω = (E - E_1(ρ, p, γ, pINF)) * v - (dot(σ, v) + q) / ρ
        self.Υ = self.Γ * (L2_1D(v) + α2 * L2_1D(J) + E_1(ρ, p, γ, pINF) - E)
        self.Φ = ρ * outer(v, ψ).reshape([3,3,3])
        self.Φ -= tensordot(v, dσdA, axes=(0,0))

def dQdP(P, PAR, SYS):
    """ Returns the Jacobian of the conserved variables with respect to the primitive variables
    """
    ρ = P.ρ; p = P.p; A = P.A; J = P.J; v = P.v; λ = P.λ; E = P.E
    ψ = E_A(A)
    ret = eye(18)

    ret[1, 0] = E - E_1(ρ, p, PAR.y, PAR.pINF)
    ret[1, 1] /= PAR.γ - 1
    ret[1, 2:5] = ρ * v
    ret[2:5, 0] = v
    ret[2:5, 2:5] *= ρ

    if SYS.viscous:
        ret[1, 5:14] = ρ * ψ.ravel()

    if SYS.thermal:
        ret[1, 14:17] = PAR.α2 * ρ * J
        ret[14:17, 0] = J
        ret[14:17, 14:17] *= ρ

    if SYS.reactive:
        ret[1, 17] = PAR.Qc * ρ
        ret[17, 0] = λ
        ret[17, 17] *= ρ

    return ret

def dQdPdot(P, x, PAR, SYS):
    """ Returns DQ/DP.x where DQ/DP is evaluated at P
    """
    ret = zeros(18)

    ρ = P[0]
    p = P[1]
    v = P[2:5]
    A = P[5:14].reshape([3,3])
    J = P[14:17]
    λ = 0

    x0 = x[0]
    E = total_energy(ρ, p, v, A, J, λ, PAR, SYS)
    E1 = E_1(ρ, p, PAR.γ, PAR.pINF)
    ψ = E_A(A, PAR.cs2)

    ret[0] = x0
    ret[1] = x0*(E-E1) + x[1]/(PAR.γ-1)
    ret[1] += ρ * (dot(v,x[2:5]) + dot(ψ.ravel(),x[5:14]) + PAR.α2*dot(J,x[14:17]))
    ret[2:5] = x0 * v + ρ * x[2:5]
    ret[5:14] = x[5:14]
    ret[14:17] = x0 * J + ρ * x[14:17]

    return ret

def dPdQ(P, jacVars, PAR, SYS):
    """ Returns the Jacobian of the primitive variables with respect to the conserved variables
    """
    ρ = P.ρ; J = P.J; v = P.v; λ = P.λ
    ρ_1 = 1 / ρ
    ψ = E_A(P.A, PAR.cs2)
    ret = eye(18)
    Γ, Υ = jacVars.Γ, jacVars.Υ

    ret[1, 0] = Υ
    ret[1, 1] = Γ
    ret[1, 2:5] = -Γ * v
    ret[2:5, 0] = -v / ρ
    for i in range(2,5):
        ret[i, i] = ρ_1

    if SYS.viscous:
        ret[1, 5:14] = -Γ * ρ * ψ.ravel()

    if SYS.thermal:
        ret[1, 14:17] = -Γ * PAR.α2 * J
        ret[14:17, 0] = -J / ρ
        for i in range(14,17):
            ret[i, i] = ρ_1

    if SYS.reactive:
        ret[17, 0] = -λ / ρ
        ret[17, 17] /= ρ
        Qc = PAR.Qc
        ret[1, 0] += Γ * Qc * λ
        ret[1, 17] -= Γ * Qc

    return ret

def dFdP(P, d, jacVars, PAR, SYS):
    """ Returns the Jacobian of the flux vector with respect to the primitive variables
        NOTE: Primitive variables are assumed to be in standard ordering
    """
    ρ = P.ρ; p = P.p; A = P.A; J = P.J; v = P.v; λ = P.λ; E = P.E; T = P.T
    γ = PAR.γ; pINF = PAR.pINF; α2 = PAR.α2
    ρvd = ρ * v[d]

    q = heat_flux(T, J, α2)
    dσdA = sigma_A(ρ, A, PAR.cs2)
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

    if SYS.viscous:
        ret[1, 5:14] = Φ[d].ravel()
        ret[2:5, 5:14] = -dσdA[d].reshape([3,9])
        ret[5+d, 2:5] = A[0]
        ret[8+d, 2:5] = A[1]
        ret[11+d, 2:5] = A[2]
        ret[5+d, 5:8] = v
        ret[8+d, 8:11] = v
        ret[11+d, 11:14] = v


    if SYS.thermal:
        ret[1, 14:17] = α2 * ρvd * J
        ret[1, 14+d] += α2 * T
        ret[14:17, 0] = v[d] * J
        ret[14+d, 0] -= T / ρ
        ret[14+d, 1] = T / (p+pINF)
        ret[14:17, 2+d] = ρ * J
        for i in range(14,17):
            ret[i, i] = ρvd

    if SYS.reactive:
        ret[17, 0] = v[d] * λ
        ret[17, 2+d] = ρ * λ
        ret[17, 17] = ρvd
        ret[1, 17] += PAR.Qc * ρvd

    return ret
