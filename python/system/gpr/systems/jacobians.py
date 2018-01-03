from numpy import dot, eye, outer, tensordot, zeros

from system.gpr.misc.functions import L2_1D
from options import VISCOUS, THERMAL, REACTIVE, nV


class jacobian_variables():

    def __init__(self, P):
        ρ = P.ρ
        J = P.J
        v = P.v
        E = P.E
        q = P.q
        σ = P.σ

        E1 = P.E1()
        dσdA = P.dσdA()
        ψ = P.ψ()

        MP = P.MP
        γ = MP.γ
        α2 = MP.α2

        self.Γ = γ - 1
        self.Ψ = ρ * outer(v, v) - σ
        self.Ω = (E - E1) * v - (dot(σ, v) + q) / ρ
        self.Υ = self.Γ * (L2_1D(v) + α2 * L2_1D(J) + E1 - E)
        self.Φ = ρ * outer(v, ψ).reshape([3,3,3])
        self.Φ -= tensordot(v, dσdA, axes=(0,0))

def dQdP(P):
    """ Returns the Jacobian of the conserved variables with respect to the primitive variables
    """
    ret = eye(nV)

    ρ = P.ρ
    J = P.J
    v = P.v
    E = P.E

    E1 = P.E1()
    ψ = P.ψ()

    MP = P.MP
    γ = MP.γ
    α2 = MP.α2
    Qc = MP.Qc

    ret[1, 0] = E - E1
    ret[1, 1] /= γ - 1
    ret[1, 2:5] = ρ * v
    ret[2:5, 0] = v
    ret[2:5, 2:5] *= ρ

    if VISCOUS:
        ret[1, 5:14] = ρ * ψ.ravel()

    if THERMAL:
        ret[1, 14:17] = α2 * ρ * J
        ret[14:17, 0] = J
        ret[14:17, 14:17] *= ρ

    if REACTIVE:
        ret[1, 17] = Qc * ρ
        ret[17, 0] = P.λ
        ret[17, 17] *= ρ

    return ret

def dPdQ(P):
    """ Returns the Jacobian of the primitive variables with respect to the conserved variables
    """
    ret = eye(nV)

    ρ = P.ρ
    v = P.v
    J = P.J

    ψ = P.ψ()

    MP = P.MP
    α2 = MP.α2

    jacVars = jacobian_variables(P)
    Γ = jacVars.Γ
    Υ = jacVars.Υ

    ret[1, 0] = Υ
    ret[1, 1] = Γ
    ret[1, 2:5] = -Γ * v
    ret[2:5, 0] = -v / ρ

    for i in range(2,5):
        ret[i, i] = 1 / ρ

    if VISCOUS:
        ret[1, 5:14] = -Γ * ρ * ψ.ravel()

    if THERMAL:
        ret[1, 14:17] = -Γ * α2 * J
        ret[14:17, 0] = -J / ρ
        for i in range(14,17):
            ret[i, i] = 1 / ρ

    if REACTIVE:
        λ = P.λ
        Qc = MP.Qc
        ret[17, 0] = -λ / ρ
        ret[17, 17] /= ρ
        ret[1, 0] += Γ * Qc * λ
        ret[1, 17] -= Γ * Qc

    return ret

def dFdP(P, d):
    """ Returns the Jacobian of the flux vector with respect to the primitive variables
        NOTE: Primitive variables are assumed to be in standard ordering
    """
    ρ = P.ρ
    p = P.p
    v = P.v
    A = P.A
    J = P.J
    E = P.E
    T = P.T
    q = P.q

    dσdA = P.dσdA()

    MP = P.MP
    γ = MP.γ
    pINF = MP.pINF
    α2 = MP.α2

    ρvd = ρ * v[d]

    jacVars = jacobian_variables(P)
    Ψ = jacVars.Ψ
    Φ = jacVars.Φ
    Ω = jacVars.Ω
    Γ = jacVars.Γ

    ret = zeros([nV, nV])
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

    if VISCOUS:
        ret[1, 5:14] = Φ[d].ravel()
        ret[2:5, 5:14] = -dσdA[d].reshape([3,9])
        ret[5+d, 2:5] = A[0]
        ret[8+d, 2:5] = A[1]
        ret[11+d, 2:5] = A[2]
        ret[5+d, 5:8] = v
        ret[8+d, 8:11] = v
        ret[11+d, 11:14] = v


    if THERMAL:
        ret[1, 14:17] = α2 * ρvd * J
        ret[1, 14+d] += α2 * T
        ret[14:17, 0] = v[d] * J
        ret[14+d, 0] -= T / ρ
        ret[14+d, 1] = T / (p+pINF)
        ret[14:17, 2+d] = ρ * J
        for i in range(14,17):
            ret[i, i] = ρvd

    if REACTIVE:
        λ = P.λ
        Qc = MP.Qc
        ret[17, 0] = v[d] * λ
        ret[17, 2+d] = ρ * λ
        ret[17, 17] = ρvd
        ret[1, 17] += Qc * ρvd

    return ret
