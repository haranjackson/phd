from numba import jit
from numpy import dot, eye, outer, tensordot, zeros

from auxiliary.funcs import L2_1D, dot3
from gpr.functions import primitive, theta_1, theta_2
from gpr.functions import arrhenius_reaction_rate, discrete_ignition_temperature_reaction_rate
from gpr.variables import heat_flux, sigma, E_1, E_A, sigma_A
from options import reactionType, reactiveEOS


def flux_ref(ret, P, d, params, subsystems):

    ρ = P.ρ; p = P.p; v = P.v
    ρvd = ρ * v[d]

    ret[1] = ρvd * P.E + p * v[d]

    if subsystems.mechanical:
        ret[0] = ρvd
        ret[2:5] = ρvd * v
        ret[2+d] += p

    if subsystems.viscous:
        A = P.A
        σd = sigma(ρ, A, params.cs2)[d]
        ret[1] -= dot3(σd, v)
        ret[2:5] -= σd
        ret[5+3*d:8+3*d] = dot(A, v)

    if subsystems.thermal:
        T = P.T
        J = P.J
        ret[1] += params.α2 * T * J[d]
        ret[14:17] = ρvd * J
        ret[14+d] += T

    if subsystems.reactive:
        ret[17] = ρvd * P.λ

@jit
def block_ref(ret, v, d, viscous):

    if viscous:
        vd = v[d]
        for i in range(5,14):
            ret[i,i] = vd
        k1 = 5+3*d
        for i in range(3):
            vi = v[i]
            k2 = 5+3*i
            for j in range(3):
                ret[k1+j, k2+j] -= vi

def source_ref(ret, P, params, subsystems):

    ρ = P.ρ

    if subsystems.viscous:
        A = P.A
        Asource = - E_A(A, params.cs2) / theta_1(A, params)
        ret[5:14] = Asource.ravel(order='F')

    if subsystems.thermal:
        T = P.T
        Jsource = - ρ * params.α2 * P.J / theta_2(ρ, T, params)
        ret[14:17] = Jsource

    if subsystems.reactive:
        if reactionType == 'a':
            K = arrhenius_reaction_rate(ρ, P.λ, T, params)
        elif reactionType == 'd':
            K = discrete_ignition_temperature_reaction_rate(ρ, P.λ, T, params)
        ret[17] = -K

        if not reactiveEOS:
            ret[1] = params.Qc * K


def flux(Q, d, params, subsystems):
    """ Returns the flux matrix in the kth direction
    """
    P = primitive(Q, params, subsystems)
    ret = zeros(18)
    flux_ref(ret, P, d, params, subsystems)
    return ret

def block(v, d, viscous):
    """ Returns the nonconvervative matrix in the kth direction
    """
    ret = zeros([18, 18])
    block_ref(ret, v, d, viscous)
    return ret

def source(Q, params, subsystems):
    """ Returns the source vector
    """
    P = primitive(Q, params, subsystems)
    ret = zeros(18)
    source_ref(ret, P, params, subsystems)
    return ret

@jit
def B0dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[5] = - v1 * Q[8] - v2 * Q[11]
        ret[6] = - v1 * Q[9] - v2 * Q[12]
        ret[7] = - v1 * Q[10] - v2 * Q[13]
        ret[8] = v0 * Q[8]
        ret[9] = v0 * Q[9]
        ret[10] = v0 * Q[10]
        ret[11] = v0 * Q[11]
        ret[12] = v0 * Q[12]
        ret[13] = v0 * Q[13]

@jit
def B1dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[8] = - v0 * Q[5] - v2 * Q[11]
        ret[9] = - v0 * Q[6] - v2 * Q[12]
        ret[10] = - v0 * Q[7] - v2 * Q[13]
        ret[5] = v1 * Q[5]
        ret[6] = v1 * Q[6]
        ret[7] = v1 * Q[7]
        ret[11] = v1 * Q[11]
        ret[12] = v1 * Q[12]
        ret[13] = v1 * Q[13]

@jit
def B2dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[11] = - v0 * Q[5] - v1 * Q[8]
        ret[12] = - v0 * Q[6] - v1 * Q[9]
        ret[13] = - v0 * Q[7] - v1 * Q[10]
        ret[5] = v2 * Q[5]
        ret[6] = v2 * Q[6]
        ret[7] = v2 * Q[7]
        ret[8] = v2 * Q[8]
        ret[9] = v2 * Q[9]
        ret[10] = v2 * Q[10]

def Bdot(ret, Q, d, v, viscous):
    if d==0:
        B0dot(ret, Q, v, viscous)
    elif d==1:
        B1dot(ret, Q, v, viscous)
    else:
        B2dot(ret, Q, v, viscous)


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
