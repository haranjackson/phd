from numpy import dot, eye, zeros

from auxiliary.funcs import L2_1D, L2_2D
from gpr.matrices.conserved import block, source
from gpr.matrices.jacobians import jacobian_variables, dFdP, dPdQ, dQdP
from gpr.variables.eos import E_A, E_J
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.state import sigma, sigma_A, temperature
from gpr.variables.vectors import primitive


def system_primitive(Q, d, params, subsystems):
    """ Returns the system matrix in the dth direction for the system of primitive variables,
        calculated directly
    """
    P = primitive(Q, params, subsystems)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    y = params.y; pINF = params.pINF
    sig = sigma(ρ, A)
    dσdA = sigma_A(ρ, A)

    ret = v[d] * eye(18)
    ret[0, 2+d] = ρ
    ret[1, 2+d] = y * p
    ret[1, 14+d] = (y-1) * params.α2 * T
    ret[2+d, 1] = 1 / ρ

    ret[2:5, 0] = -sig[d] / ρ**2
    ret[2:5, 5:14] = -1 / ρ * dσdA[d].reshape([3,9])

    ret[5+d, 2:5] = A[0]
    ret[8+d, 2:5] = A[1]
    ret[11+d, 2:5] = A[2]

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not subsystems.reactive:
        ret[17, 17] = 0

    return ret

def system_primitive_reordered(Q, d, params, subsystems):
    """ Returns the system matrix in the dth direction for the system of primitive variables,
        calculated directly.
        NOTE: Currently in column-major form.
    """
    P = primitive(Q, params, subsystems)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    γ = params.γ; pINF = params.pINF
    sig = sigma(ρ, A)
    dsdA = sigma_A(ρ, A)

    ret = v[d] * eye(18)
    ret[0, 11+d] = ρ
    ret[1, 11+d] = γ * p
    ret[1, 14+d] = (γ-1) * params.α2 * T
    ret[2+3*d:5+3*d, 11:14] = A
    ret[11:14, 0] = -sig[d] / ρ**2
    ret[11+d, 1] = 1 / ρ

    for i in range(3):
        ret[11+i, 2:11] = -1 / ρ * dsdA[i,d].ravel(order='F')

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not subsystems.reactive:
        ret[17, 17] = 0

    return ret


def source_primitive(Q, params, subsystems):

    ret = zeros(18)
    P = primitive(Q, params)
    ρ = P.ρ; A = P.A; J = P.J; T = P.T
    γ = params.γ

    if subsystems.viscous:
        ψ = E_A(A)
        θ1 = theta_1(A)
        ret[1] += (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if subsystems.thermal:
        H = E_J(J)
        θ2 = theta_2(ρ, T)
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

    return ret

def source_primitive_reordered(Q, params, subsystems):

    ret = zeros(18)
    P = primitive(Q, params)
    ρ = P.ρ; A = P.A; J = P.J; T = P.T
    γ = params.γ

    if subsystems.viscous:
        ψ = E_A(A)
        θ1 = theta_1(A)
        ret[1] += (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[2:11] = -ψ.ravel() / θ1

    if subsystems.thermal:
        H = E_J(J)
        θ2 = theta_2(ρ, T)
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

    return ret


def system_primitive_numeric(Q, d, params, subsystems):
    """ Returns the systems matrix in the dth direction for the system of primitive variables, using
        the constituent Jacobian matrices
    """
    P = primitive(Q, params, subsystems)
    jacVars = jacobian_variables(P, params)
    ret = dot(block(P.v, d, subsystems.viscous), dQdP(P, params, jacVars, subsystems))
    ret += dFdP(P, d, params, jacVars, subsystems)
    return dot(dPdQ(P, params, jacVars, subsystems), ret)

def source_primitive_numeric(Q, params):

    S = source(Q, params)
    P = primitive(Q, params)
    jacVars = jacobian_variables(P, params)
    DPDQ = dPdQ(P, params, jacVars)
    return dot(DPDQ, S)
