from numba import jit
from numpy import dot, zeros

from auxiliary.funcs import dot3
from gpr.variables.eos import E_A
from gpr.variables.material_functions import theta_1, theta_2, arrhenius_reaction_rate
from gpr.variables.material_functions import discrete_ignition_temperature_reaction_rate
from gpr.variables.vectors import primitive
from gpr.variables.state import sigma
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
