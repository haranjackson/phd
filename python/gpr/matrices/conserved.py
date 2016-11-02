from numba import jit
from numpy import dot, zeros

from auxiliary.funcs import dot3
from gpr.matrices.jacobians import jacobian_variables, dFdP, dPdQ
from gpr.variables.eos import E_A, E_J, total_energy
from gpr.variables.material_functions import theta_1, theta_2, arrhenius_reaction_rate
from gpr.variables.material_functions import discrete_ignition_temperature_reaction_rate
from gpr.variables.vectors import Cvec_to_Pvec
from gpr.variables.state import sigma, temperature
from options import reactionType, reactiveEOS


def flux_ref(ret, P, d, PAR, SYS):

    ρ = P[0]
    p = P[1]
    v = P[2:5]
    A = P[5:14].reshape([3,3])
    J = P[14:17]
    λ = P[17]

    E = total_energy(ρ, p, v, A, J, λ, PAR, SYS)
    ρvd = ρ * v[d]

    ret[1] += ρvd * E + p * v[d]

    if SYS.mechanical:
        ret[0] += ρvd
        ret[2:5] += ρvd * v
        ret[2+d] += p

    if SYS.viscous:
        σd = sigma(ρ, A, PAR.cs2)[d]
        ret[1] -= dot3(σd, v)
        ret[2:5] -= σd

        Av = dot(A,v)
        ret[5+d] += Av[0]
        ret[8+d] += Av[1]
        ret[11+d] += Av[2]

    if SYS.thermal:
        T = temperature(ρ, p, PAR.γ, PAR.pINF, PAR.cv)
        ret[1] += PAR.α2 * T * J[d]
        ret[14:17] += ρvd * J
        ret[14+d] += T

    if SYS.reactive:
        ret[17] += ρvd * λ

@jit
def block_ref(ret, v, d, viscous):

    if viscous:
        vd = v[d]
        for i in range(5,14):
            ret[i,i] = vd
        ret[5+d, 5+d:8+d] -= v
        ret[8+d, 8+d:11+d] -= v
        ret[11+d, 11+d:14+d] -= v

def source_ref(ret, P, PAR, SYS):

    ρ = P[0]
    cs2 = PAR.cs2; α2 = PAR.α2

    if SYS.viscous:
        A = P[5:14].reshape([3,3])
        Asource = - E_A(A, cs2) / theta_1(A, cs2, PAR.τ1)
        ret[5:14] = Asource.ravel()

    if SYS.thermal:
        p = P[1]
        J = P[14:17]
        T = temperature(ρ, p, PAR.γ, PAR.pINF, PAR.cv)
        Jsource = - ρ * E_J(J, α2) / theta_2(ρ, T, PAR.ρ0, PAR.T0, α2, PAR.τ2)
        ret[14:17] = Jsource

    if SYS.reactive:
        λ = P[17]
        if reactionType == 'a':
            K = arrhenius_reaction_rate(ρ, λ, T, PAR.Ea, PAR.Bc)
        elif reactionType == 'd':
            K = discrete_ignition_temperature_reaction_rate(ρ, λ, T, PAR.Kc, PAR.Ti)
        ret[17] = -K

        if not reactiveEOS:
            ret[1] = PAR.Qc * K


def flux(Q, d, PAR, SYS):
    """ Returns the flux matrix in the kth direction
    """
    P = Cvec_to_Pvec(Q, PAR, SYS)
    ret = zeros(18)
    flux_ref(ret, P, d, PAR, SYS)
    return ret

def block(v, d, viscous):
    """ Returns the nonconvervative matrix in the kth direction
    """
    ret = zeros([18, 18])
    block_ref(ret, v, d, viscous)
    return ret

def source(Q, PAR, SYS):
    """ Returns the source vector
    """
    P = Cvec_to_Pvec(Q, PAR, SYS)
    ret = zeros(18)
    source_ref(ret, P, PAR, SYS)
    return ret

@jit
def B0dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = - v1 * x[6] - v2 * x[7]
    ret[6] = v0 * x[6]
    ret[7] = v0 * x[7]
    ret[8] = - v1 * x[9] - v2 * x[10]
    ret[9] = v0 * x[9]
    ret[10] = v0 * x[10]
    ret[11] = - v1 * x[12] - v2 * x[13]
    ret[12] = v0 * x[12]
    ret[13] = v0 * x[13]

@jit
def B1dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = v1 * x[5]
    ret[6] = - v0 * x[5] - v2 * x[7]
    ret[7] = v1 * x[7]
    ret[8] = v1 * x[8]
    ret[9] = - v0 * x[8] - v2 * x[10]
    ret[10] = v1 * x[10]
    ret[11] = v1 * x[11]
    ret[12] = - v0 * x[11] - v2 * x[13]
    ret[13] = v1 * x[13]

@jit
def B2dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = v2 * x[5]
    ret[6] = v2 * x[6]
    ret[7] = - v0 * x[5] - v1 * x[6]
    ret[8] = v2 * x[8]
    ret[9] = v2 * x[9]
    ret[10] = - v0 * x[8] - v1 * x[9]
    ret[11] = v2 * x[11]
    ret[12] = v2 * x[12]
    ret[13] = - v0 * x[11] - v1 * x[12]

@jit
def Bdot(ret, x, v, d):
    if d==0:
        B0dot(ret, x, v)
    elif d==1:
        B1dot(ret, x, v)
    else:
        B2dot(ret, x, v)

def system_conserved(Q, d, PAR, SYS):
    """ Returns the Jacobian in the dth direction
    """
    P = Cvec_to_Pvec(Q, PAR, SYS)
    jacVars = jacobian_variables(P, PAR)
    DFDP = dFdP(P, d, jacVars, PAR, SYS)
    DPDQ = dPdQ(P, jacVars, PAR, SYS)
    return dot(DFDP, DPDQ) + block(P.v, d, SYS.viscous)
