from numba import jit
from numpy import dot, zeros

from system.gpr.systems.jacobians import dFdP, dPdQ
from system.gpr.variables.material_functions import K_arr, K_dis, K_ing
from system.gpr.misc.structures import Cvec_to_Pclass
from options import VISCOUS, THERMAL, MULTI, REACTIVE, nV


def flux_cons_ref(ret, Q, d, PAR):

    P = Cvec_to_Pclass(Q, PAR)

    ρ1 = P.ρ1
    ρ = P.ρ
    p = P.p
    E = P.E
    v = P.v
    z = P.z

    vd = v[d]
    ρvd = ρ * vd

    α2 = PAR.α2

    ret[0] += z * ρ1 * vd
    ret[1] += ρvd * E + p * vd
    ret[2:5] += ρvd * v
    ret[2+d] += p

    if VISCOUS:

        A = P.A
        σ = P.σ

        σd = σ[d]
        ret[1] -= dot(σd, v)
        ret[2:5] -= σd

        Av = dot(A, v)
        ret[5+d] += Av[0]
        ret[8+d] += Av[1]
        ret[11+d] += Av[2]

    if THERMAL:

        J = P.J
        T = P.T
        q = P.q

        ret[1] += q[d]
        ret[14:17] += ρvd * J
        ret[14+d] += T

    if MULTI:

        λ = P.λ
        ρ2 = P.ρ2

        ret[17] += (1-z) * ρ2 * vd
        ret[18] += ρvd * z
        if REACTIVE:
            ret[19] += (1-z) * ρ2 * vd * λ

@jit
def block_cons_ref(ret, Q, d, PAR):

    P = Cvec_to_Pclass(Q, PAR)

    v = P.v
    vd = v[d]
    for i in range(5,14):
        ret[i,i] = vd
    ret[5+d, 5+d:8+d] -= v
    ret[8+d, 8+d:11+d] -= v
    ret[11+d, 11+d:14+d] -= v

def source_cons_ref(ret, Q, PAR):

    P = Cvec_to_Pclass(Q, PAR)

    ρ = P.ρ
    ψ = P.ψ()
    H = P.H()
    θ1 = P.θ1()
    θ2 = P.θ2()

    if VISCOUS:
        ret[5:14] = - ψ.ravel() / θ1

    if THERMAL:
        ret[14:17] = - ρ * H / θ2

    if REACTIVE:

        z = P.z
        ρ2 = P.ρ2

        if PAR.REACTION == 'a':
            K = - K_arr(P, PAR)
        elif PAR.REACTION == 'd':
            K = - K_dis(P, PAR)
        elif PAR.REACTION == 'i':
            K = - K_ing(P, PAR)

        ret[19] = (1-z) * ρ2 * K

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

def Bdot_cons(ret, x, Q, d, PAR):

    P = Cvec_to_Pclass(Q, PAR)
    v = P.v
    if d==0:
        B0dot(ret, x, v)
    elif d==1:
        B1dot(ret, x, v)
    else:
        B2dot(ret, x, v)

def system_cons(Q, d, PAR):
    """ Returns the Jacobian in the dth direction
    """
    P = Cvec_to_Pclass(Q, PAR)
    DFDP = dFdP(P, d)
    DPDQ = dPdQ(P)
    B = zeros([nV, nV])
    block_cons_ref(B, Q, d, PAR)
    return dot(DFDP, DPDQ) + B
