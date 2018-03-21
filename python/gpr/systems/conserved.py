from numpy import dot, zeros

from gpr.systems.jacobians import dFdP, dPdQ
from gpr.misc.structures import Cvec_to_Pclass

from options import NV, LSET


def flux_cons_ref(ret, Q, d, MP):

    P = Cvec_to_Pclass(Q, MP)

    ρ1 = P.ρ1
    ρ = P.ρ
    p = P.p()
    E = P.E
    v = P.v
    z = P.z

    vd = v[d]
    ρvd = ρ * vd

    ret[0] = z * ρ1 * vd
    ret[1] = ρvd * E + p * vd
    ret[2:5] = ρvd * v
    ret[2 + d] += p

    if MP.VISCOUS:

        A = P.A
        σ = P.σ()

        σd = σ[d]
        ret[1] -= dot(σd, v)
        ret[2:5] -= σd

        Av = dot(A, v)
        ret[5 + d] = Av[0]
        ret[8 + d] = Av[1]
        ret[11 + d] = Av[2]

    if MP.THERMAL:

        cα2 = MP.cα2

        J = P.J
        T = P.T()
        q = P.q()

        ret[1] += q[d]
        ret[14:17] = ρvd * J
        ret[14 + d] += T

    if MP.MULTI:

        λ = P.λ
        ρ2 = P.ρ2

        ret[17] = (1 - z) * ρ2 * vd
        ret[18] = ρvd * z
        if MP.REACTIVE:
            ret[19] = (1 - z) * ρ2 * vd * λ


def source_cons_ref(ret, Q, MP):

    P = Cvec_to_Pclass(Q, MP)

    ρ = P.ρ

    ret[2:5] = P.f_body()

    if MP.VISCOUS:
        ψ = P.ψ()
        θ1_1 = P.θ1_1()
        ret[5:14] = - ψ.ravel() * θ1_1

    if MP.THERMAL:
        H = P.H()
        θ2_1 = P.θ2_1()
        ret[14:17] = - ρ * H * θ2_1

    if MP.REACTIVE:
        z = P.z
        ρ2 = P.ρ2
        K = - P.K()
        ret[19] = (1 - z) * ρ2 * K


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

    for i in range(1, LSET + 1):
        ret[-i] = v0 * x[-i]


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

    for i in range(1, LSET + 1):
        ret[-i] = v1 * x[-i]


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

    for i in range(1, LSET + 1):
        ret[-i] = v2 * x[-i]


def nonconservative_product_cons(ret, x, Q, d, MP):

    if MP.VISCOUS:
        P = Cvec_to_Pclass(Q, MP)
        v = P.v
        if d == 0:
            B0dot(ret, x, v)
        elif d == 1:
            B1dot(ret, x, v)
        else:
            B2dot(ret, x, v)


def system_cons(Q, d, MP):
    """ Returns the Jacobian in the dth direction
    """
    P = Cvec_to_Pclass(Q, MP)
    DFDP = dFdP(P, d)
    DPDQ = dPdQ(P)
    B = zeros([NV, NV])
    block_cons_ref(B, Q, d, MP)
    return dot(DFDP, DPDQ) + B
