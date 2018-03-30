from numpy import dot, zeros

from models.gpr.systems.jacobians import dFdP, dPdQ
from models.gpr.misc.structures import State

from options import NV, LSET


def flux_cons(Q, d, MP):

    ret = zeros(NV)

    P = State(Q, MP)

    ρ = P.ρ
    p = P.p()
    E = P.E
    v = P.v

    vd = v[d]
    ρvd = ρ * vd

    if MP.MULTI:
        ρ1 = P.ρ1
        z = P.z
        ret[0] = z * ρ1 * vd
    else:
        ret[0] = ρvd

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

    return ret


def source_cons(Q, MP):

    ret = zeros(NV)

    P = State(Q, MP)

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

    return ret


def nonconservative_matrix_cons(Q, d, MP):

    ret = zeros([NV, NV])

    if MP.VISCOUS:

        P = State(Q, MP)

        v = P.v
        vd = v[d]

        for i in range(5, 14):
            ret[i, i] = vd
        ret[5 + d, 5 + d:8 + d] -= v
        ret[8 + d, 8 + d:11 + d] -= v
        ret[11 + d, 11 + d:14 + d] -= v

        for i in range(1, LSET + 1):
            ret[-i, -i] = vd

    return ret


def system_cons(Q, d, MP):
    """ Returns the Jacobian in the dth direction
    """
    P = State(Q, MP)
    DFDP = dFdP(P, d)
    DPDQ = dPdQ(P)
    B = nonconservative_matrix_cons(Q, d, MP)
    return dot(DFDP, DPDQ) + B
