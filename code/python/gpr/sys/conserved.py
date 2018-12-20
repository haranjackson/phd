from numpy import dot, zeros

from gpr.misc.structures import State
from gpr.sys.jacobians import dFdP, dPdQ


def F_cons(Q, d, MP):

    ret = zeros(NV)

    P = State(Q, MP)

    ρ = P.ρ
    p = P.p()
    E = P.E
    v = P.v

    vd = v[d]
    ρvd = ρ * vd

    ret[0] = ρvd

    ret[1] = ρvd * E + p * vd
    ret[2:5] = ρvd * v
    ret[2 + d] += p

    if VISCOUS:

        A = P.A
        σ = P.σ()

        σd = σ[d]
        ret[1] -= dot(σd, v)
        ret[2:5] -= σd

        Av = dot(A, v)
        ret[5 + d] = Av[0]
        ret[8 + d] = Av[1]
        ret[11 + d] = Av[2]

    if THERMAL:

        J = P.J
        T = P.T()
        q = P.q()

        ret[1] += q[d]
        ret[14:17] = ρvd * J
        ret[14 + d] += T

    if MULTI:
        λ = P.λ
        ret[17] = ρvd * λ

    return ret


def S_cons(Q, MP):

    ret = zeros(NV)

    P = State(Q, MP)

    ρ = P.ρ

    ret[2:5] = P.f_body()

    if VISCOUS:
        ψ = P.ψ()
        θ1_1 = P.θ1_1()
        ret[5:14] = - ψ.ravel() * θ1_1

    if THERMAL:
        H = P.H()
        θ2_1 = P.θ2_1()
        ret[14:17] = - ρ * H * θ2_1

    if REACTIVE:
        K = P.K()
        ret[17] = - ρ * K

    return ret


def B_cons(Q, d, MP):

    ret = zeros([NV, NV])

    if VISCOUS:

        P = State(Q, MP)

        v = P.v
        vd = v[d]

        for i in range(5, 14):
            ret[i, i] = vd
        ret[5 + d, 5:8] -= v
        ret[8 + d, 8:11] -= v
        ret[11 + d, 11:14] -= v

    for i in range(1, LSET + 1):
        ret[-i, -i] = vd

    return ret


def M_cons(Q, d, MP):
    """ Returns the Jacobian in the dth direction
    """
    P = State(Q, MP)
    DFDP = dFdP(P, d, MP)
    DPDQ = dPdQ(P, MP)
    return dot(DFDP, DPDQ) + B_cons(Q, d, MP)
