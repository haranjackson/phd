from numpy import dot, sqrt, zeros
from numpy.linalg import eigvals

from gpr.misc.structures import State
from gpr.vars.wavespeeds import c_0, c_h


def Xi1(P, d, MP):

    ρ = P.ρ

    if THERMAL:
        ret = zeros([4, 5])
    else:
        ret = zeros([3, 5])

    ret[0, 1] = 1 / ρ

    if VISCOUS:
        dσdρ = P.dσdρ()
        dσdA = P.dσdA()
        ret[:3, 0] = -1 / ρ * dσdρ[d]
        ret[:3, 2:] = -1 / ρ * dσdA[d, :, :, d]

    if THERMAL:
        dTdρ = P.dTdρ()
        dTdp = P.dTdp()
        ret[3, 0] = dTdρ / ρ
        ret[3, 1] = dTdp / ρ

    return ret


def Xi2(P, d, MP):

    ρ = P.ρ
    p = P.p()
    A = P.A
    c0 = c_0(ρ, p, A, MP)

    if THERMAL:
        ret = zeros([5, 4])
    else:
        ret = zeros([5, 3])

    ret[0, 0] = ρ
    ret[1, d] = ρ * c0**2

    if VISCOUS:
        σ = P.σ()
        dσdρ = P.dσdρ()
        ret[1, :3] += σ[d] - ρ * dσdρ[d]
        ret[2:, :3] = A

    if THERMAL:
        T = P.T()
        dTdp = P.dTdp()
        ch = c_h(ρ, T, MP)
        ret[1, 3] = ρ * ch**2 / dTdp

    return ret



def max_eig(Q, d, MP):
    """ Returns maximum absolute value of the eigenvalues of the GPR system
    """
    P = State(Q, MP)
    vd = P.v[d]
    Ξ1 = Xi1(P, d, MP)
    Ξ2 = Xi2(P, d, MP)
    O = dot(Ξ1, Ξ2)

    lam = sqrt(eigvals(O).max().real)

    if vd > 0:
        return vd + lam
    else:
        return lam - vd