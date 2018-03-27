from numpy import dot, sqrt, zeros
from numpy.linalg import eigvals

from models.gpr.misc.structures import State
from models.gpr.variables.wavespeeds import c_0, c_h


def Xi1(P, d):

    ρ = P.ρ
    MP = P.MP

    if MP.THERMAL:
        ret = zeros([4, 5])
    else:
        ret = zeros([3, 5])

    ret[0, 1] = 1 / ρ

    if MP.VISCOUS:
        dσdρ = P.dσdρ()
        dσdA = P.dσdA()
        ret[:3, 0] = -1 / ρ * dσdρ[d]
        ret[:3, 2:] = -1 / ρ * dσdA[d, :, :, d]

    if MP.THERMAL:
        dTdρ = P.dTdρ()
        dTdp = P.dTdp()
        ret[3, 0] = dTdρ / ρ
        ret[3, 1] = dTdp / ρ

    return ret


def Xi2(P, d):

    ρ = P.ρ
    p = P.p()
    A = P.A
    MP = P.MP
    c0 = c_0(ρ, p, A, MP)

    if MP.THERMAL:
        ret = zeros([5, 4])
    else:
        ret = zeros([5, 3])

    ret[0, 0] = ρ
    ret[1, d] = ρ * c0**2

    if MP.VISCOUS:
        σ = P.σ()
        dσdρ = P.dσdρ()
        ret[1, :3] += σ[d] - ρ * dσdρ[d]
        ret[2:, :3] = A

    if MP.THERMAL:
        T = P.T()
        dTdp = P.dTdp()
        ch = c_h(ρ, T, MP)
        ret[1, 3] = ρ * ch**2 / dTdp

    return ret


def max_abs_eigs(Q, d, MP):
    """ Returns maximum absolute value of the eigenvalues of the GPR system
    """
    PERR_FROB = 0               # Use Perron-Frobenius approximation to max λ

    P = State(Q, MP)
    vd = P.v[d]
    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    O = dot(Ξ1, Ξ2)

    if PERR_FROB:
        rowSum = [sum(o) for o in O]
        colSum = [sum(oT) for oT in O.T]
        lam = sqrt(min(max(rowSum), max(colSum)))
    else:
        lam = sqrt(eigvals(O).max())

    if vd > 0:
        return vd + lam
    else:
        return lam - vd
