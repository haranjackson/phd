from numpy import eye, zeros

from gpr.misc.functions import L2_1D, L2_2D
from gpr.misc.structures import Cvec_to_Pclass
from gpr.variables.shear import c_s2, dc_s2dρ
from gpr.variables.wavespeeds import c_0, c_h

from options import nV, VISCOUS, THERMAL


def source_prim_ref(ret, P):

    ρ = P.ρ

    ψ = P.ψ()
    H = P.H()
    θ1_1 = P.θ1_1()
    θ2_1 = P.θ2_1()

    dEdp = P.dEdp()
    cs2 = c_s2(ρ, MP)
    dcs2dρ = dc_s2dρ(ρ, MP)

    if VISCOUS:
        ret[1] = (1 / dEdp - ρ**2 / cs2 * dcs2dρ) * L2_2D(ψ) * θ1_1
        ret[5:14] = -ψ.ravel() * θ1_1

    if THERMAL:
        ret[1] += 1 / dEdp * L2_1D(H) * θ2_1
        ret[14:17] = -H * θ2_1


def source_prim(Q, MP):
    ret = zeros(nV)
    P = Cvec_to_Pclass(Q, MP)
    source_prim_ref(ret, P)
    return ret


def system_prim(Q, d, MP):

    P = Cvec_to_Pclass(Q, MP)

    ρ = P.ρ
    p = P.p()
    A = P.A
    v = P.v

    c0 = c_0(ρ, p, A, MP)

    ret = v[d] * eye(nV)
    ret[0, 2 + d] = ρ
    ret[1, 2 + d] = ρ * c0**2
    ret[2 + d, 1] = 1 / ρ

    if VISCOUS:

        σ = P.σ()
        dσdρ = P.dσdρ()
        dσdA = P.dσdA()

        ret[1, 2:5] += σ[d] - ρ * dσdρ[d]
        ret[2:5, 0] = -1 / ρ * dσdρ[d]
        ret[2:5, 5:14] = -1 / ρ * dσdA[d].reshape([3, 9])

        ret[5 + d, 2:5] = A[0]
        ret[8 + d, 2:5] = A[1]
        ret[11 + d, 2:5] = A[2]

    if THERMAL:

        dTdρ = P.dTdρ()
        dTdp = P.dTdp()
        T = P.T()
        ch = c_h(ρ, T, MP)

        ret[1, 14 + d] = ρ * ch**2 / dTdp
        ret[14 + d, 0] = dTdρ / ρ
        ret[14 + d, 1] = dTdp / ρ

    return ret
