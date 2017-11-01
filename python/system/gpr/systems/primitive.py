from numpy import eye, zeros

from system.gpr.misc.functions import L2_1D, L2_2D
from system.gpr.misc.structures import Cvec_to_Pclass
from options import nV, VISCOUS, THERMAL


def source_prim_ref(ret, P):

    ρ = P.ρ

    ψ = P.ψ()
    H = P.H()
    θ1 = P.θ1()
    θ2 = P.θ2()

    PAR = P.PAR
    γ = PAR.γ

    if VISCOUS:
        ret[1] = (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if THERMAL:
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

def source_prim(Q, PAR):
    ret = zeros(nV)
    P = Cvec_to_Pclass(Q, PAR)
    source_prim_ref(ret, P)
    return ret

def system_prim(Q, d, PAR, pForm=1):

    P = Cvec_to_Pclass(Q, PAR)

    ρ = P.ρ
    p = P.p
    A = P.A
    v = P.v
    T = P.T
    σ = P.σ

    dσdA = P.dσdA()

    γ = PAR.γ
    pINF = PAR.pINF
    cv = PAR.cv
    α2 = PAR.α2
    Γ = γ-1

    ret = v[d] * eye(nV)
    ret[0, 2+d] = ρ

    ret[2:5, 0] = -σ[d] / ρ**2
    ret[2:5, 5:14] = -1 / ρ * dσdA[d].reshape([3,9])

    ret[5+d, 2:5] = A[0]
    ret[8+d, 2:5] = A[1]
    ret[11+d, 2:5] = A[2]

    if pForm:
        ret[1, 2+d] = γ * p
        ret[1, 14+d] = Γ * α2 * T
        ret[2+d, 1] = 1 / ρ
        ret[14+d, 0] = -T / ρ**2
        ret[14+d, 1] = T / (ρ * (p + pINF))
    else:
        ret[1, 2+d] = Γ * T
        ret[1, 14+d] = α2 * T / (cv * ρ)
        ret[2+d, 0] += Γ * cv * T / ρ
        ret[2+d, 1] = Γ * cv
        ret[14+d, 1] = 1 / ρ

    return ret
