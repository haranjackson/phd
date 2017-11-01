from numpy import array, dot, eye, zeros

from system.gpr.misc.functions import L2_1D, L2_2D
from system.gpr.variables.eos import E_A, E_J
from system.gpr.variables.material_functions import theta_1, theta_2
from system.gpr.variables.state import sigma, sigma_A, temperature
from system.gpr.misc.structures import Cvec_to_Pclass, Cvec_to_Pvec
from options import nV, VISCOUS, THERMAL, REACTIVE


def source_prim_ref(ret, Q, PAR):

    P = Cvec_to_Pvec(Q, PAR)
    ρ = P[0]
    γ = PAR.γ

    if VISCOUS:
        A = P[5:14].reshape([3,3])
        ψ = E_A(A, PAR.cs2)
        θ1 = theta_1(A, PAR.cs2, PAR.τ1)

        ret[1] = (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if THERMAL:
        J = P[14:17]
        T = temperature(ρ, P[1], γ, PAR.pINF, PAR.cv)
        H = E_J(J, PAR.α2)
        θ2 = theta_2(ρ, T, PAR.ρ0, PAR.T0, PAR.α2, PAR.τ2)

        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

def source_prim(Q, PAR):
    ret = zeros(nV)
    source_prim_ref(ret, Q, PAR)
    return ret

def system_prim(Q, d, PAR, pForm=1):

    P = Cvec_to_Pclass(Q, PAR)
    ρ = P.ρ; p = P.p; A = P.A; v = P.v; T = P.T
    γ = PAR.γ; pINF = PAR.pINF; cs2 = PAR.cs2; cv = PAR.cv; α2 = PAR.α2
    Γ = γ-1

    ret = v[d] * eye(nV)
    ret[0, 2+d] = ρ

    ret[2:5, 0] = -P.σ()[d] / ρ**2
    ret[2:5, 5:14] = -1 / ρ * P.dσdA()[d].reshape([3,9])

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

def reordered(X, perm=array([0,1,5,8,11,6,9,12,7,10,13,2,3,4,14,15,16])):
    if len(X.shape) == 1:
        return X[perm]
    else:
        return X[:,perm][perm,:]
