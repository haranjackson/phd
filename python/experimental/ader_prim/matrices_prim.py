from numpy import dot, zeros

from auxiliary.funcs import dot3, L2_1D, L2_2D
from gpr.variables.eos import E_A, E_J, total_energy
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.state import sigma, sigma_A, temperature


def Mdot_ref(ret, P, x, d, γ, pINF, cv, α2, viscous, thermal):
    """ Returns M(P).x
    """
    ρ = P[0]
    p = P[1]

    ret += P[2+d] * x            # v[d] * x
    ret[0] += ρ * x[2+d]
    ret[1] += γ * p * x[2+d]
    ret[2+d] += x[1] / ρ

    if viscous:
        A = P[5:14].reshape([3,3])
        σ = sigma(ρ, A)
        dσdAd = sigma_A(ρ, A)[d].reshape([3,9])

        ret[2:5] -= x[0] * σ[d] / ρ**2 + dot(dσdAd, x[5:14]) / ρ
        xv = x[2:5]
        ret[5+d] += dot(A[0],xv)
        ret[8+d] += dot(A[1],xv)
        ret[11+d] += dot(A[2],xv)

    if thermal:
        T = temperature(ρ, p, γ, pINF, cv)
        ret[1] += (γ-1) * α2 * T * x[14+d]
        ret[14+d] += T / ρ * (x[1]/(p+pINF) - x[0]/ρ)

def source_primitive_ref(ret, P, γ, pINF, cv, viscous, thermal):

    ρ = P[0]
    p = P[1]

    if viscous:
        A = P[5:14].reshape([3,3])
        ψ = E_A(A)
        θ1 = theta_1(A)
        ret[1] += (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] = -ψ.ravel() / θ1

    if thermal:
        J = P[14:17]
        T = temperature(ρ, p, γ, pINF, cv)
        H = E_J(J)
        θ2 = theta_2(ρ, T)
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] = -H / θ2

def source_primitive_ret(P, γ, pINF, cv, viscous, thermal):

    ret = zeros(18)
    source_primitive_ref(ret, P, γ, pINF, cv, viscous, thermal)
    return ret


def flux_ref(ret, P, d, γ, pINF, cv, cs2, α2, Qc, mechanical, viscous, thermal, reactive):

    ρ = P[0]
    p = P[1]
    v = P[2:5]
    A = P[5:14].reshape([3,3])
    J = P[14:17]
    λ = P[17]

    E = total_energy(ρ, p, v, A, J, λ, γ, pINF, cs2, α2, Qc, viscous, thermal, reactive)
    ρvd = ρ * v[d]

    ret[1] += ρvd * E + p * v[d]

    if mechanical:
        ret[0] += ρvd
        ret[2:5] += ρvd * v
        ret[2+d] += p

    if viscous:
        σd = sigma(ρ, A, cs2)[d]
        ret[1] -= dot3(σd, v)
        ret[2:5] -= σd

        Av = dot(A,v)
        ret[5+d] += Av[0]
        ret[8+d] += Av[1]
        ret[11+d] += Av[2]

    if thermal:
        T = temperature(ρ, p, γ, pINF, cv)
        ret[1] += α2 * T * J[d]
        ret[14:17] += ρvd * J
        ret[14+d] += T

    if reactive:
        ret[17] += ρvd * λ
