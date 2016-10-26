from numba import jit
from numpy import dot, eye

from options import ndim
from auxiliary.funcs import AdevG, det3, gram, L2_1D, L2_2D
from gpr.variables.state import temperature


@jit
def sigma(ρ, A, ADEVG, cs2):
    """ Returns the symmetric viscous shear stress tensor
    """
    return -ρ * cs2 * dot(A.T, ADEVG)

def sigma_A(ρ, A, G, ADEVG, cs2):
    """ Returns the tensor T_ijmn corresponding to the partial derivative of sigma_ij with respect
        to A_mn, holding r constant.
    """
    AdevGT = ADEVG.T
    GA = dot(G[:,:,None], A[:,None])
    ret = GA.swapaxes(0,3) + GA.swapaxes(1,3) - 2/3 * GA
    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT
    return -ρ * cs2 * ret


@jit
def theta_1(A, cs2, τ1):
    """ Returns the function used in the source terms for the distortion tensor
    """
    return (cs2 * τ1) / (3 * det3(A)**(5/3))

@jit
def theta_2(ρ, T, α2, τ2, ρ0, T0):
    """ Returns the function used in the source terms for the thermal impulse vector
    """
    return α2 * τ2 * (ρ / ρ0) * (T0 / T)


def jacobian_prim(d, ρ, p, v, A, T, G, ADEVG, γ, pINF, cv, cs2, α2, reactive):
    """ Returns the Jacobian in the dth direction for the system of primitive variables, calculated
        directly
    """
    sigd = sigma(ρ, A, ADEVG, cs2)[d]
    dsdAd = sigma_A(ρ, A, G, ADEVG, cs2)[:,d]

    ret = v[d] * eye(18)
    ret[0, 2+d] = ρ
    ret[1, 2+d] = γ * p
    ret[1, 14+d] = (γ-1) * α2 * T
    ret[2+d, 1] = 1 / ρ

    ret[2:5, 0] = -sigd / ρ**2
    for i in range(3):
        ret[2+i, 5:14] = -1 / ρ * dsdAd[i].ravel(order='F')
    ret[5+3*d:5+3*(d+1), 2:5] = A

    ret[14+d, 0] = -T / ρ**2
    ret[14+d, 1] = T / (ρ * (p + pINF))

    if not reactive:
        ret[17, 17] = 0

    return ret

def source_prim_ref(ret, ρ, A, J, T, G, ADEVG, γ, α2, cs2, τ1, τ2, ρ0, T0, viscous, thermal):

    if viscous:
        ψ = cs2 * ADEVG                             # Depends on EOS
        θ1 = theta_1(A, cs2, τ1)
        ret[1] += (γ-1) * ρ * L2_2D(ψ) / θ1
        ret[5:14] -= ψ.ravel(order='F') / θ1

    if thermal:
        H = α2 * J                                  # Depends on EOS
        θ2 = theta_2(ρ, T, α2, τ2, ρ0, T0)
        ret[1] += (γ-1) * ρ * L2_1D(H) / θ2
        ret[14:17] -= H / θ2

def fill_rhs(ret, P, TP, γ, pINF, cv, cs2, α2, τ1, τ2, ρ0, T0, viscous, thermal, reactive):

    ρ, p, v, A, J = P[0], P[1], P[2:5], P[5:14].reshape([3,3], order='F'), P[14:17]

    T = temperature(ρ, p, γ, pINF, cv)
    G = gram(A)
    ADEVG = AdevG(A, G)

    for d in range(ndim):
        M = jacobian_prim(d, ρ, p, v, A, T, G, ADEVG, γ, pINF, cv, cs2, α2, reactive)
        ret += dot(M, TP[d])

    source_prim_ref(ret, ρ, A, J, T, G, ADEVG, γ, α2, cs2, τ1, τ2, ρ0, T0, viscous, thermal)
