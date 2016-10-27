""" Some variables appearing in the HPR model
"""
from numba import jit
from numpy import dot

from auxiliary.funcs import AdevG, gram
from gpr.variables.eos import E_1r, E_2A, E_2J, E_3, E_A
from options import reactiveEOS


def pressure(E, v, A, ρ, J, λ, params, subsystems):
    """ Returns the pressure, given the total energy, velocity, distortion matrix, and density.
        NOTE: Only valid for EOS used for fluids by Dumbser et al.
    """
    E1 = E - E_3(v)
    γ = params.γ
    if subsystems.viscous:
        E1 -= E_2A(A, params.cs2)
    if subsystems.thermal:
        E1 -= E_2J(J, params.α2)
    if subsystems.reactive and reactiveEOS:
        E1 -= E_1r(λ, params.Qc)
    return (γ-1) * ρ * E1 - γ * params.pINF

def entropy(Q, params, subsystems):
    """ Returns the entropy of a stiffened gas, given density and pressure
    """
    ρ = Q[0]
    E = Q[1] / ρ
    A = Q[5:14].reshape([3,3], order='F')
    J = Q[14:17] / ρ
    v = Q[2:5] / ρ
    λ = Q[17] / ρ

    p = pressure(E, v, A, ρ, J, λ, params, subsystems)
    return (p + params.pINF) / ρ**params.y

def density(S, p, params):
    """ Returns the density of a stiffened gas, given entropy and pressure
    """
    return ((p + params.pINF) / S) ** (1 / params.y)

@jit
def temperature(ρ, p, γ, pINF, cv):
    """ Returns the temperature for an stiffened gas
    """
    return (p + pINF) / ((γ-1) * ρ * cv)

@jit
def heat_flux(T, J, α2):
    """ Returns the heat flux vector
    """
    return α2 * T * J

@jit
def sigma(ρ, A, cs2):
    """ Returns the symmetric viscous shear stress tensor
    """
    return -ρ * dot(A.T, E_A(A, cs2))

@jit
def sigma_A(ρ, A, cs2):
    """ Returns the tensor T_ijmn corresponding to the partial derivative of sigma_ij with respect
        to A_mn, holding r constant.
        NOTE: Only valid for EOS with E_2A = cs2/4 * (devG)**2
    """
    G = gram(A)
    AdevGT = AdevG(A,G).T
    GA = dot(G[:,:,None], A[:,None])
    ret = GA.swapaxes(0,3) + GA.swapaxes(1,3) - 2/3 * GA
    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT
    return -ρ * cs2 * ret
