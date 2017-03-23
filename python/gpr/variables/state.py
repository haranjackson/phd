""" Some variables appearing in the HPR model
"""
from numba import jit
from numpy import dot

from auxiliary.funcs import AdevG, GdevG, gram
from gpr.variables.eos import E_1r, E_2A, E_2Avec, E_2J, E_3


def pressure(E, v, A, ρ, J, λ, PAR, SYS, vecA=0):
    """ Returns the pressure, given the total energy, velocity, distortion matrix, and density.
        NOTE: Only valid for EOS used for fluids by Dumbser et al.
    """
    E1 = E - E_3(v)
    γ = PAR.γ

    if SYS.viscous:
        if vecA:
            E1 -= E_2Avec(A, PAR.cs2)
        else:
            E1 -= E_2A(A, PAR.cs2)

    if SYS.thermal:
        E1 -= E_2J(J, PAR.α2)

    if SYS.reactive:
        E1 -= E_1r(λ, PAR.Qc)

    return (γ-1) * ρ * E1 - γ * PAR.pINF

def entropy(Q, PAR, SYS):
    """ Returns the entropy of a stiffened gas, given density and pressure
    """
    ρ = Q[0]
    E = Q[1] / ρ
    A = Q[5:14].reshape([3,3])
    J = Q[14:17] / ρ
    v = Q[2:5] / ρ
    λ = Q[17] / ρ

    p = pressure(E, v, A, ρ, J, λ, PAR, SYS)
    return (p + PAR.pINF) / ρ**PAR.y

def density(S, p, PAR):
    """ Returns the density of a stiffened gas, given entropy and pressure
    """
    return ((p + PAR.pINF) / S) ** (1 / PAR.y)

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
    G = gram(A)
    return -ρ * GdevG(G)
    # return -ρ * dot(A.T, E_A(A, cs2))     # Generic definition of σ

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
