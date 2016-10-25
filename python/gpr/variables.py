""" Some variables appearing in the HPR model
"""
from numba import jit
from numpy import dot, sqrt

from auxiliary.funcs import L2_1D, L2_2D, dev, AdevG, GdevG, gram
from options import reactiveEOS, minE


@jit
def c_0(ρ, p, γ, pINF):
    """ Returns the adiabatic sound speed of a stiffened gas.
        NB The result for an ideal gas is obtained if pINF=0.
    """
    return sqrt(γ * (p+pINF) / ρ)

@jit
def c_inf(ρ, p, γ, pINF, cs2):
    """ Returns the longitudinal characteristic speed
    """
    c0 = c_0(ρ, p, γ, pINF)
    return sqrt(c0**2 + 4/3 * cs2)

def c_h(ρ, T, α, cv):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    return α / ρ * sqrt(T / cv)

@jit
def E_1(ρ, p, γ, pINF):
    """ Returns the microscale energy corresponding to a stiffened gas.
        NB The ideal gas equation is obtained if pINF=0.
    """
    return (p + γ*pINF) / ((γ-1) * ρ)

def E_1r(λ, Qc):
    """ Returns the microscale energy corresponding to the chemical energy in a discrete ignition
        temperature reaction
    """
    return Qc * (λ - minE)

@jit
def E_2A(A, cs2):
    """ Returns the mesoscale energy
    """
    G = gram(A)
    return cs2 / 4 * L2_2D(dev(G))

@jit
def E_2J(J, α2):
    """ Returns the mesoscale energy
    """
    return α2 / 2 * L2_1D(J)

@jit
def E_3(v):
    """ Returns the macroscale energy
    """
    return L2_1D(v) / 2

def total_energy(ρ, p, v, A, J, λ, params, subsystems):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, params.γ, params.pINF) + E_3(v)
    if subsystems.viscous:
        ret += E_2A(A, params.cs2)
    if subsystems.thermal:
        ret += E_2J(J, params.α2)
    if subsystems.reactive and reactiveEOS:
        ret += E_1r(λ, params.Qc)
    return ret

def pressure(E, v, A, ρ, J, λ, params, subsystems):
    """ Returns the pressure, given the total energy, velocity, distortion matrix, and density
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
def sigma(ρ, A, cs2):
    """ Returns the symmetric viscous shear stress tensor
    """
    G = gram(A)
    return -ρ * cs2 * GdevG(G)

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
def E_A(A, cs2):
    """ Returns the partial derivative of E by A
    """
    G = gram(A)
    return cs2 * AdevG(A,G)

@jit
def E_J(J, α2):
    """ Returns the partial derivative of E by J
    """
    return α2 * J

@jit
def sigma_A(ρ, A, cs2):
    """ Returns the tensor T_ijmn corresponding to the partial derivative of sigma_ij with respect
        to A_mn, holding r constant.
    """
    G = gram(A)
    AdevGT = AdevG(A,G).T
    GA = dot(G[:,:,None], A[:,None])
    ret = GA.swapaxes(0,3) + GA.swapaxes(1,3) - 2/3 * GA
    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT
    return -ρ * cs2 * ret
