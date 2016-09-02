""" Some variables appearing in the HPR model
"""
from numba import jit
from numpy import dot, sqrt

from auxiliary.funcs import L2_1D, L2_2D, dev, AdevG, GdevG, gram
from options import reactiveEOS, minE


@jit
def c_0(r, p, y, pINF):
    """ Returns the adiabatic sound speed of a stiffened gas.
        NB The result for an ideal gas is obtained if pINF=0.
    """
    return sqrt(y * (p+pINF) / r)

@jit
def c_inf(r, p, y, pINF, cs2):
    """ Returns the longitudinal characteristic speed
    """
    c0 = c_0(r, p, y, pINF)
    return sqrt(c0**2 + 4/3 * cs2)

def c_h(r, T, alpha, cv):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    return alpha / r * sqrt(T / cv)

@jit
def E_1(r, p, y, pINF):
    """ Returns the microscale energy corresponding to a stiffened gas.
        NB The ideal gas equation is obtained if pINF=0.
    """
    return (p + y*pINF) / ((y-1) * r)

def E_1r(c, Qc):
    """ Returns the microscale energy corresponding to the chemical energy in a discrete ignition
        temperature reaction
    """
    return Qc * (c - minE)

@jit
def E_2A(A, cs2):
    """ Returns the mesoscale energy
    """
    G = gram(A)
    return cs2 / 4 * L2_2D(dev(G))

@jit
def E_2J(J, alpha2):
    """ Returns the mesoscale energy
    """
    return alpha2 / 2 * L2_1D(J)

@jit
def E_3(v):
    """ Returns the macroscale energy
    """
    return L2_1D(v) / 2

def total_energy(r, p, v, A, J, c, params, viscous, thermal, reactive):
    """ Returns the total energy
    """
    ret = E_1(r, p, params.y, params.pINF) + E_3(v)
    if viscous:
        ret += E_2A(A, params.cs2)
    if thermal:
        ret += E_2J(J, params.alpha2)
    if reactive and reactiveEOS:
        ret += E_1r(c, params.Qc)
    return ret

def pressure(E, v, A, r, J, c, params, viscous, thermal, reactive):
    """ Returns the pressure, given the total energy, velocity, distortion matrix, and density
    """
    E1 = E - E_3(v)
    y = params.y
    if viscous:
        E1 -= E_2A(A, params.cs2)
    if thermal:
        E1 -= E_2J(J, params.alpha2)
    if reactive and reactiveEOS:
        E1 -= E_1r(c, params.Qc)
    return (y-1) * r * E1 - y * params.pINF

def entropy(Q, params, viscous, thermal, reactive):
    """ Returns the entropy of a stiffened gas, given density and pressure
    """
    r = Q[0]
    E = Q[1] / r
    A = Q[5:14].reshape([3,3], order='F')
    J = Q[14:17] / r
    v = Q[2:5] / r
    c = Q[17] / r

    p = pressure(E, v, A, r, J, c, params, viscous, thermal, reactive)
    return (p + params.pINF) / r**params.y

def density(S, p, params):
    """ Returns the density of a stiffened gas, given entropy and pressure
    """
    return ((p + params.pINF) / S) ** (1 / params.y)

@jit
def sigma(r, A, cs2):
    """ Returns the symmetric viscous shear stress tensor
    """
    G = gram(A)
    return -r * cs2 * GdevG(G)

@jit
def temperature(r, p, y, pINF, cv):
    """ Returns the temperature for an stiffened gas
    """
    return (p + pINF) / ((y-1) * r * cv)

@jit
def heat_flux(T, J, alpha2):
    """ Returns the heat flux vector
    """
    return alpha2 * T * J

@jit
def E_A(A, cs2):
    """ Returns the partial derivative of E by A
    """
    G = gram(A)
    return cs2 * AdevG(A,G)

@jit
def E_J(J, alpha2):
    """ Returns the partial derivative of E by J
    """
    return alpha2 * J

@jit
def sigma_A(r, A, cs2):
    """ Returns the tensor T_ijmn corresponding to the partial derivative of sigma_ij with respect
        to A_mn, holding r constant.
    """
    G = gram(A)
    AdevG_ = AdevG(A,G)
    ret = -2/3 * dot(G[:,:,None], A[:,None])
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for n in range(3):
                    ret[i, j, m, n] += A[m, i] * G[j, n] + A[m, j] * G[i, n]
            ret[i, j, :, j] += AdevG_[:, i]
        ret[i, :, :, i] += AdevG_.T
    return -r * cs2 * ret
