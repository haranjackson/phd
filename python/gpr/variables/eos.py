from numba import jit

from options import minE, reactiveEOS
from auxiliary.funcs import AdevG, dev, gram, L2_1D, L2_2D


@jit
def E_1(ρ, p, γ, pINF):
    """ Returns the microscale energy corresponding to a stiffened gas
        NOTE: The ideal gas equation is obtained if pINF=0
    """
    return (p + γ*pINF) / ((γ-1) * ρ)

def E_1r(λ, Qc):
    """ Returns the microscale energy corresponding to the chemical energy in a reactive material
    """
    return Qc * (λ - minE)

@jit
def E_2A(A, cs2):
    """ Returns the mesoscale energy dependent on the distortion
    """
    G = gram(A)
    return cs2 / 4 * L2_2D(dev(G))

@jit
def E_2J(J, α2):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    return α2 / 2 * L2_1D(J)

@jit
def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return L2_1D(v) / 2

def total_energy(ρ, p, v, A, J, λ, γ, pINF, cs2, α2, Qc, viscous, thermal, reactive):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, γ, pINF) + E_3(v)
    if viscous:
        ret += E_2A(A, cs2)
    if thermal:
        ret += E_2J(J, α2)
    if reactive and reactiveEOS:
        ret += E_1r(λ, Qc)
    return ret

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
