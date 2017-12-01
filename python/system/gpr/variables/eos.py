from numba import jit

from system.gpr.variables.mg import Γ_MG, e_ref, p_ref
from system.gpr.misc.functions import AdevG, dev, gram, L2_1D, L2_2D
from options import VISCOUS, THERMAL, REACTIVE


def E_1(ρ, p, PAR):
    """ Returns the microscale energy for the Mie-Gruneisen EOS
    """
    Γ = Γ_MG(ρ, PAR)
    p0 = p_ref(ρ, PAR)
    e0 = e_ref(ρ, PAR)
    return e0 + (p - p0) / (ρ * Γ)

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

def E_R(λ, Qc):
    """ Returns the microscale energy corresponding to the chemical energy in a
        reactive material
    """
    return Qc * (λ - 1)

def total_energy(ρ, p, v, A, J, λ, PAR):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, PAR)

    if VISCOUS:
        ret += E_2A(A, PAR.cs2)

    if THERMAL:
        ret += E_2J(J, PAR.α2)

    if REACTIVE:
        ret += E_R(λ, PAR.Qc)

    ret += E_3(v)

    return ret

@jit
def dEdA(A, cs2):
    """ Returns the partial derivative of E by A
    """
    G = gram(A)
    return cs2 * AdevG(A,G)

@jit
def dEdJ(J, α2):
    """ Returns the partial derivative of E by J
    """
    return α2 * J

def E_to_T(E, A, J, PAR):
    """ Returns the temperature of an ideal gas, given the energy
        (minus the kinetic energy or any chemical energy)
    """
    E1 = E - E_2A(A, PAR.cs2) - E_2J(J, PAR.α2)
    return E1 / PAR.cv
