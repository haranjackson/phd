from numba import jit
from numpy import exp

from system.gpr.misc.functions import AdevG, dev, gram, L2_1D, L2_2D
from options import VISCOUS, THERMAL, REACTIVE


def Γ_MG(ρ, PAR):
    """ Returns the Mie-Gruneisen parameter
    """
    if PAR.EOS == 'sg':
        return PAR.γ - 1
    elif PAR.EOS == 'jwl':
        return PAR.Γ0
    elif PAR.EOS == 'smg':
        return PAR.Γ0 * PAR.ρ0 / ρ

def p_ref(ρ, PAR):
    """ Returns the reference pressure in the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return - PAR.γ * PAR.pINF
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v_ = - ρ0 / ρ
        return A * exp(R1 * v_) + B * exp(R2 * v_)
    elif PAR.EOS == 'smg':
        c02 = PAR.c02
        v0 = PAR.v0
        s = PAR.s
        v = 1 / ρ
        return c02 * (v0 - v) / (v0 - s * (v0 - v))**2

def e_ref(ρ, PAR):
    """ Returns the reference energy for the Mie-Gruneisen EOS
    """
    if PAR.EOS == 'sg':
        return 0
    elif PAR.EOS == 'jwl':
        A = PAR.A
        B = PAR.B
        R1 = PAR.R1
        R2 = PAR.R2
        ρ0 = PAR.ρ0
        v0 = PAR.v0
        v_ = - ρ0 / ρ
        return A * v0 * exp(R1 * v_) / R1  +  B * v0 * exp(R2 * v_) / R2
    elif PAR.EOS == 'smg':
        v0 = PAR.v0
        v = 1 / ρ
        p0 = p_ref(ρ, PAR)
        return 0.5 * p0 * (v0 - v)

def E_1(ρ, p, PAR):
    """ Returns the microscale energy for the Mie-Gruneisen EOS
    """
    Γ = Γ_MG(ρ, PAR)
    p0 = p_ref(ρ, PAR)
    e0 = e_ref(p0, PAR)
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
