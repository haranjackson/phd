from system.gpr.variables.mg import Γ_MG, e_ref, p_ref
from system.gpr.misc.functions import AdevG, dev, gram, L2_1D, L2_2D
from options import VISCOUS, THERMAL, REACTIVE


def E_1(ρ, p, PAR):
    """ Returns the microscale energy for the Mie-Gruneisen EOS
    """
    Γ = Γ_MG(ρ, PAR)
    pr = p_ref(ρ, PAR)
    er = e_ref(ρ, PAR)
    return er + (p - pr) / (ρ * Γ)

def E_2A(ρ, A, PAR):
    """ Returns the mesoscale energy dependent on the distortion
    """
    cs2 = PAR.cs2
    ρ0 = PAR.ρ0
    β = PAR.β
    G = gram(A)
    return cs2/4 * (ρ/ρ0)**β * L2_2D(dev(G))

def E_2J(J, PAR):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    α2 = PAR.α2
    return α2 / 2 * L2_1D(J)

def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return L2_1D(v) / 2

def E_R(λ, PAR):
    """ Returns the microscale energy corresponding to the chemical energy in a
        reactive material
    """
    Qc = PAR.Qc
    return Qc * (λ - 1)

def total_energy(ρ, p, v, A, J, λ, PAR):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, PAR)

    if VISCOUS:
        ret += E_2A(ρ, A, PAR)

    if THERMAL:
        ret += E_2J(J, PAR)

    if REACTIVE:
        ret += E_R(λ, PAR)

    ret += E_3(v)

    return ret

def dEdA(ρ, A, PAR):
    """ Returns the partial derivative of E by A
    """
    ρ0 = PAR.ρ0
    cs2 = PAR.cs2
    β = PAR.β
    G = gram(A)
    return cs2 * (ρ/ρ0)**β * AdevG(A,G)

def dEdJ(J, PAR):
    """ Returns the partial derivative of E by J
    """
    α2 = PAR.α2
    return α2 * J

def E_to_T(ρ, E, A, J, PAR):
    """ Returns the temperature of an ideal gas, given the energy
        (minus the kinetic energy or any chemical energy)
    """
    E1 = E - E_2A(ρ, A, PAR) - E_2J(J, PAR)
    return E1 / PAR.cv
