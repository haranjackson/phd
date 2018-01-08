from system.gpr.variables import mg
from system.gpr.variables.wavespeeds import c_s2, dc_s2dρ
from system.gpr.misc.functions import AdevG, dev, gram, L2_1D, L2_2D
from options import VISCOUS, THERMAL, REACTIVE


def E_1(ρ, p, MP):
    """ Returns the microscale energy
    """
    return mg.internal_energy(ρ, p, MP)

def E_2A(ρ, A, MP):
    """ Returns the mesoscale energy dependent on the distortion
    """
    cs2 = c_s2(ρ, MP)
    G = gram(A)
    return cs2 / 4 * L2_2D(dev(G))

def E_2J(J, MP):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    cα2 = MP.cα2
    return cα2 / 2 * L2_1D(J)

def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return L2_1D(v) / 2

def E_R(λ, MP):
    """ Returns the microscale energy corresponding to the chemical energy in a
        reactive material
    """
    Qc = MP.Qc
    return Qc * (λ - 1)

def total_energy(ρ, p, v, A, J, λ, MP):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, MP)

    if VISCOUS:
        ret += E_2A(ρ, A, MP)

    if THERMAL:
        ret += E_2J(J, MP)

    if REACTIVE:
        ret += E_R(λ, MP)

    ret += E_3(v)

    return ret

def dEdρ(ρ, p, A, MP):
    """ Returns the partial derivative of E by ρ (holding p constant)
    """
    dcs2dρ = dc_s2dρ(ρ, MP)
    G = gram(A)
    return mg.dedρ(ρ, p, MP) + dcs2dρ / 4 * L2_2D(dev(G))

def dEdp(ρ, MP):
    """ Returns the partial derivative of E by p
    """
    return mg.dedp(ρ, MP)

def dEdv(v):
    """ Returns the partial derivative of E by v
    """
    return v

def dEdA(ρ, A, MP):
    """ Returns the partial derivative of E by A
    """
    cs2 = c_s2(ρ, MP)
    G = gram(A)
    return cs2 * AdevG(A,G)

def dEdJ(J, MP):
    """ Returns the partial derivative of E by J
    """
    cα2 = MP.cα2
    return cα2 * J
