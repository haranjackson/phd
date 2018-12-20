from gpr.misc.functions import dev, gram, L2_1D, L2_2D
from gpr.vars import mg
from gpr.vars.shear import C_0


def E_1(ρ, p, MP):
    """ Returns the microscale energy
    """
    return mg.internal_energy(ρ, p, MP)


def E_2A(ρ, A, MP):
    """ Returns the mesoscale energy dependent on the distortion
    """
    C0 = C_0(ρ, MP)
    G = gram(A)
    return C0 / 4 * L2_2D(dev(G))


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

    if A is not None:
        ret += E_2A(ρ, A, MP)

    if J is not None:
        ret += E_2J(J, MP)

    if λ is not None:
        ret += E_R(λ, MP)

    ret += E_3(v)

    return ret
