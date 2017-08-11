from numba import jit

from auxiliary.funcs import AdevG, dev, gram, L2_1D, L2_2D
from options import VISCOUS, THERMAL, REACTIVE


@jit
def E_1(ρ, p, γ, pINF):
    """ Returns the microscale energy corresponding to a stiffened gas
        NOTE: The ideal gas equation is obtained if pINF=0
    """
    return (p + γ*pINF) / ((γ-1) * ρ)

def E_1r(λ, Qc):
    """ Returns the microscale energy corresponding to the chemical energy in a
        reactive material
    """
    return Qc * (λ - 1)

@jit
def E_2A(A, cs2):
    """ Returns the mesoscale energy dependent on the distortion
    """
    G = gram(A)
    return cs2 / 4 * L2_2D(dev(G))

@jit
def E_2Avec(A, cs2):
    """ Returns the mesoscale energy dependent on the distortion
        NOTE: A must be in row-major vector form
    """
    A11 = A[0]
    A12 = A[1]
    A13 = A[2]
    A21 = A[3]
    A22 = A[4]
    A23 = A[5]
    A31 = A[6]
    A32 = A[7]
    A33 = A[8]

    a1 = A11**2 + A21**2 + A31**2
    a2 = A12**2 + A22**2 + A32**2
    a3 = A13**2 + A23**2 + A33**2

    tr = (a1+a2+a3)/3

    return cs2/4 * (2 * (  (A11*A12 + A21*A22 + A31*A32)**2
                         + (A11*A13 + A21*A23 + A31*A33)**2
                         + (A12*A13 + A22*A23 + A32*A33)**2)
                    + (a1-tr)**2 + (a2-tr)**2 + (a3-tr)**2)

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

def total_energy(ρ, p, v, A, J, λ, PAR):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, PAR.γ, PAR.pINF) + E_3(v)

    if VISCOUS:
        ret += E_2A(A, PAR.cs2)

    if THERMAL:
        ret += E_2J(J, PAR.α2)

    if REACTIVE:
        ret += E_1r(λ, PAR.Qc)

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

def energy_to_temperature(E, A, J, PAR):
    """ Returns the temperature of an ideal gas, given the energy
        (minus the kinetic energy or any chemical energy)
    """
    E1 = E - E_2A(A, PAR.cs2) - E_2J(J, PAR.α2)
    return E1 / PAR.cv
