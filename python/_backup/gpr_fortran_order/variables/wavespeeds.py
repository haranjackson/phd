from numba import jit
from numpy import sqrt


@jit
def c_0(ρ, p, γ, pINF):
    """ Returns the adiabatic sound speed of a stiffened gas.
        NOTE: The result for an ideal gas is obtained if pINF=0.
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
