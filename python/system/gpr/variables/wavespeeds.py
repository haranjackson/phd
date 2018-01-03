from numpy import sqrt

from system.gpr.variables.mg import dedρ, dedp


def c_0(ρ, p, MP):
    """ Returns the adiabatic sound speed for the Mie-Gruneisen EOS
    """
    de_dρ = dedρ(ρ, p, MP)
    de_dp = dedp(ρ, MP)
    return sqrt((p / ρ**2 - de_dρ) / de_dp)

def c_inf(ρ, p, MP):
    """ Returns the longitudinal characteristic speed
    """
    c0 = c_0(ρ, p, MP)
    cs2 = MP.cs2
    return sqrt(c0**2 + 4/3 * cs2)

def c_h(ρ, T, MP):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    α2 = MP.α2
    cv = MP.cv
    return sqrt(α2 * T / cv) / ρ
