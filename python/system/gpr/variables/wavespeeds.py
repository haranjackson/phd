from numpy import sqrt

from system.gpr.variables.mg import dedρ, dedp


def c_0(ρ, p, PAR):
    """ Returns the adiabatic sound speed for the Mie-Gruneisen EOS
    """
    de_dρ = dedρ(ρ, p, PAR)
    de_dp = dedp(ρ, PAR)
    return sqrt((p / ρ**2 - de_dρ) / de_dp)

def c_inf(ρ, p, PAR):
    """ Returns the longitudinal characteristic speed
    """
    c0 = c_0(ρ, p, PAR)
    cs2 = PAR.cs2
    return sqrt(c0**2 + 4/3 * cs2)

def c_h(ρ, T, PAR):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    α = PAR.α
    cv = PAR.cv
    return α / ρ * sqrt(T / cv)
