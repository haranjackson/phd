from numpy import sqrt

from models.gpr.variables.derivatives import dEdρ, dEdp


def c_0(ρ, p, A, MP):
    """ Returns the adiabatic sound speed for the Mie-Gruneisen EOS
    """
    dE_dρ = dEdρ(ρ, p, A, MP)
    dE_dp = dEdp(ρ, MP)
    return sqrt((p / ρ**2 - dE_dρ) / dE_dp)


def c_inf(ρ, p, MP):
    """ Returns the longitudinal characteristic speed
    """
    c0 = c_0(ρ, p, MP)
    cs2 = c_s2(ρ, MP)
    return sqrt(c0**2 + 4 / 3 * cs2)


def c_h(ρ, T, MP):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    cα2 = MP.cα2
    cv = MP.cv
    return sqrt(cα2 * T / cv) / ρ
