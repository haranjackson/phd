from numpy import sqrt

from gpr.variables.mg import Γ_MG, dedρ, dedp


def c_s2(ρ, MP):
    """ Returns the square of the characteristic velocity of propagation of
        transverse perturbations
    """
    B0 = MP.B0
    ρ0 = MP.ρ0
    β = MP.β
    return B0 * (ρ/ρ0)**β

def dc_s2dρ(ρ, MP):
    """ Returns the derivative of cs^2 with respect to ρ
    """
    cs2 = c_s2(ρ, MP)
    β = MP.β
    return β / ρ * cs2

def C_0(ρ, MP):
    """ Returns the coefficient of |dev(G)|^2 in the total energy, expressed
        as a function of ρ,p,A,J,v
    """
    β = MP.β
    cs2 = c_s2(ρ, MP)
    Γ = Γ_MG(ρ, MP)
    return (1 - β/Γ) * cs2

def dC_0dρ(ρ, MP):
    """ Returns the derivative of C0 with respect to ρ
    """
    β = MP.β
    dcs2dρ = dc_s2dρ(ρ, MP)
    Γ = Γ_MG(ρ, MP)
    return (1 - β/Γ) * dcs2dρ

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
    cs2 = c_s2(ρ, MP)
    return sqrt(c0**2 + 4/3 * cs2)

def c_h(ρ, T, MP):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    cα2 = MP.cα2
    cv = MP.cv
    return sqrt(cα2 * T / cv) / ρ
