from gpr.vars import mg


def c_s2(ρ, MP):
    """ Returns the square of the characteristic velocity of propagation of
        transverse perturbations
    """
    b02 = MP.b02
    ρ0 = MP.ρ0
    β = MP.β
    return b02 * (ρ / ρ0)**β


def dc_s2dρ(ρ, MP):
    """ Returns the derivative of cs^2 with respect to ρ
    """
    cs2 = c_s2(ρ, MP)
    β = MP.β
    return β / ρ * cs2


def C_0(ρ, MP):
    """ Returns the coefficient of |dev(G)|^2 in E(ρ,p,A,J,v)
    """
    β = MP.β
    cs2 = c_s2(ρ, MP)
    Γ = mg.Γ_MG(ρ, MP)
    return (1 - β / Γ) * cs2


def dC_0dρ(ρ, MP):
    """ Returns the derivative of C0 with respect to ρ
    """
    β = MP.β

    cs2 = c_s2(ρ, MP)
    dcs2dρ = dc_s2dρ(ρ, MP)

    Γ = mg.Γ_MG(ρ, MP)
    dΓdρ = mg.dΓ_MG(ρ, MP)

    return (1 - β / Γ) * dcs2dρ + β / Γ**2 * dΓdρ * cs2
