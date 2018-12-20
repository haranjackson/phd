from numpy import exp, sqrt
from numpy.linalg import norm

from gpr.misc.functions import det3, sigma_norm

from gpr.vars.state import sigma
from gpr.vars.shear import c_s2


def theta1inv(ρ, A, MP):
    """ Returns 1/θ1
    """
    detA5_3 = det3(A)**(5 / 3)

    if MP.SOLID:

        if MP.POWER_LAW:   # plastic solid

            τ0 = MP.τ0
            n = MP.n
            σY = MP.σY

            cs2 = c_s2(ρ, MP)
            σ = sigma(ρ, A, MP)
            sn = sigma_norm(σ)
            sn = min(sn, 1e8)   # Hacky fix

            return 3 * detA5_3 / (cs2 * τ0) * (sn / σY) ** n

        else:   # elastic solid
            return 0

    else:

        n = MP.n
        const = detA5_3 * ρ0 / (2 * μ**(1/n))

        if MP.POWER_LAW:  # power law fluid

            σ = sigma(ρ, A, MP)
            sn = norm(σ) / sqrt(2.)
            sn = min(sn, 1e8)   # Hacky fix

            return const * sn**((1-n)/n)

        else:   # newtonian fluid
            return const



def theta2inv(ρ, T, MP):
    """ Returns 1/θ2
    """
    return T / (MP.κ * ρ)


def K_arr(ρ, λ, T, MP):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    Bc = MP.Bc
    Ea = MP.Ea
    Rc = MP.Rc

    return Bc * λ * exp(-Ea / (Rc * T))


def K_dis(ρ, λ, T, MP):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    Ti = MP.Ti
    Kc = MP.Kc

    if T > Ti:
        return λ * Kc
    else:
        return 0


def K_ing(ρ, λ, p, MP):
    """ Returns the rate of reaction according to ignition and growth
        reaction kinetics
    """
    I = MP.I
    G1 = MP.G1
    G2 = MP.G2
    a = MP.a
    b = MP.b
    c = MP.c
    d = MP.d
    e = MP.e
    g = MP.g
    x = MP.x
    y = MP.y
    z = MP.z
    φIG = MP.φIG
    φG1 = MP.φG1
    φG2 = MP.φG2

    φ = 1 - λ

    ret = 0
    if φIG > φ:
        ret += I * λ**b * (ρ - 1 - a)**x
    if φG1 > φ:
        ret += G1 * λ**c * φ**d * p**y
    if φ > φG2:
        ret += G2 * λ**e * φ**g * p**z

    return ret


def f_δp(MP):
    """ Returns the body force density produced by a constant pressure gradient
    """
    return MP.δp
