from numpy import exp, inf

from gpr.misc.functions import det3, sigma_norm
from gpr.variables.state import sigma
from gpr.variables.wavespeeds import c_s2


def theta1inv(ρ, A, MP):
    """ Returns 1/θ1
    """
    τ1 = MP.τ1

    if τ1==inf:
        return 0

    cs2 = c_s2(ρ, MP)

    if MP.PLASTIC:
        σY = MP.σY
        n = MP.n
        ρ = det3(A) * MP.ρ0
        σ = sigma(ρ, A, MP)
        sn = sigma_norm(σ)
        if sn == 0:
            return 0
        τ = τ1 * (σY / sn) ** n
    else:
        τ = τ1

    return 3 * det3(A)**(5/3) / (cs2 * τ)

def theta2inv(ρ, T, MP):
    """ Returns 1/θ2
    """
    ρ0 = MP.ρ0
    T0 = MP.T0
    cα2 = MP.cα2
    τ2 = MP.τ2

    return 1 / (cα2 * τ2 * (ρ / ρ0) * (T0 / T))


def K_arr(ρ, λ, T, MP):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    Bc = MP.Bc
    Ea = MP.Ea
    Rc = MP.Rc

    return Bc * ρ * λ * exp(-Ea / (Rc*T))

def K_dis(ρ, λ, T, MP):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    Ti = MP.Ti
    Kc = MP.Kc

    if T > Ti:
        return ρ * λ * Kc
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
        ret += I * λ**b * (ρ-1-a)**x
    if φG1 > φ:
        ret += G1 * λ**c * φ**d * p**y
    if φ > φG2:
        ret += G2 * λ**e * φ**g * p**z

    return ret
