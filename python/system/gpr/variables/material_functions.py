from numba import jit
from numpy import exp

from system.gpr.misc.functions import det3


@jit
def theta_1(A, PAR):
    """ Returns the function used in source terms for the distortion tensor
        NOTE: May be more suitable to use a different form for other materials
    """
    cs2 = PAR.cs2
    τ1 = PAR.τ1

    if PAR.PLASTIC:
        σY = PAR.σY
        n = PAR.n
        σ = sigma(ρ, A, cs2)
        τ = τ1 * (σY / L2_2D(dev(σ))) ** n
    else:
        τ = τ1

    return (cs2 * τ) / (3 * det3(A)**(5/3))

@jit
def theta_2(ρ, T, PAR):
    """ Returns the function used in source terms for the thermal impulse vector
        NOTE: May be more suitable to use a different form for other materials
    """
    ρ0 = PAR.ρ0
    T0 = PAR.T0
    α2 = PAR.α2
    τ2 = PAR.τ2

    return α2 * τ2 * (ρ / ρ0) * (T0 / T)

@jit
def K_arr(P, PAR):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    ρ = P.ρ
    λ = P.λ
    T = P.T

    Bc = PAR.Bc
    Ea = PAR.Ea
    Rc = PAR.Rc

    return Bc * ρ * λ * exp(-Ea / (Rc*T))

def K_dis(P, PAR):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    ρ = P.ρ
    λ = P.λ
    T = P.T

    Ti = PAR.Ti
    Kc = PAR.Kc

    if T > Ti:
        return ρ * λ * Kc
    else:
        return 0

def K_ing(P, PAR):
    """ Returns the rate of reaction according to ignition and growth
        reaction kinetics
    """
    I = PAR.I
    G1 = PAR.G1
    G2 = PAR.G2
    a = PAR.a
    b = PAR.b
    c = PAR.c
    d = PAR.d
    e = PAR.e
    g = PAR.g
    x = PAR.x
    y = PAR.y
    z = PAR.z
    φIG = PAR.φIG
    φG1 = PAR.φG1
    φG2 = PAR.φG2

    ρ = P.ρ
    p = P.p
    λ = P.λ
    φ = 1 - λ

    ret = 0
    if φIG > φ:
        ret += I * λ**b * (ρ-1-a)**x
    if φG1 > φ:
        ret += G1 * λ**c * φ**d * p**y
    if φ > φG2:
        ret += G2 * λ**e * φ**g * p**z

    return ret
