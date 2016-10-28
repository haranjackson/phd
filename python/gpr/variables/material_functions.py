from numba import jit
from numpy import exp

from options import Rc
from auxiliary.funcs import det3


@jit
def theta_1(A, cs2, τ1):
    """ Returns the function used in the source terms for the distortion tensor
        NOTE: May be more suitable to use a different form for other fluids/solids
    """
    return (cs2 * τ1) / (3 * det3(A)**(5/3))

@jit
def theta_2(ρ, T, ρ0, T0, α2, τ2):
    """ Returns the function used in the source terms for the thermal impulse vector
        NOTE: May be more suitable to use a different form for other fluids/solids
    """
    return α2 * τ2 * (ρ / ρ0) * (T0 / T)

@jit
def arrhenius_reaction_rate(ρ, λ, T, Ea, Bc):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    return Bc * ρ * λ * exp(-Ea / (Rc*T))

def discrete_ignition_temperature_reaction_rate(ρ, λ, T, Kc, Ti):
    """ Returns the rate of reaction according to discrete ignition temperature reaction kinetics
    """
    if T > Ti:
        return ρ * λ * Kc
    else:
        return 0
