from numba import jit
from numpy import exp

from options import Rc
from auxiliary.funcs import det3


@jit
def theta_1(A, params):
    """ Returns the function used in the source terms for the distortion tensor
        NOTE: May be more suitable to use a different form for other fluids/solids
    """
    return (params.cs2 * params.t1) / (3 * det3(A)**(5/3))

@jit
def theta_2(ρ, T, params):
    """ Returns the function used in the source terms for the thermal impulse vector
        NOTE: May be more suitable to use a different form for other fluids/solids
    """
    return params.α2 * params.t2 * (ρ / params.ρ0) * (params.T0 / T)

@jit
def arrhenius_reaction_rate(ρ, λ, T, params):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    return params.Bc * ρ * λ * exp(-params.Ea / (Rc*T))

def discrete_ignition_temperature_reaction_rate(ρ, λ, T, params):
    """ Returns the rate of reaction according to discrete ignition temperature reaction kinetics
    """
    if T > params.Ti:
        return ρ * λ * params.Kc
    else:
        return 0
