from numba import jit
from numpy import exp, zeros

from auxiliary.funcs import det3
from gpr.variables import pressure, total_energy, temperature
from options import Rc


class primitive():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """
    def __init__(self, Q, params, subsystems):
        self.ρ = Q[0]
        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3,3], order='F').copy()
        self.J = Q[14:17] / self.ρ
        self.λ = Q[17] / self.ρ

        self.p = pressure(self.E, self.v, self.A, self.ρ, self.J, self.λ, params, subsystems)
        self.T = temperature(self.ρ, self.p, params.γ, params.pINF, params.cv)

def conserved(ρ, p, v, A, J, λ, params, subsystems):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(18)
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, params, subsystems)
    Q[2:5] = ρ * v
    Q[5:14] = A.reshape(9, order='F')
    Q[14:17] = ρ * J
    Q[17] = ρ * λ
    return Q

def primitive_vector(P):
    ret = zeros(18)
    ret[0] = P.ρ
    ret[1] = P.p
    ret[2:11] = P.A.ravel(order='F')
    ret[11:14] = P.v
    ret[14:17] = P.J
    return ret

def primitive_to_conserved(P, params, subsystems):
    Q = zeros(18)
    ρ = P[0]
    p = P[1]
    A = P[2:11].reshape([3,3], order='F')
    v = P[11:14]
    J = P[14:17]
    λ = 0
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, params, subsystems)
    Q[2:5] = ρ * v
    Q[5:14] = A.reshape(9, order='F')
    Q[14:17] = ρ * J
    return Q

@jit
def theta_1(A, params):
    """ Returns the function used in the source terms for the distortion tensor
    """
    return (params.cs2 * params.t1) / (3 * det3(A)**(5/3))

@jit
def theta_2(ρ, T, params):
    """ Returns the function used in the source terms for the thermal impulse vector
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
