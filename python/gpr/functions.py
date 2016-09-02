from numba import jit
from numpy import exp, zeros

from auxiliary.funcs import det3
from gpr.variables import pressure, total_energy, temperature
from options import Rc


class primitive():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """
    def __init__(self, Q, params, viscous, thermal, reactive):
        self.r = Q[0]
        self.E = Q[1] / self.r
        self.v = Q[2:5] / self.r
        self.A = Q[5:14].reshape([3,3], order='F').copy()
        self.J = Q[14:17] / self.r
        self.c = Q[17] / self.r

        self.p = pressure(self.E, self.v, self.A, self.r, self.J, self.c, params,
                          viscous, thermal, reactive)
        self.T = temperature(self.r, self.p, params.y, params.pINF, params.cv)

def conserved(r, p, v, A, J, c, params, viscous, thermal, reactive):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(18)
    Q[0] = r
    Q[1] = r * total_energy(r, p, v, A, J, c, params, viscous, thermal, reactive)
    Q[2:5] = r * v
    Q[5:14] = A.reshape(9, order='F')
    Q[14:17] = r * J
    Q[17] = r * c
    return Q

def primitive_vector(P):
    ret = zeros(18)
    ret[0] = P.r
    ret[1] = P.p
    ret[2:11] = P.A.ravel(order='F')
    ret[11:14] = P.v
    ret[14:17] = P.J
    return ret

def primitive_to_conserved(P, params, viscous, thermal, reactive):
    Q = zeros(18)
    r = P[0]
    p = P[1]
    A = P[2:11].reshape([3,3], order='F')
    v = P[11:14]
    J = P[14:17]
    c = 0
    Q[0] = r
    Q[1] = r * total_energy(r, p, v, A, J, c, params, viscous, thermal, reactive)
    Q[2:5] = r * v
    Q[5:14] = A.reshape(9, order='F')
    Q[14:17] = r * J
    return Q

@jit
def theta_1(A, params):
    """ Returns the function used in the source terms for the distortion tensor
    """
    return (params.cs2 * params.t1) / (3 * det3(A)**(5/3))

@jit
def theta_2(r, T, params):
    """ Returns the function used in the source terms for the thermal impulse vector
    """
    return params.alpha2 * params.t2 * (r / params.r0) * (params.T0 / T)

@jit
def arrhenius_reaction_rate(r, c, T, params):
    """ Returns the rate of reaction according to Arrhenius kinetics
    """
    return params.Bc * r * c * exp(-params.Ea / (Rc*T))

def discrete_ignition_temperature_reaction_rate(r, c, T, params):
    """ Returns the rate of reaction according to discrete ignition temperature reaction kinetics
    """
    if T > params.Ti:
        return r * c * params.Kc
    else:
        return 0
