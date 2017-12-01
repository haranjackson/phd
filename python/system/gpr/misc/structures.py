from numpy import zeros

from system.gpr.misc.functions import gram
from system.gpr.variables.eos import total_energy, dEdA, dEdJ, E_1
from system.gpr.variables.sources import theta1inv, theta2inv
from system.gpr.variables.state import heat_flux, pressure, temperature, entropy
from system.gpr.variables.state import sigma, dsigmadA, Sigma
from options import nV, VISCOUS, THERMAL, MULTI, REACTIVE


class Cvec_to_Pclass():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """
    def __init__(self, Q, PAR):

        if MULTI:
            self.ρ = Q[0] + Q[17]
            self.z = Q[18] / self.ρ
            self.ρ1 = Q[0] / self.z
            self.ρ2 = Q[17] / self.z
            if REACTIVE:
                self.λ = Q[19] / Q[17]
        else:
            self.ρ = Q[0]
            self.ρ1 = self.ρ
            self.z = 1

        self.E  = Q[1] / self.ρ
        self.v  = Q[2:5] / self.ρ

        if VISCOUS:
            self.A  = Q[5:14].reshape([3,3])
            self.σ = sigma(self.ρ, self.A, PAR.cs2)

        if THERMAL:
            self.J  = Q[14:17] / self.ρ
        else:
            self.J = zeros(3)

        if REACTIVE:
            self.p = pressure(self.ρ, self.E, self.v, self.A, self.J, PAR,
                              self.λ)
        else:
            self.p = pressure(self.ρ, self.E, self.v, self.A, self.J, PAR)

        self.T = temperature(self.ρ, self.p, PAR)

        if THERMAL:
            self.q = heat_flux(self.T, self.J, PAR.α2)

        self.PAR = PAR

    def s(self):
        return entropy(self.ρ, self.p, self.PAR)

    def dσdA(self):
        return dsigmadA(self.ρ, self.A, self.PAR.cs2)

    def Σ(self):
        return Sigma(self.p, self.ρ, self.A, self.PAR.cs2)

    def ψ(self):
        return dEdA(self.A, self.PAR.cs2)

    def H(self):
        return dEdJ(self.J, self.PAR.α2)

    def G(self):
        return gram(self.A)

    def θ1_1(self):
        return theta1inv(self.A, self.PAR)

    def θ2_1(self):
        return theta2inv(self.ρ, self.T, self.PAR)

    def E1(self):
        return E_1(self.ρ, self.p, self.PAR)


def Cvec(ρ1, p, v, A, J, PAR, ρ2=None, z=1, λ=None):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(nV)

    if MULTI:
        ρ = z * ρ1 + (1-z) * ρ2
        Q[0] = z * ρ1
        Q[17] = (1-z) * ρ2
        Q[18] = z * ρ
        if REACTIVE:
            Q[19] = (1-z) * ρ2 * λ
    else:
        ρ = ρ1
        Q[0] = ρ

    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, PAR)
    Q[2:5] = ρ * v

    if VISCOUS:
        Q[5:14] = A.ravel()

    if THERMAL:
        Q[14:17] = ρ * J

    return Q

def Pvec(P):
    ret = zeros(nV)
    ret[0] = P.ρ
    ret[1] = P.p
    ret[2:5] = P.v
    ret[5:14] = P.A.ravel()
    ret[14:17] = P.J
    return ret

def Pvec_to_Cvec(P, PAR):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3,3])

    if REACTIVE:
        λ = P[17]
    else:
        λ = 0

    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], λ, PAR)
    Q[2:5] *= ρ
    Q[14:] *= ρ
    return Q

def Cvec_to_Pvec(Q, PAR):
    """ Returns the vector of primitive variables in standard ordering,
        given the vector of conserved variables.
    """
    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3,3])
    J = Q[14:17] / ρ

    if REACTIVE:
        λ = Q[17] / ρ
    else:
        λ = None

    p = pressure(ρ, E, v, A, J, PAR, λ)

    ret = Q.copy()
    ret[1] = p
    ret[2:5] /= ρ
    ret[14:] /= ρ

    return ret
