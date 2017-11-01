from numpy import zeros

from system.gpr.misc.functions import gram
from system.gpr.variables.eos import total_energy, dEdA, dEdJ, E_1
from system.gpr.variables.material_functions import theta_1, theta_2
from system.gpr.variables.state import heat_flux, pressure, temperature, entropy
from system.gpr.variables.state import sigma, dsigmadA, Sigma
from options import nV, THERMAL, REACTIVE


class Cvec_to_Pclass():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """
    def __init__(self, Q, PAR):
        self.ρ = Q[0]
        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3,3])
        self.J = Q[14:17] / self.ρ

        if REACTIVE:
            self.λ = Q[17] / self.ρ
        else:
            self.λ = 0

        self.p = pressure(self.E, self.v, self.A, self.ρ, self.J, self.λ, PAR)
        self.T = temperature(self.ρ, self.p, PAR.γ, PAR.pINF, PAR.cv)
        self.q = heat_flux(self.T, self.J, PAR.α2)
        self.σ = sigma(self.ρ, self.A, PAR.cs2)

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

    def θ1(self):
        return theta_1(self.A, self.PAR.cs2, self.PAR.τ1)

    def θ2(self):
        return theta_2(self.ρ, self.T, self.PAR.ρ0, self.PAR.T0, self.PAR.α2,
                       self.PAR.τ2)

    def E1(self):
        return E_1(self.ρ, self.p, self.PAR.γ, self.PAR.pINF)


def Cvec(ρ, p, v, A, J, λ, PAR):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(nV)

    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, PAR)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()

    if THERMAL:
        Q[14:17] = ρ * J

    if REACTIVE:
        Q[17] = ρ * λ

    return Q

def Pvec_reordered(P):
    ret = zeros(nV)
    ret[0] = P.ρ
    ret[1] = P.p
    ret[2:11] = P.A.ravel()
    ret[11:14] = P.v
    ret[14:17] = P.J
    return ret

def Pvec_reordered_to_Cvec(P, PAR):
    """ Returns the vector of conserved variables, given the vector of
        (reordered) primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    p = P[1]
    A = P[2:11].reshape([3,3])
    v = P[11:14]
    J = P[14:17]

    if REACTIVE:
        λ = P[17]
    else:
        λ = 0

    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, PAR)
    Q[2:5] = ρ * v
    Q[5:14] = P[2:11]
    Q[14:] *= ρ

    return Q

def Pvec_to_Cvec(P, PAR):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3,3])
    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], P[17], PAR)
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
        λ = 0

    p = pressure(E, v, A, ρ, J, λ, PAR)

    ret = Q.copy()
    ret[1] = p
    ret[2:5] /= ρ
    ret[14:] /= ρ

    return ret
