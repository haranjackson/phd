from numpy import zeros

from gpr.misc.functions import gram
from gpr.vars.derivatives import dEdρ, dEdp, dEdA, dEdA_s, dEdJ, dTdρ, dTdp
from gpr.vars.eos import total_energy
from gpr.vars.sources import theta1inv, theta2inv, K_arr, K_dis, K_ing, f_δp
from gpr.vars.state import heat_flux, pressure, temperature, sigma, dsigmadρ, \
    dsigmadA, Sigma


class State():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """

    def __init__(self, Q, MP):

        self.ρ = Q[0]
        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3, 3])

        if MP.cα2:
            self.J = Q[14:17] / self.ρ
            if MP.MULTI:
                self.λ = Q[17] / Q[0]
        else:
            if MP.MULTI:
                self.λ = Q[14] / Q[0]

        self.MP = MP

    def G(self):
        return gram(self.A)

    def p(self):
        if hasattr(self, 'p_'):
            return self.p_
        else:
            # TODO: update for multi
            self.p_ = pressure(self.ρ, self.E, self.v, self.A, self.J, self.λ,
                               self.MP)
            return self.p_

    def T(self):
        if hasattr(self, 'T_'):
            return self.T_
        else:
            self.T_ = temperature(self.ρ, self.p(), self.MP)
            return self.T_

    def σ(self):
        return sigma(self.ρ, self.A, self.MP)

    def dσdρ(self):
        return dsigmadρ(self.ρ, self.A, self.MP)

    def dσdA(self):
        return dsigmadA(self.ρ, self.A, self.MP)

    def Σ(self):
        return Sigma(self.p(), self.ρ, self.A, self.MP)

    def q(self):
        return heat_flux(self.T(), self.J, self.MP)

    def dEdρ(self):
        return dEdρ(self.ρ, self.p(), self.A, self.MP)

    def dEdp(self):
        return dEdp(self.ρ, self.MP)

    def dEdA(self):
        return dEdA(self.ρ, self.A, self.MP)

    def ψ(self):
        return dEdA_s(self.ρ, self.A, self.MP)

    def H(self):
        return dEdJ(self.J, self.MP)

    def dTdρ(self):
        return dTdρ(self.ρ, self.p(), self.MP)

    def dTdp(self):
        return dTdp(self.ρ, self.MP)

    def θ1_1(self):
        return theta1inv(self.ρ, self.A, self.MP)

    def θ2_1(self):
        return theta2inv(self.ρ, self.T(), self.MP)

    def K(self):
        if self.MP.REACTION == 'a':
            return K_arr(self.ρ, self.λ, self.T(), self.MP)
        elif self.MP.REACTION == 'd':
            return K_dis(self.ρ, self.λ, self.T(), self.MP)
        elif self.MP.REACTION == 'i':
            return K_ing(self.ρ, self.λ, self.p(), self.MP)

    def f_body(self):
        return f_δp(self.MP)


def Cvec(ρ, p, v, MP, A=None, J=None, λ=None):
    """ Returns vector of conserved variables, given primitive variables
    """

    NV = 5 + 9 * (A is not None) + 3 * (J is not None) + (λ is not None)
    Q = zeros(NV)

    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, MP)
    Q[2:5] = ρ * v

    if A is not None:
        Q[5:14] = A.ravel()

    if J is not None:
        Q[14:17] = ρ * J
        if λ is not None:
            Q[17] = ρ * λ
    else:
        if λ is not None:
            Q[14] = ρ * λ

    return Q
