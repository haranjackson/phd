from numpy import zeros

from gpr.misc.functions import gram
from gpr.variables.derivatives import dEdρ, dEdp, dEdA, dEdA_s, dEdJ, dTdρ, dTdp
from gpr.variables.eos import total_energy
from gpr.variables.sources import theta1inv, theta2inv, K_arr, K_dis, K_ing, f_δp
from gpr.variables.state import heat_flux, pressure, temperature
from gpr.variables.state import sigma, dsigmadρ, dsigmadA, Sigma


def extract_densities(Q, MP, state):
    if MP.MULTI:
        state.ρ = Q[0] + Q[17]
        state.z = Q[18] / state.ρ
        state.ρ1 = Q[0] / state.z
        state.ρ2 = Q[17] / state.z
        if MP.REACTIVE:
            state.λ = Q[19] / Q[17]
    else:
        state.ρ = Q[0]


def calculate_pressure(state):
    if hasattr(state, 'p_'):
        return state.p_
    else:
        if state.MP.THERMAL:
            if state.MP.REACTIVE:
                state.p_ = pressure(state.ρ, state.E, state.v, state.A, state.J,
                                    state.MP, state.λ)
            else:
                state.p_ = pressure(state.ρ, state.E, state.v, state.A, state.J,
                                    state.MP)
        else:
            state.p_ = pressure(state.ρ, state.E, state.v, state.A, zeros(3),
                                state.MP)
        return state.p_


def get_NV(MP):
    return 17 + int(MP.REACTIVE) + int(MP.MULTI)


class State():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """

    def __init__(self, Q, MP):

        self.NV = len(Q)

        extract_densities(Q, MP, self)

        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3, 3])
        self.J = Q[14:17] / self.ρ

        self.MP = MP

    def G(self):
        return gram(self.A)

    def p(self):
        return calculate_pressure(self)

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


def Cvec(ρ1, p, v, A, J, MP, ρ2=None, z=1, λ=None):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    if MP.THERMAL:
        Q = zeros(17)
    else:
        Q = zeros(14)

    if MP.MULTI:
        ρ = z * ρ1 + (1 - z) * ρ2
        Q[0] = z * ρ1
        Q[17] = (1 - z) * ρ2
        Q[18] = z * ρ
        if MP.REACTIVE:
            Q[19] = (1 - z) * ρ2 * λ
    else:
        ρ = ρ1
        Q[0] = ρ

    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, MP)
    Q[2:5] = ρ * v

    if MP.VISCOUS:
        Q[5:14] = A.ravel()

    if MP.THERMAL:
        Q[14:17] = ρ * J

    return Q
