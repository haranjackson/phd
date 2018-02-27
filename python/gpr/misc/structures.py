from numpy import zeros

from gpr.misc.functions import gram
from gpr.variables.derivatives import dEdρ, dEdp, dEdA, dEdA_s, dEdJ, dTdρ, dTdp
from gpr.variables.eos import total_energy
from gpr.variables.sources import theta1inv, theta2inv, K_arr, K_dis, K_ing
from gpr.variables.state import heat_flux, pressure, temperature
from gpr.variables.state import sigma, dsigmadρ, dsigmadA, Sigma

from options import NV


class Cvec_to_Pclass():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """

    def __init__(self, Q, MP):

        if MP.MULTI:
            self.ρ = Q[0] + Q[17]
            self.z = Q[18] / self.ρ
            self.ρ1 = Q[0] / self.z
            self.ρ2 = Q[17] / self.z
            if MP.REACTIVE:
                self.λ = Q[19] / Q[17]
        else:
            self.ρ = Q[0]
            self.ρ1 = self.ρ
            self.z = 1

        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ

        if MP.VISCOUS:
            self.A = Q[5:14].reshape([3, 3])

        if MP.THERMAL:
            self.J = Q[14:17] / self.ρ
        else:
            self.J = zeros(3)

        self.MP = MP

    def G(self):
        return gram(self.A)

    def p(self):
        if hasattr(self, 'p_'):
            return self.p_
        else:
            if self.MP.REACTIVE:
                self.p_ = pressure(self.ρ, self.E, self.v, self.A, self.J,
                                   self.MP, self.λ)
            else:
                self.p_ = pressure(self.ρ, self.E, self.v, self.A, self.J,
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


def Cvec(ρ1, p, v, A, J, MP, ρ2=None, z=1, λ=None):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(NV)

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


def Pvec(P):
    ret = zeros(NV)
    ret[0] = P.ρ
    ret[1] = P.p()
    ret[2:5] = P.v
    ret[5:14] = P.A.ravel()
    ret[14:17] = P.J
    return ret


def Pvec_to_Cvec(P, MP):
    """ Returns the vector of conserved variables, given the vector of
        primitive variables
    """
    Q = P.copy()
    ρ = P[0]
    A = P[5:14].reshape([3, 3])

    if MP.REACTIVE:
        λ = P[17]
    else:
        λ = 0

    Q[1] = ρ * total_energy(ρ, P[1], P[2:5], A, P[14:17], λ, MP)
    Q[2:5] *= ρ
    Q[14:] *= ρ
    return Q


def Cvec_to_Pvec(Q, MP):
    """ Returns the vector of primitive variables in standard ordering,
        given the vector of conserved variables.
    """
    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ

    if MP.REACTIVE:
        λ = Q[17] / ρ
    else:
        λ = None

    p = pressure(ρ, E, v, A, J, MP, λ)

    ret = Q.copy()
    ret[1] = p
    ret[2:5] /= ρ
    ret[14:] /= ρ

    return ret


def Cgrid_to_Pgrid(u, MP):
    nx, ny, nz = u.shape[:3]
    ret = zeros(u.shape)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ret[i, j, k] = Cvec_to_Pvec(u[i, j, k], MP)
    return ret
