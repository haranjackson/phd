from numpy import zeros

from gpr.variables.eos import total_energy
from gpr.variables.state import pressure, temperature


class primitive():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """
    def __init__(self, Q, params, subsystems):
        self.ρ = Q[0]
        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3,3])
        self.J = Q[14:17] / self.ρ
        self.λ = Q[17] / self.ρ

        self.p = pressure(self.E, self.v, self.A, self.ρ, self.J, self.λ, params, subsystems)
        self.T = temperature(self.ρ, self.p, params.γ, params.pINF, params.cv)

def conserved(ρ, p, v, A, J, λ, params, subsystems):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(18)
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, params.γ, params.pINF, params.cs2, params.α2,
                            params.Qc, subsystems.viscous, subsystems.thermal, subsystems.reactive)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    Q[14:17] = ρ * J
    Q[17] = ρ * λ
    return Q

def primitive_vector(P):
    ret = zeros(18)
    ret[0] = P.ρ
    ret[1] = P.p
    ret[2:11] = P.A.ravel()
    ret[11:14] = P.v
    ret[14:17] = P.J
    return ret

def primitive_to_conserved(P, params, subsystems):
    Q = zeros(18)
    ρ = P[0]
    p = P[1]
    A = P[2:11].reshape([3,3])
    v = P[11:14]
    J = P[14:17]
    λ = 0
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, λ, params.γ, params.pINF, params.cs2, params.α2,
                            params.Qc, subsystems.viscous, subsystems.thermal, subsystems.reactive)
    Q[2:5] = ρ * v
    Q[5:14] = P[2:11]
    Q[14:17] = ρ * J
    return Q
