from numpy import amax, arange, sign
from numpy.linalg import det
from numpy.random import rand

from models.gpr.misc.structures import Cvec
from models.gpr.variables.eos import total_energy


def generate_vector(MP):
    A = rand(3, 3)
    A *= sign(det(A))
    ρ = det(A) * MP.ρ0
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, MP)
    return Cvec(ρ, p, v, A, J, MP)


def check(x1, x2):
    if amax(abs(x1 - x2)) < 1e-12:
        return "✓"
    else:
        return amax(abs(x1 - x2))
