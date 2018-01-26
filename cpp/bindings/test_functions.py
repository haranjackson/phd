from numpy import amax, arange, sign
from numpy.linalg import det
from numpy.random import rand

from gpr.misc.structures import Cvec
from gpr.variables.eos import total_energy


def generate_vector(MP):
    A = rand(3, 3)
    A *= sign(det(A))
    ρ = det(A)
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, MP)
    return Cvec(ρ, p, v, A, J, MP)


def diff(x1, x2):
    return amax(abs(x1 - x2))
