from numpy import abs, amax, arange, sign, zeros
from numpy.linalg import det
from numpy.random import rand

from gpr.misc.structures import Cvec
from gpr.vars.eos import total_energy


def generate_vector(MP):
    A = rand(3, 3)
    A *= sign(det(A))
    ρ = det(A) * MP.ρ0
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, MP)
    return Cvec(ρ, p, v, A, J, MP)


def relative_diff(x1, x2):
    return amax(abs(x1 - x2) / (1 + abs(x1)))


def check(x1, x2):
    relDiff = max(relative_diff(x1, x2), relative_diff(x2, x1))
    if relDiff < 1e-12:
        return "✓"
    else:
        return "abs:", amax(abs(x1 - x2)), "rel:", relDiff
