from numpy import amax, arange, sign, zeros
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


def check(x1, x2):
    diff = x1-x2
    absDiff=amax(abs(diff))
    diff[abs(diff)<1e-11] = 0
    relDiff1 = amax(abs(diff[x1!=0]/x1[x1!=0]))
    relDiff2 = amax(abs(diff[x2!=0]/x2[x2!=0]))
    relDiff = max(relDiff1, relDiff2)
    if relDiff < 1e-12:
        return "✓"
    else:
        return "abs:", absDiff, "rel:", relDiff


def cpp_dx(dX):

    ret = zeros(3)
    for i in range(len(dX)):
        ret[i] = dX[i]
    return ret
