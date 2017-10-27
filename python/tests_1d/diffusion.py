from itertools import product

from numpy import array, eye, zeros

from auxiliary.boundaries import standard_BC
from system.gpr.misc.objects import material_parameters
from system.gpr.misc.structures import Cvec
from options import nx, ny, nz


midx = int(nx/2)
midy = int(ny/2)


def defaults():
    γ = 1.4
    μ = 1e-2 # 1e-3 # 1e-4

    ρ = 1
    p = 1 / γ
    v = array([0, -0.1, 0])
    A = eye(3)
    J = zeros(3)
    λ = 0

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=1, α=1e-16, μ=μ, Pr=0.75)
    Q1 = Cvec(ρ, p, v, A, J, λ, PAR)
    Q2 = Cvec(ρ, p, zeros(3), A, J, λ, PAR)
    return Q1, Q2, PAR

def barrier_IC():
    """ nx = 40
        ny = 60
    """
    Q1, Q2, PAR = defaults()
    u = zeros([nx, ny, nz, nV])
    for i, j in product(range(nx), range(ny)):
        u[i,j,0] = Q1
    for i, j in product(range(midx), range(midy)):
        u[i,j,0] = Q2

    return u, [PAR], []

def barrier_BC(u):

    _, Q2, PAR = defaults()
    for i, j in product(range(midx), range(midy)):
        u[i,j,0] = Q2

    for i in range(midx):
        Q = u[i, midy, 0]
        Q[2] = 0
        Q[3] *= -1
        u[i, midy-1, 0] = Q
    for j in range(midy):
        Q = u[midx, j, 0]
        Q[2] *= -1
        Q[3] = 0
        u[midx-1, j, 0] = Q

    u[midx-1, midy-1, 0, 2:4] = 0
    return standard_BC(u)
