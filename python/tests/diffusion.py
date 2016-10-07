from itertools import product

from numpy import array, eye, zeros

from auxiliary.bc import standard_BC
from auxiliary.classes import material_parameters
from gpr.functions import conserved
from options import nx, ny, nz, subsystems


def barrier_IC():
    """ nx = 100
        ny = 40
    """
    γ = 1.4
    μ = 1e-2 # 1e-3 # 1e-4

    ρ = 1
    p = 1 / γ
    v = array([0, 0.1, 0])
    A = eye(3)
    J = zeros(3)
    λ = 0

    params = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=1, α=1e-16, μ=μ, Pr=0.75)

    Q = conserved(ρ, p, v, A, J, λ, params, subsystems)
    u = zeros([nx, ny, nz, 18])
    for i, j in product(range(nx), range(ny)):
        u[i,j,0] = Q

    return u, [params], []

def barrier_BC(u):
    midx = int(nx/2)
    midy = int(ny/2)

    for i in range(midx,nx):
        Q = u[i, midy, 0]
        Q[3] *= -1
        u[i, midy-1, 0] = Q
    for j in range(midy):
        Q = u[midx-1, j, 0]
        Q[2] *= -1
        u[midx, j, 0] = Q

    return standard_BC(u)
