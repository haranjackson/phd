from itertools import product

from numpy import array, exp, eye, pi, zeros

from options import SYS, nx, ny, nz, dx, dy
from auxiliary.classes import material_parameters
from gpr.variables.vectors import conserved


def convected_isentropic_vortex():
    """ tf = 1
        Lx = 10
        Ly = 10
    """
    ε = 5

    μ = 1e-6
    κ = 1e-6
    γ = 1.4

    ρ = 1
    p = 1
    v = array([1, 1, 0])
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=2.5, ρ0=ρ, p0=p, cs=0.5, α=1, μ=μ, κ=κ)

    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):

        x = i*dx
        y = i*dy
        r2 = (x-5)**2 + (y-5)**2

        dv = ε/(2*pi) * exp((1-r2)/2) * array([5-y, x-5, 0])
        dT = -(γ-1) * ε**2 / (8 * γ * pi**2) * exp(1-r2)
        dρ = (1+dT)**(1/(γ-1)) - 1
        dp = (1+dT)**(γ/(γ-1)) - 1
        A = (ρ+dρ)**(1/3) * eye(3)

        u[i,j,k] = conserved(ρ+dρ, p+dp, v+dv, A, J, 0, PAR, SYS)

    return u, [PAR]*1, []