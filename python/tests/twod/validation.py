from itertools import product

import numpy as np
from numpy import array, cos, exp, eye, sin, sqrt, tanh, zeros

from options import SYS, nx, ny, nz, dx, dy, Lx, Ly
from auxiliary.classes import material_parameters
from gpr.variables.vectors import conserved


def vortex(x, y, x0, y0, ε, γ, ρ, p):
    r2 = (x-x0)**2 + (y-y0)**2
    dv = ε/(2*np.pi) * exp((1-r2)/2) * array([-(y-y0), x-x0, 0])
    dT = -(γ-1) * ε**2 / (8 * γ * np.pi**2) * exp(1-r2)
    dρ = (1+dT)**(1/(γ-1)) - 1
    dp = (1+dT)**(γ/(γ-1)) - 1
    A = (ρ+dρ)**(1/3) * eye(3)
    return dv, dT, dρ, dp, A

def convected_isentropic_vortex_IC():
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
        x = (i+0.5)*dx
        y = (j+0.5)*dy
        dv, dT, dρ, dp, A = vortex(x, y, 5, 5, ε, γ, ρ, p)
        u[i,j,k] = conserved(ρ+dρ, p+dp, v+dv, A, J, 0, PAR, SYS)

    return u, [PAR]*1, []

def convected_isentropic_vortex_exact(t=1):
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
        x = (i+0.5)*dx
        y = (j+0.5)*dy
        dv, dT, dρ, dp, A = vortex(x, y, 5+t, 5+t, ε, γ, ρ, p)
        u[i,j,k] = conserved(ρ+dρ, p+dp, v+dv, A, J, 0, PAR, SYS)

    return u, [PAR], []


def circular_explosion_IC():
    """ Lx = 2
        Ly = 2
    """
    R = 0.5 * Lx
    PAR = material_parameters(γ=1.4, pINF=0, cv=2.5, ρ0=1, p0=1, cs=0.5, α=0.5, μ=1e-4, κ=1e-4)
    v = zeros([3])
    J = zeros([3])

    ρi = 1
    pi = 1
    Ai = eye(3)
    Qi = conserved(ρi, pi, v, Ai, J, 0, PAR, SYS)

    ρo = 0.125
    po = 0.1
    Ao = 0.5 * eye(3)
    Qo = conserved(ρo, po, v, Ao, J, 0, PAR, SYS)

    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        x = -Lx/2 + (i+0.5)*dx
        y = -Ly/2 + (j+0.5)*dy
        r = sqrt(x**2 + y**2)
        if r < R:
            u[i,j,k] = Qi
        else:
            u[i,j,k] = Qo

    return u, [PAR], []

def laminar_boundary_layer_IC():
    """ Lx = 1.5
        Ly = 0.4
    """
    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = array([1, 0, 0])
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=8, α=0, μ=1e-3, κ=0)

    u = zeros([nx, ny, nz, 18])
    Q = conserved(ρ, p, v, A, J, 0, PAR, SYS)
    for i,j,k in product(range(nx), range(ny), range(nz)):
        u[i,j,k] = Q

    return u, [PAR], []

def hagen_poiseuille_duct_IC():
    """ Lx = 10
        Ly = 0.5
    """
    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=8, α=0, μ=1e-2, κ=0)
    dp = 4.8

    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        pi = p - i/nx*dp
        u[i,j,k] = conserved(ρ, pi, v, A, J, 0, PAR, SYS)

    return u, [PAR], []

def lid_driven_cavity_IC():
    """ Lx = 1
        Ly = 1
    """
    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=8, α=0, μ=1e-2, κ=0)

    u = zeros([nx, ny, nz, 18])
    Q = conserved(ρ, p, v, A, J, 0, PAR, SYS)
    for i,j,k in product(range(nx), range(ny), range(nz)):
        u[i,j,k] = Q

    return u, [PAR], []

def double_shear_layer_IC():
    """ Lx = 1
        Ly = 1
    """
    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=8, α=0, μ=2e-4, κ=0)
    ρ_ = 30
    δ = 0.05

    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        x = (i+0.5) * dx
        y = (j+0.5) * dy
        if y > 0.75:
            v1 = tanh(ρ_ * (0.75-y))
        else:
            v1 = tanh(ρ_ * (y-0.25))
        v2 = δ * sin(2*np.pi*x)
        v = array([v1,v2,0])
        u[i,j,k] = conserved(ρ, p, v, A, J, 0, PAR, SYS)

    return u, [PAR], []

def compressible_mixing_layer_IC():
    pass

def taylor_green_vortex_IC():
    """ Lx = 2π
        Ly = 2π
    """
    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=10, α=0, μ=1e-2, κ=0)

    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        x = (i+0.5) * dx
        y = (j+0.5) * dy
        v = array([sin(x)*cos(y), -cos(x)*sin(y), 0])
        pi = p + (cos(2*x) + cos(2*y)) / 4
        u[i,j,k] = conserved(ρ, pi, v, A, J, 0, PAR, SYS)

    return u, [PAR], []
