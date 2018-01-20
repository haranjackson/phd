from itertools import product

import numpy as np
from numpy import array, cos, exp, eye, sin, sqrt, tanh, zeros

from gpr.misc.objects import material_parameters
from gpr.misc.structures import Cvec
from solvers.weno.weno import extend
from options import nx, ny, nz, nV, dx, dy, Lx, Ly


def vortex(x, y, x0, y0, ε, γ, ρ, p):
    r2 = (x - x0)**2 + (y - y0)**2
    dv = ε / (2 * np.pi) * exp((1 - r2) / 2) * array([-(y - y0), x - x0, 0])
    dT = -(γ - 1) * ε**2 / (8 * γ * np.pi**2) * exp(1 - r2)
    dρ = (1 + dT)**(1 / (γ - 1)) - 1
    dp = (1 + dT)**(γ / (γ - 1)) - 1
    A = (ρ + dρ)**(1 / 3) * eye(3)
    return dv, dT, dρ, dp, A


def convected_isentropic_vortex_IC(μ=1e-6, κ=1e-6, t=0):

    tf = 1

    ε = 5
    γ = 1.4

    ρ = 1
    p = 1
    v = array([1, 1, 0])
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=2.5, p0=p, γ=γ,
                             b0=0.5, cα=1, μ=μ, κ=κ)

    u = zeros([nx, ny, nz, nV])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        x = (i + 0.5) * dx
        y = (j + 0.5) * dy
        dv, dT, dρ, dp, A = vortex(x, y, 5 + t, 5 + t, ε, γ, ρ, p)
        u[i, j, k] = Cvec(ρ + dρ, p + dp, v + dv, A, J, MP)

    return u, [MP], tf


def circular_explosion_IC():
    """ Lx = 2
        Ly = 2
        nx = 400
        ny = 400
        N = 2
    """

    tf = 0.2

    R = 0.25 * Lx
    MP = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4,
                             b0=0.5, cα=0.5, μ=1e-4, κ=1e-4)
    v = zeros([3])
    J = zeros([3])

    ρi = 1
    pi = 1
    Ai = eye(3)
    Qi = Cvec(ρi, pi, v, Ai, J, MP)

    ρo = 0.125
    po = 0.1
    Ao = 0.5 * eye(3)
    Qo = Cvec(ρo, po, v, Ao, J, MP)

    u = zeros([nx, ny, nz, nV])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        x = -Lx / 2 + (i + 0.5) * dx
        y = -Ly / 2 + (j + 0.5) * dy
        r = sqrt(x**2 + y**2)
        if r < R:
            u[i, j, k] = Qi
        else:
            u[i, j, k] = Qo

    return u, [MP], tf


def laminar_boundary_layer_IC():
    """ Lx = 1.5
        Ly = 0.4
        nx = 75
        ny = 100
        N = 2
    """
    tf = 10

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = array([1, 0, 0])
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-3)

    u = zeros([nx, ny, nz, nV])
    Q = Cvec(ρ, p, v, A, J, MP)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        u[i, j, k] = Q

    return u, [MP], tf


def hagen_poiseuille_duct_IC():
    """ Lx = 10
        Ly = 0.5
        nx = 100
        ny = 50
        N = 2
    """
    tf = 10

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)
    dp = 4.8

    u = zeros([nx, ny, nz, nV])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        pi = p - i / nx * dp
        u[i, j, k] = Cvec(ρ, pi, v, A, J, MP)

    return u, [MP], tf


def lid_driven_cavity_IC():
    """ Lx = 1
        Ly = 1
        nx = 100
        ny = 100
        N = 2
    """
    tf = 10

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)

    u = zeros([nx, ny, nz, nV])
    Q = Cvec(ρ, p, v, A, J, MP)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        u[i, j, k] = Q

    return u, [MP], tf


def lid_driven_cavity_BC(u):

    ret = extend(u, 1, 1)
    nx, ny, _, _ = ret.shape

    for i in range(nx):
        v = 2 - ret[i, 1, 0, 2] / ret[i, 1, 0, 0]
        ret[i, 0, 0, 2] = ret[i, 0, 0, 0] * v
        ret[i, 0, 0, 3] *= -1
        ret[i, -1, 0, 2:5] *= -1

    ret = extend(ret, 1, 0)
    for j in range(ny):
        ret[0, j, 0, 2:5] *= -1
        ret[-1, j, 0, 2:5] *= -1

    ret[0, 0, 0, 2:5] *= 0
    ret[0, -1, 0, 2:5] *= 0
    ret[-1, 0, 0, 2:5] *= 0
    ret[-1, -1, 0, 2:5] *= 0

    return ret


def double_shear_layer_IC():
    """ Lx = 1
        Ly = 1
        nx = 200
        ny = 200
        N = 3
    """
    tf = 1.8

    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=2e-4)
    ρ_ = 30
    δ = 0.05

    u = zeros([nx, ny, nz, nV])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        x = (i + 0.5) * dx
        y = (j + 0.5) * dy
        if y > 0.75:
            v1 = tanh(ρ_ * (0.75 - y))
        else:
            v1 = tanh(ρ_ * (y - 0.25))
        v2 = δ * sin(2 * np.pi * x)
        v = array([v1, v2, 0])
        u[i, j, k] = Cvec(ρ, p, v, A, J, MP)

    return u, [MP], tf


def compressible_mixing_layer_IC():
    pass


def taylor_green_vortex_IC():
    """ Lx = 2π
        Ly = 2π
        nx = 50
        ny = 50
        N = 3
    """
    tf = 10

    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=10, μ=1e-2)

    u = zeros([nx, ny, nz, nV])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        x = (i + 0.5) * dx
        y = (j + 0.5) * dy
        v = array([sin(x) * cos(y), -cos(x) * sin(y), 0])
        pi = p + (cos(2 * x) + cos(2 * y)) / 4
        u[i, j, k] = Cvec(ρ, pi, v, A, J, MP)

    return u, [MP], tf
