from numpy import array, cos, exp, eye, pi, sin, sqrt, tanh, zeros

from etc.boundaries import extend
from gpr.misc.objects import material_parameters
from gpr.misc.structures import Cvec, Cvec_to_Pclass
from tests_1d.common import cell_sizes
from options import NV, N


def vortex(x, y, x0, y0, ε, γ, ρ):

    r2 = (x - x0)**2 + (y - y0)**2
    dv = ε / (2 * pi) * exp((1 - r2) / 2) * array([-(y - y0), x - x0, 0])
    dT = -(γ - 1) * ε**2 / (8 * γ * pi**2) * exp(1 - r2)

    dρ = (1 + dT)**(1 / (γ - 1)) - 1
    dp = (1 + dT)**(γ / (γ - 1)) - 1
    A = (ρ + dρ)**(1 / 3) * eye(3)
    return dv, dT, dρ, dp, A


def convected_isentropic_vortex_IC(μ=1e-6, κ=1e-6, t=0):

    tf = 1
    Lx = 10
    Ly = 10
    nx = 10
    ny = 10

    ε = 5
    γ = 1.4

    ρ = 1
    p = 1
    v = array([1, 1, 0])
    J = zeros(3)

    dX = cell_sizes(Lx, nx, Ly, ny)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=2.5, p0=p, γ=γ,
                             b0=0.5, cα=1, μ=μ, κ=κ)

    u = zeros([nx, ny, 1, NV])
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dX[0]
            y = (j + 0.5) * dX[1]
            dv, dT, dρ, dp, A = vortex(x, y, Lx / 2 + t, Ly / 2 + t, ε, γ, ρ)
            u[i, j] = Cvec(ρ + dρ, p + dp, v + dv, A, J, MP)

    print("CONVECTED ISENTROPIC VORTEX: N =", N)
    return u, [MP], tf, dX


def circular_explosion_IC():

    tf = 0.2
    Lx = 2
    Ly = 2
    nx = 400
    ny = 400

    R = 0.25 * Lx
    MP = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4,
                             b0=0.5, cα=0.5, μ=1e-4, κ=1e-4)

    dX = cell_sizes(Lx, nx, Ly, ny)

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

    u = zeros([nx, ny, 1, NV])
    for i in range(nx):
        for j in range(ny):
            x = -Lx / 2 + (i + 0.5) * dX[0]
            y = -Ly / 2 + (j + 0.5) * dX[1]
            r = sqrt(x**2 + y**2)
            if r < R:
                u[i, j] = Qi
            else:
                u[i, j] = Qo

    print("CIRCULAR EXPLOSION: N =", N)
    return u, [MP], tf, dX


def laminar_boundary_layer_IC():

    tf = 10
    Lx = 1.5
    Ly = 0.4
    nx = 75
    ny = 100

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = array([1, 0, 0])
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-3)

    u = zeros([nx, ny, 1, NV])
    Q = Cvec(ρ, p, v, A, J, MP)
    for i in range(nx):
        for j in range(ny):
            u[i, j] = Q

    print("LAMINAR BOUNDARY LAYER: N =", N)
    return u, [MP], tf, cell_sizes(Lx, nx, Ly, ny)


def hagen_poiseuille_duct_IC():

    tf = 10
    Lx = 10
    Ly = 0.5
    nx = 10
    ny = 50

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)
    dp = 4.8

    u = zeros([nx, ny, 1, NV])
    for i in range(nx):
        for j in range(ny):
            pi = p - (i+1) / (nx+2) * dp
            u[i, j] = Cvec(ρ, pi, v, A, J, MP)

    print("HAGEN-POISEUILLE DUCT: N =", N)
    return u, [MP], tf, cell_sizes(Lx, nx, Ly, ny)


def hagen_poiseuille_duct_BC(u):

    γ = 1.4
    ρ = 1
    p = 100 / γ
    dp = 4.8
    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)

    ret = extend(u, 1, 1)
    ret[:, 0, 0, 2:5] *= -1
    ret[:, -1, 0, 2:5] *= -1

    ret = extend(ret, 1, 0)
    ny = ret.shape[1]
    for j in range(ny):
        QL = ret[0, j, 0]
        QR = ret[-1, j, 0]
        PL = Cvec_to_Pclass(QL, MP)
        PR = Cvec_to_Pclass(QR, MP)
        ret[0, j, 0] = Cvec(PL.ρ,  p, PL.v, PL.A, PL.J, MP)
        ret[-1, j, 0] = Cvec(PR.ρ,  p - dp, PR.v, PR.A, PR.J, MP)

    return ret


def lid_driven_cavity_IC():

    tf = 10
    Lx = 1
    Ly = 1
    nx = 100
    ny = 100

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)

    u = zeros([nx, ny, 1, NV])
    Q = Cvec(ρ, p, v, A, J, MP)
    for i in range(nx):
        for j in range(ny):
            u[i, j] = Q

    print("LID-DRIVEN CAVITY: N =", N)
    return u, [MP], tf, cell_sizes(Lx, nx, Ly, ny)


def lid_driven_cavity_BC(u):

    ret = extend(u, 1, 1)
    nx, ny = ret.shape[:2]

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

    tf = 1.8
    Lx = 1
    Ly = 1
    nx = 200
    ny = 200

    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=2e-4)
    dX = cell_sizes(Lx, nx, Ly, ny)

    ρ_ = 30
    δ = 0.05

    u = zeros([nx, ny, 1, NV])
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dX[0]
            y = (j + 0.5) * dX[1]
            if y > 0.75:
                v1 = tanh(ρ_ * (0.75 - y))
            else:
                v1 = tanh(ρ_ * (y - 0.25))
            v2 = δ * sin(2 * pi * x)
            v = array([v1, v2, 0])
            u[i, j] = Cvec(ρ, p, v, A, J, MP)

    print("DOUBLE SHEAR LAYER: N =", N)
    return u, [MP], tf, dX


def compressible_mixing_layer_IC():
    pass


def taylor_green_vortex_IC():

    tf = 10
    Lx = 2 * pi
    Ly = 2 * pi
    nx = 50
    ny = 50

    γ = 1.4

    ρ = 1
    p = 100 / γ
    A = eye(3)
    J = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=10, μ=1e-2)
    dX = cell_sizes(Lx, nx, Ly, ny)

    u = zeros([nx, ny, 1, NV])

    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dX[0]
            y = (j + 0.5) * dX[1]
            v = array([sin(x) * cos(y), -cos(x) * sin(y), 0])
            pi = p + (cos(2 * x) + cos(2 * y)) / 4
            u[i, j] = Cvec(ρ, pi, v, A, J, MP)

    print("TAYLOR-GREEN VORTEX: N =", N)
    return u, [MP], tf, dX
