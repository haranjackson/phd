from numpy import eye, zeros

from ader.etc.boundaries import standard_BC

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec


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

    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2)

    u = zeros([nx, ny, 14])
    Q = Cvec(ρ, p, v, MP, A)
    for i in range(nx):
        for j in range(ny):
            u[i, j] = Q

    print("LID-DRIVEN CAVITY")
    return u, [MP], tf, [Lx / nx, Ly / ny]


def lid_driven_cavity_BC(u):

    ret = standard_BC(u, [1, 1])
    nx, ny = ret.shape[:2]

    for i in range(nx):
        v = 2 - ret[i, 1, 2] / ret[i, 1, 0]
        ret[i, 0, 2] = ret[i, 0, 0] * v
        ret[i, 0, 3] *= -1
        ret[i, -1, 2:5] *= -1

    for j in range(ny):
        ret[0, j, 2:5] *= -1
        ret[-1, j, 2:5] *= -1

    ret[0, 0, 2:5] *= 0
    ret[0, -1, 2:5] *= 0
    ret[-1, 0, 2:5] *= 0
    ret[-1, -1, 2:5] *= 0

    return ret