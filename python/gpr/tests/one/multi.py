from numpy import array, eye, sqrt, zeros

from gpr.misc.structures import Cvec
from gpr.tests.one.common import MP_AIR, MP_AIR2, MP_HEL2, MP_WAT2
from gpr.tests.one.common import cell_sizes
from gpr.tests.one.fluids import fluids_IC


def sod_shock_IC():

    tf = 0.2
    nx = 200
    Lx = 1

    dX = cell_sizes(Lx, nx)

    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def water_gas_IC():

    tf = 237.44e-6
    nx = 200
    Lx = 1

    dX = cell_sizes(Lx, nx)

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_AIR2, MP_WAT2, 0.7)


def water_water_IC():

    tf = 1.5e-4
    nx = 200
    Lx = 1

    dX = cell_sizes(Lx, nx)

    ρL = 1000
    pL = 7e8
    vL = zeros(3)

    ρR = 1000
    pR = pL / 7000
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_WAT2)


def helium_bubble_IC():

    tf = 0.0014
    nx = 200
    Lx = 1

    dX = cell_sizes(Lx, nx)
    dx = dX[0]

    ρL = 1.3333
    pL = 1.5e5
    vL = array([35.35 * sqrt(10), 0, 0])
    AL = ρL**(1 / 3) * eye(3)

    ρM = 1
    pM = 1e5
    vM = zeros(3)
    AM = ρM**(1 / 3) * eye(3)

    ρR = 0.1379
    pR = 1e5
    vR = zeros(3)
    AR = ρR**(1 / 3) * eye(3)

    J = zeros(3)

    u = zeros([nx, 17])
    Q1 = Cvec(ρL, pL, vL, AL, J, MP_AIR2)
    Q2 = Cvec(ρM, pM, vM, AM, J, MP_AIR2)
    Q3 = Cvec(ρR, pR, vR, AR, J, MP_HEL2)

    for i in range(nx):

        u[i, 0, 0, -3] = i * dx - 0.05
        u[i, 0, 0, -2] = i * dx - 0.4
        u[i, 0, 0, -1] = i * dx - 0.6

        if i * dx < 0.05:
            u[i, 0, 0, :-3] = Q1
        elif i * dx < 0.4:
            u[i, 0, 0, :-3] = Q2
        elif i * dx < 0.6:
            u[i, 0, 0, :-3] = Q3
        else:
            u[i, 0, 0, :-3] = Q2

    return u, [MP_AIR2, MP_AIR2, MP_HEL2, MP_AIR2], tf, dX
