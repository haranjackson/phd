from numpy import array, eye, sqrt, zeros

from gpr.misc.structures import Cvec
from gpr.opts import NV
from gpr.tests.one.common import fluids_IC
from gpr.tests.one.params import MP_Air_ND, MP_Air, MP_He, MP_H20


def sod_shock_IC():

    tf = 0.2
    nx = 100
    Lx = 1

    dX = [Lx / nx]

    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_Air_ND, MP_Air_ND,
                     0.5)


def water_gas_IC():

    tf = 2.3744e-4
    nx = 200
    Lx = 1

    dX = [Lx / nx]

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_H20, MP_Air, 0.7)


def water_water_IC():

    tf = 1.5e-4
    nx = 200
    Lx = 1

    dX = [Lx / nx]

    ρL = 1000
    pL = 7e8
    vL = zeros(3)

    ρR = 1000
    pR = pL / 7000
    vR = zeros(3)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_H20)


def helium_bubble_IC():

    tf = 14e-4
    nx = 200
    Lx = 1

    dX = [Lx / nx]
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

    u = zeros([nx, NV])
    Q1 = Cvec(ρL, pL, vL, AL, J, MP_Air)
    Q2 = Cvec(ρM, pM, vM, AM, J, MP_Air)
    Q3 = Cvec(ρR, pR, vR, AR, J, MP_He)

    for i in range(nx):

        if i * dx < 0.05:
            u[i] = Q1
        elif i * dx < 0.4:
            u[i] = Q2
        elif i * dx < 0.6:
            u[i] = Q3
        else:
            u[i] = Q2

        # u[i, -3] = i * dx - 0.05
        u[i, -2] = i * dx - 0.4
        u[i, -1] = i * dx - 0.6

    return u, [MP_Air, MP_Air, MP_He, MP_Air], tf, dX


def heat_conduction_IC():

    tf = 1
    nx = 200
    Lx = 1

    ρL = 2
    pL = 1
    vL = zeros(3)

    ρR = 0.5
    pR = 1
    vR = zeros(3)

    dX = [Lx / nx]

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP_Air_ND, MP_Air_ND)
