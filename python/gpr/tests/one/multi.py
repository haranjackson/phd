from numpy import array, eye, sqrt, zeros

from gpr.misc.structures import Cvec
from gpr.vars.hyp import Cvec_hyp
from gpr.opts import NV

from gpr.tests.params import MP_Air_ND, MP_Air, MP_He, MP_H20, MP_Cu_SMG, HYP_Cu
from gpr.tests.one.common import riemann_IC, fluids_IC
from gpr.tests.one.fluids import heat_conduction_IC
from gpr.tests.one.solids import barton1_IC
from gpr.opts import VISCOUS, THERMAL, REACTIVE, MULTI, LSET


def sod_shock_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 0.2
    nx = 100
    Lx = 1
    MPs = [MP_Air_ND, MP_Air_ND]

    dX = [Lx / nx]

    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    u = fluids_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def water_gas_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 2.3744e-4
    nx = 200
    Lx = 1
    MPs = [MP_H20, MP_Air]

    dX = [Lx / nx]

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    u = fluids_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs, x0=0.7)
    return u, MPs, tf, dX


def water_water_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 1.5e-4
    nx = 200
    Lx = 1
    MPs = [MP_H20, MP_H20]

    dX = [Lx / nx]

    ρL = 1000
    pL = 7e8
    vL = zeros(3)

    ρR = 1000
    pR = pL / 7000
    vR = zeros(3)

    u = fluids_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def helium_bubble_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 2)

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

    return u, [MP_Air, MP_He, MP_Air], tf, dX


def gas_solid_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 0.05
    nx = 250
    Lx = 1
    MPs = [MP_Air, MP_Cu_SMG]

    dX = [Lx / nx]

    vL = zeros(3)
    ρL = 1.18
    pL = 18.9  # pressure is in Km^2 s^-2 g cm^-3 = 10^9 Kg m^-1 s^-2
    AL = eye(3)
    QL = Cvec(ρL, pL, vL, AL, zeros(3), MP_Air)

    vR = zeros(3)
    FR = eye(3)
    SR = 0
    QR = Cvec_hyp(FR, SR, vR, HYP_Cu)

    u = riemann_IC(nx, dX, QL, QR, 0.5, True)
    return u, MPs, tf, dX


def heat_conduction_multi_IC():

    assert(VISCOUS)
    assert(THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    return heat_conduction_IC(isMulti=True)


def barton1_multi_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    return barton1_IC(isMulti=True)
