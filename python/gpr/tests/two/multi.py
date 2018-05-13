from numpy import array, eye, zeros

from gpr.misc.structures import Cvec
from gpr.vars.hyp import Cvec_hyp

from gpr.tests.params import MP_Air, MP_H20, MP_Cu_SMG, HYP_Cu
from gpr.tests.one.common import riemann_IC, fluids_IC
from gpr.opts import VISCOUS, THERMAL, REACTIVE, MULTI, LSET


def water_gas_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 2.3744e-4
    ny = 200
    Ly = 1

    nx = 5
    Lx = nx / ny * Ly
    MPs = [MP_H20, MP_Air]

    dX = [Lx / nx, Ly / ny]

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    u = fluids_IC(ny, dX[1:], ρL, pL, vL, ρR, pR, vR, MPs, x0=0.7)
    u = array([u] * nx)
    return u, MPs, tf, dX


def gas_solid_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 1)

    tf = 0.05
    ny = 250
    Ly = 1

    nx = 5
    Lx = nx / ny * Ly
    MPs = [MP_Air, MP_Cu_SMG]

    dX = [Lx / nx, Ly / ny]

    vL = zeros(3)
    ρL = 1.18
    pL = 18.9  # pressure is in Km^2 s^-2 g cm^-3 = 10^9 Kg m^-1 s^-2
    AL = eye(3)
    QL = Cvec(ρL, pL, vL, AL, zeros(3), MP_Air)

    vR = zeros(3)
    FR = eye(3)
    SR = 0
    QR = Cvec_hyp(FR, SR, vR, HYP_Cu)

    u = riemann_IC(ny, dX[1:], QL, QR, 0.5, True)
    u = array([u] * nx)
    return u, MPs, tf, dX
