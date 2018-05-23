from numpy import array, zeros

from gpr.tests.params import Air_SG_SI, H20_SG_SI
from gpr.tests.one.common import primitive_IC


def water_air_IC():

    tf = 2.3744e-4
    ny = 200
    Ly = 1

    nx = 5
    Lx = nx / ny * Ly
    MPs = [H20_SG_SI, Air_SG_SI]

    dX = [Lx / nx, Ly / ny]

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    u = primitive_IC(ny, dX[1:], ρL, pL, vL, ρR, pR, vR, MPs, x0=0.7)
    u = array([u] * nx)
    return u, MPs, tf, dX
