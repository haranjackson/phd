from numpy import array, zeros

from gpr.tests.params import Air_SG_ND
from gpr.tests.one.common import primitive_IC


def toro1_IC():

    tf = 0.25
    nx = 200
    Lx = 1
    MPs = [Air_SG_ND]

    dX = [Lx / nx]

    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def toro2_IC():

    tf = 0.15
    nx = 200
    Lx = 1
    MPs = [Air_SG_ND]

    dX = [Lx / nx]

    ρL = 1
    pL = 0.4
    vL = array([-2, 0, 0])

    ρR = 1
    pR = 0.4
    vR = array([2, 0, 0])

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def toro3_IC():

    tf = 0.012
    nx = 200
    Lx = 1
    MPs = [Air_SG_ND]

    dX = [Lx / nx]

    ρL = 1
    pL = 1000
    vL = zeros(3)

    ρR = 1
    pR = 0.01
    vR = zeros(3)

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def toro4_IC():

    tf = 0.035
    nx = 200
    Lx = 1
    MPs = [Air_SG_ND]

    dX = [Lx / nx]

    ρL = 1
    pL = 0.01
    vL = zeros(3)

    ρR = 1
    pR = 100
    vR = zeros(3)

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def toro5_IC():

    tf = 0.035
    nx = 200
    Lx = 1
    MPs = [Air_SG_ND]

    dX = [Lx / nx]

    ρL = 5.99924
    pL = 460.894
    vL = array([19.5975, 0, 0])

    ρR = 5.99242
    pR = 46.095
    vR = array([-6.19633, 0, 0])

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX
