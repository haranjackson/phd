from numpy import array, zeros

from tests_1d.common import MP_AIR
from tests_1d.fluids import fluids_IC


def toro1_IC():

    tf = 0.25

    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    return fluids_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def toro2_IC():

    tf = 0.15

    ρL = 1
    pL = 0.4
    vL = array([-2, 0, 0])

    ρR = 1
    pR = 0.4
    vR = array([2, 0, 0])

    return fluids_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def toro3_IC():

    tf = 0.012

    ρL = 1
    pL = 1000
    vL = zeros(3)

    ρR = 1
    pR = 0.01
    vR = zeros(3)

    return fluids_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def toro4_IC():

    tf = 0.035

    ρL = 1
    pL = 0.01
    vL = zeros(3)

    ρR = 1
    pR = 100
    vR = zeros(3)

    return fluids_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def toro_test5_IC():

    tf = 0.035

    ρL = 5.99924
    pL = 460.894
    vL = array([19.5975, 0, 0])

    ρR = 5.99242
    pR = 46.095
    vR = array([-6.19633, 0, 0])

    return fluids_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)
