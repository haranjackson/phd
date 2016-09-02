from numpy import array, eye, zeros

from auxiliary.common import material_parameters
from gpr.functions import conserved
from options import nx, ny, nz, L, y, pINF


def toro_test1_IC():
    rL = 1
    pL = 1
    vL = zeros(3)
    AL = rL**(1/3) * eye(3)
    JL = zeros(3)

    rR = 0.125
    pR = 0.1
    vR = zeros(3)
    AR = rR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, y, pINF, 1, 1, 0)
    QR = conserved(rR, pR, vR, AR, JR, 0, y, pINF, 1, 1, 0)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [material_parameters()]*2, [L/2]

def toro_test2_IC():
    rL = 1
    vL = array([-2, 0, 0])
    pL = 0.4
    AL = rL**(1/3) * eye(3)
    JL = zeros(3)

    rR = 1
    vR = array([2, 0, 0])
    pR = 0.4
    AR = rR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, y, pINF)
    QR = conserved(rR, pR, vR, AR, JR, 0, y, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test3_IC():
    rL = 1
    vL = zeros(3)
    pL = 1000
    AL = rL**(1/3) * eye(3)
    JL = zeros(3)

    rR = 1
    vR = zeros(3)
    pR = 0.01
    AR = rR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, y, pINF)
    QR = conserved(rR, pR, vR, AR, JR, 0, y, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test4_IC():
    rL = 1
    vL = zeros(3)
    pL = 0.01
    AL = rL**(1/3) * eye(3)
    JL = zeros(3)

    rR = 1
    vR = zeros(3)
    pR = 100
    AR = rR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, y, pINF)
    QR = conserved(rR, pR, vR, AR, JR, 0, y, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test5_IC():
    rL = 5.99924
    vL = array([19.5975, 0, 0])
    pL = 460.894
    AL = rL**(1/3) * eye(3)
    JL = zeros(3)

    rR = 5.99242
    vR = array([-6.19633, 0, 0])
    pR = 46.095
    AR = rR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, y, pINF)
    QR = conserved(rR, pR, vR, AR, JR, 0, y, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u
