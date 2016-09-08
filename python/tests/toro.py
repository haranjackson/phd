from numpy import array, eye, zeros

from auxiliary.classes import material_parameters
from gpr.functions import conserved
from options import nx, ny, nz, L


γ = 1.4
pINF = 0

def toro_test1_IC():
    ρL = 1
    pL = 1
    vL = zeros(3)
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, vL, AL, JL, 0, γ, pINF, 1, 1, 0)
    QR = conserved(ρR, pR, vR, AR, JR, 0, γ, pINF, 1, 1, 0)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [material_parameters()]*2, [L/2]

def toro_test2_IC():
    ρL = 1
    vL = array([-2, 0, 0])
    pL = 0.4
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)

    ρR = 1
    vR = array([2, 0, 0])
    pR = 0.4
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, vL, AL, JL, 0, γ, pINF)
    QR = conserved(ρR, pR, vR, AR, JR, 0, γ, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test3_IC():
    ρL = 1
    vL = zeros(3)
    pL = 1000
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)

    ρR = 1
    vR = zeros(3)
    pR = 0.01
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, vL, AL, JL, 0, γ, pINF)
    QR = conserved(ρR, pR, vR, AR, JR, 0, γ, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test4_IC():
    ρL = 1
    vL = zeros(3)
    pL = 0.01
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)

    ρR = 1
    vR = zeros(3)
    pR = 100
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, vL, AL, JL, 0, γ, pINF)
    QR = conserved(ρR, pR, vR, AR, JR, 0, γ, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u

def toro_test5_IC():
    ρL = 5.99924
    vL = array([19.5975, 0, 0])
    pL = 460.894
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)

    ρR = 5.99242
    vR = array([-6.19633, 0, 0])
    pR = 46.095
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, vL, AL, JL, 0, γ, pINF)
    QR = conserved(ρR, pR, vR, AR, JR, 0, γ, pINF)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u
