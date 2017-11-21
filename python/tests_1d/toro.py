from numpy import array, eye, zeros

from system.gpr.misc.objects import material_parameters
from system.gpr.misc.structures import Cvec
from options import nx, ny, nz, nV


PAR = material_parameters(EOS='sg', ρ0=1, cv=1, γ=1.4, p0=1,
                          cs=1, α=1, μ=1e-2, Pr=0.75)


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

    u = zeros([nx, ny, nz, nV])
    QL = Cvec(ρL, pL, vL, AL, JL, 0, PAR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PAR)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [PAR]*2, [0.5]

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

    u = zeros([nx, ny, nz, nV])
    QL = Cvec(ρL, pL, vL, AL, JL, 0, PAR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PAR)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [PAR]*1, []

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

    u = zeros([nx, ny, nz, nV])
    QL = Cvec(ρL, pL, vL, AL, JL, 0, PAR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PAR)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [PAR]*1, []

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

    u = zeros([nx, ny, nz, nV])
    QL = Cvec(ρL, pL, vL, AL, JL, 0, PAR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PAR)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [PAR]*1, []

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

    u = zeros([nx, ny, nz, nV])
    QL = Cvec(ρL, pL, vL, AL, JL, 0, PAR)
    QR = Cvec(ρR, pR, vR, AR, JR, 0, PAR)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [PAR]*1, []
