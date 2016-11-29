from numpy import array, eye, sqrt, zeros

from auxiliary.classes import material_parameters
from gpr.variables.vectors import conserved
from options import nx, ny, nz, Lx, dx, SYS


def sod_shock_IC():
    """ tf = 0.2
        L = 1
    """
    ρL = 1
    pL = 1
    AL = ρL**(1/3) * eye(3)

    ρR = 0.125
    pR = 0.1
    AR = ρR**(1/3) * eye(3)

    v = zeros(3)
    J = zeros(3)
    λ = 0

    PAR = material_parameters(y=1.4, pINF=0, cv=2.5, ρ0=1, p0=1, cs=1, α=1, μ=5e-4, Pr=2/3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, v, AL, J, λ, PAR, SYS)
    QR = conserved(ρR, pR, v, AR, J, λ, PAR, SYS)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [PAR]*2, [Lx/2]

def water_gas_IC():
    """ tf = 237.44e-6
        L = 1
    """
    ρL = 1000
    pL = 1e9
    AL = ρL**(1/3) * eye(3)
    PARL = material_parameters(y=4.4, pINF=6e8, cv=950, ρ0=1000, p0=1, cs=1e-4, α=1e-4, μ=1e-3,
                               Pr=7)

    ρR = 50
    pR = 101325
    AR = ρR**(1/3) * eye(3)
    PARR = material_parameters(y=1.4, pINF=0, cv=718, ρ0=1.176, p0=101325, cs=55, α=5e2, μ=1.98e-5,
                               Pr=0.72)

    v = zeros(3)
    J = zeros(3)
    λ = 0

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, v, AL, J, λ, PARL, SYS)
    QR = conserved(ρR, pR, v, AR, J, λ, PARR, SYS)
    for i in range(nx):
        if i*dx < 0.7:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [PARL, PARR], [0.7]

def water_water_IC():
    """ tf = 1.5e-4
        L = 1
    """
    PAR = material_parameters(y=4.4, pINF=6e8, cv=950, ρ0=1000, p0=1e5, cs=1e-4, α=1e-4, μ=1e-3,
                              Pr=7)

    ρL = 1000
    pL = 7e8
    AL = ρL**(1/3) * eye(3)

    ρR = 1000
    pR = pL / 7000
    AR = ρR**(1/3) * eye(3)

    v = zeros(3)
    J = zeros(3)
    λ = 0

    u = zeros([nx, ny, nz, 18])
    QL = conserved(ρL, pL, v, AL, J, λ, PAR, SYS)
    QR = conserved(ρR, pR, v, AR, J, λ, PAR, SYS)
    for i in range(nx):
        if i*dx < 0.5:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [PAR]*2, [0.5]

def helium_bubble_IC():
    """ tf = 0.0014
        L = 1
    """
    PAR_air = material_parameters(y=1.4, pINF=0, cv=721, ρ0=1.18, p0=10100, cs=1, α=1, μ=1.85e-5,
                                  Pr=0.714)
    PAR_hel = material_parameters(y=1.66, pINF=0, cv=3127, ρ0=0.163, p0=10100, cs=1, α=1, μ=1.99e-5,
                                  Pr=0.688)
    ρL = 1.3333
    pL = 1.5e5
    vL = array([35.35*sqrt(10), 0, 0])
    AL = ρL**(1/3) * eye(3)

    ρM = 1
    pM = 1e5
    vM = zeros(3)
    AM = ρM**(1/3) * eye(3)

    ρR = 0.1379
    pR = 1e5
    vR = zeros(3)
    AR = ρR**(1/3) * eye(3)

    J = zeros(3)
    λ = 0

    u = zeros([nx, ny, nz, 18])
    Q1 = conserved(ρL, pL, vL, AL, J, λ, PAR_air, SYS)
    Q2 = conserved(ρM, pM, vM, AM, J, λ, PAR_air, SYS)
    Q3 = conserved(ρR, pR, vR, AR, J, λ, PAR_hel, SYS)
    for i in range(nx):
        if i*dx < 0.05:
            u[i, 0, 0] = Q1
        elif i*dx < 0.4:
            u[i, 0, 0] = Q2
        elif i*dx < 0.6:
            u[i, 0, 0] = Q3
        else:
            u[i, 0, 0] = Q2

    return u, [PAR_air, PAR_hel, PAR_air], [0.4, 0.6]

def helium_heat_transmission_IC():
    """ tf = 5e-9
        L = 4.25e-6
        nx = 400
        W = 1
    """
    ρL = 1.18
    ρR = 0.164
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    λ = 0
    p0 = 101325

    PAR_air = material_parameters(y=1.4, pINF=0, cv=718, ρ0=ρL, p0=p0, cs=55, α=5e2, μ=1.84e-5,
                                  Pr=0.715)
    PAR_hel = material_parameters(y=1.66, pINF=0, cv=3128, ρ0=ρR, p0=p0, cs=55, α=5e2, μ=1.98e-5,
                                  Pr=0.688)

    QL = conserved(ρL, p0, v, A, J, λ, PAR_air, SYS)
    QR = conserved(ρR, p0, v, A, J, λ, PAR_hel, SYS)
    u = zeros([nx, ny, nz, 18])
    x0 = Lx/4
    for i in range(nx):
        if i*dx < x0:
            u[i] = QL
        else:
            u[i] = QR

    return u, [PAR_air, PAR_hel], [x0]
