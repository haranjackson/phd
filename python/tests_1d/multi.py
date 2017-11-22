from numpy import array, eye, sqrt, zeros

from system.gpr.misc.objects import material_parameters
from system.gpr.misc.structures import Cvec
from tests_1d.common import riemann_IC
from options import nx, ny, nz, nV, dx


def sod_shock_IC():
    """ tf = 0.2
        L = 1
    """
    ρL = 1
    pL = 1
    vL = zeros(3)

    ρR = 0.125
    pR = 0.1
    vR = zeros(3)

    PAR = material_parameters(EOS='sg', γ=1.4, pINF=0, cv=2.5, ρ0=1, p0=1,
                              cs=1, α=1, μ=5e-4, Pr=2/3)

    return riemann_IC(ρL, pL, vL, ρR, pR, vR, PAR)

def water_gas_IC():
    """ tf = 237.44e-6
        L = 1
    """
    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    PARR = material_parameters(EOS='sg', γ=1.4, pINF=0, cv=718, ρ0=1.176, p0=101325,
                               cs=55, α=5e2, μ=1.98e-5, Pr=0.72)
    PARL = material_parameters(EOS='sg', γ=4.4, pINF=6e8, cv=950, ρ0=1000, p0=1,
                               cs=1e-4, α=1e-4, μ=1e-3, Pr=7)

    return riemann_IC(ρL, pL, vL, ρR, pR, vR, PARL, PARR, 0.7)

def water_water_IC():
    """ tf = 1.5e-4
        L = 1
    """
    PAR = material_parameters(EOS='sg', γ=4.4, pINF=6e8, cv=950, ρ0=1000, p0=1e5,
                              cs=1e-4, α=1e-4, μ=1e-3, Pr=7)

    ρL = 1000
    pL = 7e8
    vL = zeros(3)

    ρR = 1000
    pR = pL / 7000
    vR = zeros(3)

    return riemann_IC(ρL, pL, vL, ρR, pR, vR, PAR)

def helium_bubble_IC():
    """ tf = 0.0014
        L = 1
    """
    PAR_air = material_parameters(EOS='sg', γ=1.4, pINF=0, cv=721, ρ0=1.18, p0=10100,
                                  cs=1, α=1, μ=1.85e-5, Pr=0.714)
    PAR_hel = material_parameters(EOS='sg', γ=1.66, pINF=0, cv=3127, ρ0=0.163, p0=10100,
                                  cs=1, α=1, μ=1.99e-5, Pr=0.688)
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

    u = zeros([nx, ny, nz, nV])
    Q1 = Cvec(ρL, pL, vL, AL, J, PAR_air)
    Q2 = Cvec(ρM, pM, vM, AM, J, PAR_air)
    Q3 = Cvec(ρR, pR, vR, AR, J, PAR_hel)

    for i in range(nx):

        u[i, 0, 0, -3] = i*dx - 0.05
        u[i, 0, 0, -2] = i*dx - 0.4
        u[i, 0, 0, -1] = i*dx - 0.6

        if i*dx < 0.05:
            u[i, 0, 0, :-3] = Q1
        elif i*dx < 0.4:
            u[i, 0, 0, :-3] = Q2
        elif i*dx < 0.6:
            u[i, 0, 0, :-3] = Q3
        else:
            u[i, 0, 0, :-3] = Q2

    return u, [PAR_air, PAR_air, PAR_hel, PAR_air]
