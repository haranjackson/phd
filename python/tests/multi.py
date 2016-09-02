from numpy import array, eye, sqrt, zeros

from auxiliary.common import material_parameters
from gpr.functions import conserved
from options import nx, ny, nz, L, dx, viscous, thermal, reactive


def sod_shock_IC():
    """ tf = 0.2
        L = 1
    """
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

    params = material_parameters(y=1.4, pINF=0, cv=2.5, r0=1, p0=1, cs=1, alpha=1, mu=5e-4, Pr=2/3)
    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, vL, AL, JL, 0, params, 1, 1, 0)
    QR = conserved(rR, pR, vR, AR, JR, 0, params, 1, 1, 0)
    for i in range(nx):
        if i < int(nx/2):
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [params]*2, [L/2]

def water_gas_IC():
    """ tf = 237.44e-6
        L = 1
    """
    J = zeros(3)
    c = 0
    v = zeros(3)

    rL = 1000
    pL = 1e9
    AL = rL**(1/3) * eye(3)
    paramsL = material_parameters(y=4.4, pINF=6e8, cv=950, r0=1000, p0=1,
                                  cs=1e-4, alpha=1e-4, mu=1e-3, Pr=7)

    rR = 50
    pR = 101325
    AR = rR**(1/3) * eye(3)
    paramsR = material_parameters(y=1.4, pINF=0, cv=718, r0=1.176, p0=101325,
                                  cs=55, alpha=5e2, mu=1.98e-5, Pr=0.72)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, v, AL, J, c, paramsL, viscous, thermal, reactive)
    QR = conserved(rR, pR, v, AR, J, c, paramsR, viscous, thermal, reactive)
    for i in range(nx):
        if i*dx < 0.7:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [paramsL, paramsR], [0.7]

def water_water_IC():
    """ tf = 1.5e-4
        L = 1
    """
    J = zeros(3)
    c = 0
    v = zeros(3)
    params = material_parameters(y=4.4, pINF=6e8, cv=950, r0=1000, p0=1e5,
                                 cs=1e-4, alpha=1e-4, mu=1e-3, Pr=7)

    rL = 1000
    pL = 7e8
    AL = rL**(1/3) * eye(3)

    rR = 1000
    pR = pL / 7000
    AR = rR**(1/3) * eye(3)

    u = zeros([nx, ny, nz, 18])
    QL = conserved(rL, pL, v, AL, J, c, params, viscous, thermal, reactive)
    QR = conserved(rR, pR, v, AR, J, c, params, viscous, thermal, reactive)
    for i in range(nx):
        if i*dx < 0.5:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [params]*2, [0.5]

def multimaterial2_IC():
    """ tf = 0.0014
        L = 1
    """
    params_air = material_parameters(y=1.4, pINF=0, cv=721, r0=1.18, p0=10100,
                                     cs=1, alpha=1, mu=1.85e-5, Pr=0.714)
    params_hel = material_parameters(y=1.66, pINF=0, cv=3127, r0=0.163, p0=10100,
                                     cs=1, alpha=1, mu=1.99e-5, Pr=0.688)
    rL = 1.3333
    pL = 1.5e5
    vL = array([35.35*sqrt(10), 0, 0])
    AL = rL**(1/3) * eye(3)

    rM = 1
    pM = 1e5
    vM = zeros(3)
    AM = rM**(1/3) * eye(3)

    rR = 0.1379
    pR = 1e5
    vR = zeros(3)
    AR = rR**(1/3) * eye(3)

    u = zeros([nx, ny, nz, 18])
    J = zeros(3)
    c = 0
    Q1 = conserved(rL, pL, vL, AL, J, c, params_air, viscous, thermal, reactive)
    Q2 = conserved(rM, pM, vM, AM, J, c, params_air, viscous, thermal, reactive)
    Q3 = conserved(rR, pR, vR, AR, J, c, params_hel, viscous, thermal, reactive)
    for i in range(nx):
        if i*dx < 0.05:
            u[i, 0, 0] = Q1
        elif i*dx < 0.4:
            u[i, 0, 0] = Q2
        elif i*dx < 0.6:
            u[i, 0, 0] = Q3
        else:
            u[i, 0, 0] = Q2

    return u, [params_air, params_hel, params_air], [0.4, 0.6]

def helium_heat_transmission_IC():
    """ tf = 5e-9
        L = 4.25e-6
        nx = 400
        W = 1
    """
    rL = 1.18
    rR = 0.164
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    c = 0
    p0 = 101325

    params_air = material_parameters(y=1.4, pINF=0, cv=718, r0=rL, p0=p0,
                                     cs=55, alpha=5e2, mu=1.84e-5, Pr=0.715)
    params_hel = material_parameters(y=1.66, pINF=0, cv=3128, r0=rR, p0=p0,
                                     cs=55, alpha=5e2, mu=1.98e-5, Pr=0.688)

    QL = conserved(rL, p0, v, A, J, c, params_air, viscous, thermal, reactive)
    QR = conserved(rR, p0, v, A, J, c, params_hel, viscous, thermal, reactive)
    u = zeros([nx, ny, nz, 18])
    x0 = L/4
    for i in range(nx):
        if i*dx < x0:
            u[i] = QL
        else:
            u[i] = QR

    return u, [params_air, params_hel], [x0]
