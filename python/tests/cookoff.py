from numpy import array, eye, sqrt, zeros

from auxiliary.bc import temperature_fix_pressure, standard_BC
from auxiliary.common import material_parameters
from gpr.functions import conserved, primitive
from gpr.variables import c_0, temperature
from options import nx, ny, nz, L, dx, viscous, thermal, reactive, doubleTime, W


SET_THERMAL_IMPULSE = 0


def chapman_jouguet_IC():
    """ tf = 0.5
        L = 1
        reactionType = 'd'
    """
    params = material_parameters(y=1.4, pINF=0, cv=2.5, r0=1, p0=1,
                                 cs=1e-8, alpha=1e-8, mu=1e-4, Pr=0.75, Qc=1, Kc=250, Ti=0.25)

    rL = 1.4
    pL = 1
    vL = zeros(3)
    AL = rL**(1/3) * eye(3)
    cL = 0

    rR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    AR = rR**(1/3) * eye(3)
    cR = 1

    J = zeros(3)

    QL = conserved(rL, pL, vL, AL, J, cL, params, 1, 1, 1)
    QR = conserved(rR, pR, vR, AR, J, cR, params, 1, 1, 1)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i*dx < L/4:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR
    return u, [params]*1, []

def CKR_IC():
    """ tf = 1.05e-8
        L = 8.5e-6
        nx = 360
        reactionType = 'a'
    """
    y = 1.4
    pINF = 0
    cv = 718
    r0 = 1.17601
    p0 = 101325
    mu = 1.98e-5

    params = material_parameters(y=y, pINF=pINF, cv=cv, r0=r0, p0=p0,
                                 cs=55, alpha=5e2, mu=mu, Pr=0.72,
                                 Qc=6*y*cv*300, epsilon=1/20, Bc=7e10)
    u = zeros([nx, ny, nz, 18])
    v = zeros(3)
    A = r0**(1/3) * eye(3)
    J = zeros(3)
    c = 1
    Q = conserved(r0, p0, v, A, J, c, params, viscous, thermal, reactive)
    for i in range(nx):
        u[i,0,0] = Q
    return u, [params], []

def detonation_IC():
    """ tf = 6e-9
        L = 8.5e-6
        nx = 400 / 1600
        reactionType = 'a'
    """
    y = 1.4
    pINF = 0
    cv = 718
    r0 = 1.17601
    p0 = 101325
    mu = 1.98e-5

    params = material_parameters(y=y, pINF=pINF, cv=cv, r0=r0, p0=p0,
                                 cs=55, alpha=5e2, mu=mu, Pr=0.72,
                                 Qc=6*y*cv*300, epsilon=1/15, Bc=7e10)
    u = zeros([nx, ny, nz, 18])
    v = zeros(3)
    A = r0**(1/3) * eye(3)
    J = zeros(3)
    c = 1
    Q = conserved(r0, p0, v, A, J, c, params, viscous, thermal, reactive)
    for i in range(nx):
        u[i,0,0] = Q
    return u, [params], []


def CKR_BC(u, params, dt):

    c0 = c_0(params.r0, params.p0, params.y, params.pINF)
    q = params.y * params.p0 * c0 * W / ((params.y-1) * params.Pr)
    u[0,0,0,1] += q * dt / dx

#    Q = u[0,0,0]
#    P = primitive(Q, defaultParams, viscous, thermal, reactive)
#    if SET_THERMAL_IMPULSE:
#        T = temperature(P.r, P.p, y, pINF)
#        J = array([q, 0, 0]) / (alpha2 * T)
#    else:
#        J = zeros(3)
#    u[0,0,0] = conserved(P.r, P.p, P.v, P.A, J, P.c, y, pINF, viscous, thermal, reactive)

    return standard_BC(u, reflectLeft=1)

def fixed_wall_temp_BC(u, t, params, viscous, thermal, reactive):
    Tc = params.T0 * (1 + t / doubleTime)                       # Temperature at which wall is held
    uNew = standard_BC(u, reflectLeft=1, reflectRight=1)
    Q0 = uNew[0,0,0]
    P0 = primitive(Q0, params, viscous, thermal, reactive)

    if P0.T < Tc:
        p0 = temperature_fix_pressure(P0.r, Tc, params)
    else:
        p0 = P0.p

    if SET_THERMAL_IMPULSE:
        Q1 = u[1,0,0]
        P1 = primitive(Q1, params)
        T1 = temperature(P1.r, P1.p)
        J0 = array([params.kappa * (temperature(P0.r, P0.p)-T1) / (params.alpha2 * Tc * dx), 0, 0])
    else:
        J0 = P0.J

    uNew[0,0,0] = conserved(P0.r, p0, P0.v, P0.A, J0, P0.c, params, viscous, thermal, reactive)
    return uNew

def scales(params):
    tScale = params.mu / (params.p0 * params.y)                 # Pr * (kappa / (r0 * cp)) / c0^2
    xScale = tScale * sqrt(params.y * (params.p0+params.pINF) / params.r0)  # tScale * c0
    return tScale, xScale