from numpy import array, eye, sqrt, zeros

from auxiliary.bc import temperature_fix_pressure, standard_BC
from auxiliary.classes import material_parameters
from gpr.functions import conserved, primitive
from gpr.variables import c_0, temperature
from options import nx, ny, nz, L, dx, viscous, thermal, reactive, doubleTime, W, subsystems


SET_THERMAL_IMPULSE = 0


def chapman_jouguet_IC():
    """ tf = 0.5
        L = 1
        reactionType = 'd'
    """
    params = material_parameters(γ=1.4, pINF=0, cv=2.5, ρ0=1, p0=1, cs=1e-8, α=1e-8, μ=1e-4,
                                 Pr=0.75, Qc=1, Kc=250, Ti=0.25)

    ρL = 1.4
    pL = 1
    vL = zeros(3)
    AL = ρL**(1/3) * eye(3)
    λL = 0

    ρR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    AR = ρR**(1/3) * eye(3)
    λR = 1

    J = zeros(3)

    QL = conserved(ρL, pL, vL, AL, J, λL, params, subsystems)
    QR = conserved(ρR, pR, vR, AR, J, λR, params, subsystems)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i*dx < L/4:
            u[i, 0, 0] = QL
        else:
            u[i, 0, 0] = QR

    return u, [params]*1, []

def CKR_IC():
    """ nx = 360
        reactionType = 'a'

        CKR Deflagration:
        tf = 1.05e-8
        L = 8.5e-6
        ε = 1/20

        Detonation:
        tf = 6e-9
        L = 8.5e-6
        ε = 1/15
    """
    γ = 1.4
    cv = 718
    ρ0 = 1.17601
    p0 = 101325

    params = material_parameters(γ=γ, pINF=0, cv=cv, ρ0=ρ0, p0=p0, cs=55, α=5e2, μ=1.98e-5, Pr=0.72,
                                 Qc=6*γ*cv*300, ε=1/20, Bc=7e10)

    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    λ = 1

    u = zeros([nx, ny, nz, 18])
    Q = conserved(ρ0, p0, v, A, J, λ, params, subsystems)
    for i in range(nx):
        u[i,0,0] = Q

    return u, [params], []


def CKR_BC(u, dt, params, subsystems):

    ρ0 = params.ρ0, p0 = params.p0; γ = params.γ; pINF = params.pINF; Pr = params.Pr

    c0 = c_0(ρ0, p0, γ, pINF)
    q = γ * p0 * c0 * W / ((γ-1) * Pr)
    u[0,0,0,1] += q * dt / dx

    if SET_THERMAL_IMPULSE:
        Q = u[0,0,0]
        P = primitive(Q, params, subsystems)
        T = temperature(P.ρ, P.p, γ, pINF)
        J = array([q, 0, 0]) / (params.α2 * T)
        u[0,0,0] = conserved(P.ρ, P.p, P.v, P.A, J, P.λ, params, subsystems)

    return standard_BC(u, reflectLeft=1)

def fixed_wall_temp_BC(u, t, params, subsystems):

    Tc = params.T0 * (1 + t / doubleTime)                       # Temperature at which wall is held
    uNew = standard_BC(u, reflectLeft=1, reflectRight=1)
    Q0 = uNew[0,0,0]
    P0 = primitive(Q0, params, subsystems)

    if P0.T < Tc:
        p0 = temperature_fix_pressure(P0.ρ, Tc, params)
    else:
        p0 = P0.p

    if SET_THERMAL_IMPULSE:
        Q1 = u[1,0,0]
        P1 = primitive(Q1, params)
        T1 = temperature(P1.ρ, P1.p)
        J0 = array([params.κ * (temperature(P0.ρ, P0.p)-T1) / (params.α2 * Tc * dx), 0, 0])
    else:
        J0 = P0.J

    uNew[0,0,0] = conserved(P0.ρ, p0, P0.v, P0.A, J0, P0.c, params, viscous, thermal, reactive)
    return uNew

def scales(params):
    tScale = params.μ / (params.p0 * params.y)                 # Pr * (κ / (ρ0 * cp)) / c0^2
    xScale = tScale * sqrt(params.y * (params.p0+params.pINF) / params.ρ0)  # tScale * c0
    return tScale, xScale
