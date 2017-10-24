from itertools import product

from numpy import array, eye, zeros, arange, around, concatenate, exp, int64, ones, sqrt
from scipy.optimize import brentq
from scipy.special import erf

from auxiliary.classes import material_parameters
from gpr.variables.vectors import Cvec, Cvec_to_Pclass
from gpr.variables.wavespeeds import c_0
from options import nx, ny, nz, dx, Lx, RGFM


def first_stokes_problem_IC():
    """ tf = 1
        L = 1
    """
    γ = 1.4
    μ = 1e-2 # 1e-3 # 1e-4

    ρ = 1
    p = 1 / γ
    v = array([0, 0.1, 0])
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=1, α=1e-16, μ=μ, Pr=0.75)

    QL = Cvec(ρ, p, -v, A, J, 0, PAR)
    QR = Cvec(ρ, p,  v, A, J, 0, PAR)
    u = zeros([nx, ny, nz, 18])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        if i*dx < Lx/2:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR]*1, []

def first_stokes_problem_exact(μ, n=200, v0=0.1, t=1):
    dx = 1/n
    x = linspace(-0.5+dx/2, 0.5-dx/2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))

def viscous_shock_exact(x, Ms, PAR, center=0):
    """ Returns the density, pressure, and velocity of the viscous shock (Mach number Ms) at x
    """
    x -= center
    ρ0 = PAR.ρ0
    p0 = PAR.p0
    γ = PAR.γ
    pINF = PAR.pINF
    μ = PAR.μ

    if Ms==2:
        l = 0.3
    elif Ms==3:
        l = 0.13

    if x > l:
        x=l
    elif x < -l:
        x=-l

    c0 = c_0(ρ0, p0, γ, pINF)
    a = 2 / (Ms**2 * (γ+1)) + (γ-1)/(γ+1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1-a)/2)**(1-a)
    c2 = 3/4 * Re * (Ms**2-1) / (γ*Ms**2)

    f = lambda z: (1-z)/(z-a)**a - c1 * exp(c2*-x)

    vbar = brentq(f, a+1e-16, 1)
    p = p0 / vbar * (1 + (γ-1)/2 * Ms**2 * (1-vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0  - v    # Shock travelling into fluid at rest

    return ρ, p, v

def viscous_shock_IC(center=0):
    Ms = 2
    γ = 1.4
    pINF = 0
    ρ0 = 1
    p0 = 1 / γ
    μ = 2e-2

    PAR = material_parameters(γ=γ, pINF=0, cv=2.5, ρ0=ρ0, p0=p0, cs=5, α=5, μ=2e-2, Pr=0.75)

    x = arange(-Lx/2, Lx/2, 1/nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i], Ms, PAR, center=center)

    v -= v[0]           # Velocity in shock 0

    u = zeros([nx, 1, 1, 18])
    for i in range(nx):
        A = (ρ[i])**(1/3) * eye(3)
        J = zeros(3)
        λ = 0
        u[i,0,0] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J, λ, PAR)

    return u, [PAR], []

def viscous_shock_exact_x(n, M=2, t=0.2):
    return arange(M*t-0.25, M*t+0.75, 1/n)

def heat_conduction_IC():
    """ tf = 1
        L = 1
    """
    ρL = 2
    ρR = 0.5
    p0 = 1
    v0 = zeros(3)
    AL = ρL**(1/3) * eye(3)
    AR = ρR**(1/3) * eye(3)
    J0 = zeros(3)

    PAR = material_parameters(γ=1.4, pINF=0, cv=2.5, ρ0=1, p0=p0, cs=1, α=2, μ=1e-2, κ=1e-2)

    QL = Cvec(ρL, p0, v0, AL, J0, 0, PAR)
    QR = Cvec(ρR, p0, v0, AR, J0, 0, PAR)
    u = zeros([nx, ny, nz, 18])
    x0 = Lx / 2
    for i in range(nx):
        if i*dx < x0:
            u[i,0,0] = QL
        else:
            u[i,0,0] = QR

    if RGFM:
        return u, [PAR, PAR], [0.5]
    else:
        return u, [PAR], []

def semenov_IC():
    Rc = 8.314459848

    cv = 2.5
    T0 = 1
    Qc = 4
    ε = 1/20 # 1/15 # 1/10
    Ea = Rc * 1 / ε
    Bc = (cv * T0**2 * Rc) / (Ea * Qc) * exp(Ea/(Rc*T0))

    ρ = 1
    p = 1
    v = zeros(3)
    A = ρ**(1/3) * eye(3)
    J = zeros(3)
    λ = 1

    PAR = material_parameters(γ=1.4, pINF=0, cv=cv, ρ0=ρ, p0=p, Qc=Qc, ε=ε, Bc=Bc)

    Q = Cvec(ρ, p, v, A, J, λ, PAR)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        u[i,0,0] = Q

    return u, [PAR], []

def semenov_temp(dataArray, PAR):
    states = [da[0, 0, 0] for da in dataArray]
    return [Cvec_to_Pclass(state, PAR, 0, 0, 1).T for state in states]
