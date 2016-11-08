from numpy import array, eye, zeros, arange, around, concatenate, exp, int64, ones, sqrt
from scipy.optimize import brentq
from scipy.special import erf

from auxiliary.classes import material_parameters
from gpr.variables.vectors import conserved, primitive
from gpr.variables.wavespeeds import c_0
from options import nx, ny, nz, Ms, dx, Rc, L, SYS


def first_stokes_problem_IC():
    """ tf = 1
        L = 1
    """
    γ = 1.4
    μ = 1e-4 # 1e-3 # 1e-4

    ρ = 1
    p = 1 / γ
    v = array([0, 0.1, 0])
    A = eye(3)
    J = zeros(3)

    PAR = material_parameters(γ=γ, pINF=0, cv=1, ρ0=ρ, p0=p, cs=1, α=1e-16, μ=μ, Pr=0.75)

    QL = conserved(ρ, p, -v, A, J, 0, PAR, SYS)
    QR = conserved(ρ, p,  v, A, J, 0, PAR, SYS)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i*dx < L/2:
            u[i,0,0] = QL
        else:
            u[i,0,0] = QR

    return u, [PAR]*1, []

def first_stokes_problem_exact(x, μ, v0=0.1, t=1):
    return v0 * erf(x / (2 * sqrt(μ * t)))

def viscous_shock_IC():
    CENTER = 1
    γ = 1.4
    pINF = 0
    ρ0 = 1
    p0 = 1 / γ
    μ = 2e-2

    PAR = material_parameters(γ=γ, pINF=0, cv=2.5, ρ0=ρ0, p0=p0, cs=5, α=5, μ=2e-2, Pr=0.75)

    if Ms==2:
        x0 = 0.07   # Position of center of shock for shock to start at x = 0
        l = 0.3
    elif Ms==3:
        x0 = 0.04   # Position of center of shock for shock to start at x = 0
        l = 0.13

    c0 = c_0(ρ0, p0, γ, pINF)
    a = 2 / (Ms**2 * (γ+1)) + (γ-1)/(γ+1)

    Re = ρ0 * c0 * Ms / μ
    c1 = ((1-a)/2)**(1-a)
    c2 = 3/4 * Re * (Ms**2-1) / (γ*Ms**2)

    x = around(arange(-l, l, dx), decimals=14)
    n = x.size
    vbar = zeros(n)
    for i in range(n):
        f = lambda v: (1-v)/(v-a)**a - c1 * exp(c2*-x[i])
        vbar[i] = brentq(f, a+1e-16, 1)

    p = p0 / vbar * (1 + (γ-1)/2 * Ms**2 * (1-vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0  - v    # Shock travelling into fluid at rest
    v -= v[0]           # Velocity in shock 0

    if CENTER:
        rem = int((nx-n)/2)
        reps = ones(n)
        reps[0] = rem+1
        reps[-1] = rem+1
        v = v.repeat(reps.astype(int64))
        p = p.repeat(reps.astype(int64))
        ρ = ρ.repeat(reps.astype(int64))
    else:
        x = x + x0
        p = p[x>=0]
        ρ = ρ[x>=0]
        v = v[x>=0]
        x = x[x>=0]

        n = x.size
        reps = ones(n)
        reps[-1] = nx - n + 1
        v = v.repeat(reps.astype(int64))
        p = p.repeat(reps.astype(int64))
        ρ = ρ.repeat(reps.astype(int64))
        x = concatenate((x, arange(l+x0, 1, dx)))

    u = zeros([nx, 1, 1, 18])
    for i in range(nx):
        A = (ρ[i])**(1/3) * eye(3)
        J = zeros(3)
        λ = 1
        u[i,0,0] = conserved(ρ[i], p[i], array([v[i], 0, 0]), A, J, λ, PAR, SYS)

    return u, [PAR], []

def viscous_shock_exact_x(n, M=2, t=0.2):
    return arange(M*t-0.25, M*t+0.75, 1/n)

def heat_conduction_IC():
    ρL = 2
    ρR = 0.5
    p0 = 1
    v0 = zeros(3)
    AL = ρL**(1/3) * eye(3)
    AR = ρR**(1/3) * eye(3)
    J0 = zeros(3)

    PAR = material_parameters(γ=1.4, pINF=0, cv=2.5, ρ0=1, p0=p0, cs=1, α=2, μ=1e-2, κ=1e-2)

    QL = conserved(ρL, p0, v0, AL, J0, 0, PAR, SYS)
    QR = conserved(ρR, p0, v0, AR, J0, 0, PAR, SYS)
    u = zeros([nx, ny, nz, 18])
    x0 = L / 2
    for i in range(nx):
        if i*dx < x0:
            u[i,0,0] = QL
        else:
            u[i,0,0] = QR

    return u, [PAR]*1, []

def semenov_IC():
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

    Q = conserved(ρ, p, v, A, J, λ, PAR, SYS)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        u[i,0,0] = Q

    return u, [PAR], []

def semenov_temp(dataArray, PAR):
    states = [da[0, 0, 0] for da in dataArray]
    return [primitive(state, PAR, 0, 0, 1).T for state in states]
