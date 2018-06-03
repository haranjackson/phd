from numpy import array, eye, arange, concatenate, exp, flip, linspace, sqrt, zeros
from scipy.optimize import brentq
from scipy.special import erf

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec, State
from gpr.opts import NV
from gpr.vars.wavespeeds import c_0
from gpr.tests.boundaries import wall_BC
from gpr.tests.one.common import primitive_IC


def heat_conduction_IC(isMulti=False):

    tf = 1
    nx = 200
    Lx = 1
    MPs = [material_params(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0, b0=1,
                           cα=2, μ=1e-2, κ=1e-2)]

    if isMulti:
        MPs = 2 * MPs

    ρL = 2
    pL = 1
    vL = zeros(3)

    ρR = 0.5
    pR = 1
    vR = zeros(3)

    dX = [Lx / nx]

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def first_stokes_problem_exact(μ, n=100, v0=0.1, t=1):
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))


def first_stokes_problem_IC(isMulti=False):

    tf = 1
    nx = 200
    Lx = 1

    γ = 1.4
    μ = 1e-2  # 1e-3 # 1e-4

    MPs = [material_params(EOS='sg', ρ0=1, cv=1, p0=1 / γ, γ=γ, b0=1, cα=1e-16,
                           μ=μ, Pr=0.75)]

    if isMulti:
        MPs = 2 * MPs

    ρL = 1
    pL = 1 / γ
    vL = array([0, -0.1, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, 0.1, 0])

    dX = [Lx / nx]

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs)
    return u, MPs, tf, dX


def viscous_shock_exact(x, Ms, MP, μ, center=0):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    x -= center
    ρ0 = MP.ρ0
    p0 = MP.p0
    γ = MP.γ

    if Ms == 2:
        L = 0.3
    elif Ms == 3:
        L = 0.13

    x = min(x, L)
    x = max(x, -L)

    c0 = c_0(ρ0, p0, eye(3), MP)
    a = 2 / (Ms**2 * (γ + 1)) + (γ - 1) / (γ + 1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1 - a) / 2)**(1 - a)
    c2 = 3 / 4 * Re * (Ms**2 - 1) / (γ * Ms**2)

    def f(z): return (1 - z) / (z - a)**a - c1 * exp(c2 * -x)

    vbar = brentq(f, a + 1e-16, 1)
    p = p0 / vbar * (1 + (γ - 1) / 2 * Ms**2 * (1 - vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0 - v    # Shock travelling into fluid at rest

    return ρ, p, v


def viscous_shock_exact_x(n, M=2, t=0.2):
    return arange(M * t - 0.25, M * t + 0.75, 1 / n)


def viscous_shock_IC(center=0):

    tf = 0.2
    nx = 100
    Lx = 1

    Ms = 2
    γ = 1.4
    ρ0 = 1
    p0 = 1 / γ
    μ = 2e-2

    MP = material_params(EOS='sg', ρ0=ρ0, cv=2.5, p0=p0, γ=γ, pINF=0, b0=5,
                         cα=5, μ=2e-2, Pr=0.75)

    dX = [Lx / nx]

    x = arange(-Lx / 2, Lx / 2, 1 / nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i], Ms, MP, μ, center=center)

    v -= v[0]           # Velocity in shock 0

    u = zeros([nx, NV])
    for i in range(nx):
        A = (ρ[i])**(1 / 3) * eye(3)
        J = zeros(3)
        u[i] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J, MP)

    return u, [MP], tf, dX


def hagen_poiseuille_IC():

    tf = 3
    Lx = 0.25
    nx = 100
    dp = 0.48

    γ = 1.4
    cs = 8
    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    δp = array([0, dp, 0])

    K = 1e-2; n = 0.9; τ1 = 6 * K**(1/n) / ρ / cs**2
    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=cs,
                         σY=1, τ1=τ1, n=(1-n)/n, PLASTIC=True, δp=δp)

    Q = Cvec(ρ, p, v, A, J, MP)
    u = array([Q] * nx)

    return u, [MP], tf, [Lx / nx]


def hagen_poiseuille_BC(u, N, *args):
    dp = 0.48

    γ = 1.4
    cs = 8
    ρ = 1
    p = 100 / γ
    δp = array([0, dp, 0])

    K = 1e-2; n = 0.9; τ1 = 6 * K**(1/n) / ρ / cs**2
    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=cs,
                         σY=1, τ1=τ1, n=(1-n)/n, PLASTIC=True, δp=δp)

    return wall_BC(u, N, 1, [1], MP)


def hagen_poiseuille_modifier(u):

    nx = u.shape[0]

    γ = 1.4
    cs = 8
    ρ = 1
    p = 100 / γ

    K = 1e-2; n = 0.9; τ1 = 6 * K**(1/n) / ρ / cs**2
    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=cs,
                         σY=1, τ1=τ1, n=(1-n)/n, PLASTIC=True)

    for i in range(nx):
        Q = u[i]
        P = State(Q, MP)
        u[i] = Cvec(P.ρ, p, P.v, P.A, P.J, MP)


def hagen_poiseuille_exact(nx=100):

    Lx = 0.25
    dp = 0.48
    ρ = 1
    μ = 1e-2
    n = 0.9

    k = (n + 1) / n
    x = linspace(0, Lx, nx+2)[int(nx/2+1):-1]
    y = ρ / k * (dp / μ)**(1 / n) * ((Lx / 2)**k - (x - Lx / 2)**k)
    return concatenate([flip(y, axis=0), y])
