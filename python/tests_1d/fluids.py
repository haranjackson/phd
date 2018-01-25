from numpy import array, eye, arange, exp, sqrt, zeros
from scipy.optimize import brentq
from scipy.special import erf

from gpr.misc.objects import material_parameters
from gpr.misc.structures import Cvec
from gpr.variables.wavespeeds import c_0
from tests_1d.common import riemann_IC, MP_AIR
from options import nx, nV, Lx, N


def heat_conduction_IC():

    tf = 1

    ρL = 2
    pL = 1
    vL = zeros(3)

    ρR = 0.5
    pR = 1
    vR = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0,
                             b0=1, cα=2, μ=1e-2, κ=1e-2)

    print("HEAT CONDUCTION IN A GAS: N =", N)
    return riemann_IC(tf, ρL, pL, vL, ρR, pR, vR, MP_AIR)


def first_stokes_problem_exact(μ, n=200, v0=0.1, t=1):
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))


def first_stokes_problem_IC():

    tf = 1

    γ = 1.4
    μ = 1e-2  # 1e-3 # 1e-4

    ρL = 1
    pL = 1 / γ
    vL = array([0, -0.1, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, 0.1, 0])

    MP = material_parameters(EOS='sg', ρ0=1, cv=1, p0=1 / γ, γ=γ, pINF=0,
                             b0=1, cα=1e-16, μ=μ, Pr=0.75)

    print("FIST STOKES PROBLEM: N =", N, "μ =", μ)
    return riemann_IC(tf, ρL, pL, vL, ρR, pR, vR, MP)


def viscous_shock_exact(x, Ms, MP, μ, center=0):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    x -= center
    ρ0 = MP.ρ0
    p0 = MP.p0
    γ = MP.γ
    pINF = MP.pINF

    if Ms == 2:
        l = 0.3
    elif Ms == 3:
        l = 0.13

    if x > l:
        x = l
    elif x < -l:
        x = -l

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

    Ms = 2
    γ = 1.4
    pINF = 0
    ρ0 = 1
    p0 = 1 / γ
    μ = 2e-2

    MP = material_parameters(EOS='sg', ρ0=ρ0, cv=2.5, p0=p0, γ=γ, pINF=0,
                             b0=5, cα=5, μ=2e-2, Pr=0.75)

    x = arange(-Lx / 2, Lx / 2, 1 / nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i], Ms, MP, μ, center=center)

    v -= v[0]           # Velocity in shock 0

    u = zeros([nx, 1, 1, nV])
    for i in range(nx):
        A = (ρ[i])**(1 / 3) * eye(3)
        J = zeros(3)
        λ = 0
        u[i] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J, MP)

    print("VISCOUS SHOCK: N =", N)
    return u, [MP], tf
