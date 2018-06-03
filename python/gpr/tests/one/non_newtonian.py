from numpy import array, concatenate, eye, flip, linspace, zeros

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec, State
from gpr.tests.boundaries import wall_BC


HP_n = 1.3


def hagen_poiseuille_IC():

    tf = 3
    Lx = 0.25
    nx = 200
    dp = 0.48

    γ = 1.4
    cs = 8
    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    δp = array([0, dp, 0])

    K = 1e-2
    n = HP_n
    τ1 = 6 * K**(1/n) / ρ / cs**2
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

    K = 1e-2
    n = HP_n
    τ1 = 6 * K**(1/n) / ρ / cs**2
    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=cs,
                         σY=1, τ1=τ1, n=(1-n)/n, PLASTIC=True, δp=δp)

    return wall_BC(u, N, 1, [1], MP)


def hagen_poiseuille_modifier(u):

    nx = u.shape[0]

    γ = 1.4
    cs = 8
    ρ = 1
    p = 100 / γ

    K = 1e-2
    n = HP_n
    τ1 = 6 * K**(1/n) / ρ / cs**2
    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=cs,
                         σY=1, τ1=τ1, n=(1-n)/n, PLASTIC=True)

    for i in range(nx):
        Q = u[i]
        P = State(Q, MP)
        u[i] = Cvec(P.ρ, p, P.v, P.A, P.J, MP)


def hagen_poiseuille_exact(n, nx=200):

    Lx = 0.25
    dp = 0.48
    ρ = 1
    μ = 1e-2

    d = 1 / (2 * nx)
    x = linspace(d, Lx-d, nx)[int(nx/2):]

    k = (n + 1) / n
    y = ρ / k * (dp / μ)**(1 / n) * ((Lx / 2)**k - (x - Lx / 2)**k)
    return concatenate([flip(y, axis=0), y])
