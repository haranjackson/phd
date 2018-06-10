from numpy import amax, array, concatenate, eye, flip, linspace, zeros

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec
from gpr.tests.boundaries import wall_BC


HP_n = 1.3


def poiseuille_exact(n, nx=400):

    Lx = 0.25
    dp = 0.48
    ρ = 1
    μ = 1e-2

    d = Lx / (2 * nx)
    x = linspace(d, Lx-d, nx)[int(nx/2):]

    k = (n + 1) / n
    y = ρ / k * (dp / μ)**(1 / n) * ((Lx / 2)**k - (x - Lx / 2)**k)
    return concatenate([flip(y, axis=0), y])


def poiseuille_max(n):
    return amax(poiseuille_exact(n))


def poiseuille_average(n):

    Lx = 0.25
    dp = 0.48
    ρ = 1
    μ = 1e-2
    k = (n + 1) / n

    return ρ / k * (dp / μ)**(1 / n) * 2**(-k) * k * Lx**k / (k+1)


def poiseuille_IC():
    """ N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'rusanov'
    """
    tf = 3
    Lx = 0.25
    nx = 400
    dp = 0.48

    γ = 1.4
    ρ = 1
    p = 100 / γ
    vy = poiseuille_max(HP_n)
    v = array([0, vy, 0])
    A = eye(3)
    J = zeros(3)
    δp = array([0, dp, 0])

    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ,
                         b0=1, μ=1e-2, n=HP_n, δp=δp)

    Q = Cvec(ρ, p, v, A, J, MP)
    u = array([Q] * nx)

    return u, [MP], tf, [Lx / nx]


def poiseuille_BC(u, N, *args):
    dp = 0.48
    γ = 1.4
    δp = array([0, dp, 0])
    MP = material_params(EOS='sg', ρ0=1, cv=1, p0=100/γ, γ=γ,
                         b0=1, μ=1e-2, n=HP_n, δp=δp)
    return wall_BC(u, N, 1, [1], MP)
