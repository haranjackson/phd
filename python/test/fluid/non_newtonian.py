from numpy import amax, array, concatenate, eye, flip, linspace, zeros
from scipy.integrate import odeint

from gpr.misc.objects import material_params
from gpr.misc.structures import Cvec
from gpr.sys.analytical import ode_solver_cons
from gpr.sys.conserved import S_cons
from plot import colors, plot_energy, plot_distortion, plot_sigma


def poiseuille_exact(n, nx=400, Lx=0.25, μ=1e-2, dp=0.48):

    ρ = 1
    d = Lx / (2 * nx)
    x = linspace(d, Lx-d, nx)[int(nx/2):]

    k = (n + 1) / n
    y = ρ / k * (dp / μ)**(1 / n) * ((Lx / 2)**k - (x - Lx / 2)**k)
    return concatenate([flip(y, axis=0), y])


def poiseuille_max(n):
    return amax(poiseuille_exact(n))


def poiseuille_average(n, μ, Lx, dp):
    ρ = 1
    k = (n + 1) / n
    return ρ / k * (dp / μ)**(1 / n) * 2**(-k) * k * Lx**k / (k+1)


def reynolds_number(n):
    ρ = 1
    return Lx * poiseuille_average(n) * ρ / μ


def poiseuille():
    """ n = 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3

        N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'rusanov'
        contorted_tol = 1

        DESTRESS = false
    """
    n = 0.8
    BINGHAM = True

    tf = 10
    Lx = 0.25
    nx = 200

    γ = 1.4
    μ = 1e-2
    dp = 0.48

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)
    δp = array([0, dp, 0])

    if BINGHAM:
        σY = Bi * μ / Lx * poiseuille_average(1, μ, Lx, dp)
        MP = material_params(EOS='sg', ρ0=ρ, cv=1, γ=γ,
                             b0=1, bf=1, bs=1, μ=μ, σY=σY, δp=δp)
    else:
        MP = material_params(EOS='sg', ρ0=ρ, cv=1, γ=γ, b0=1, μ=μ, n=n, δp=δp)

    Q = Cvec(ρ, p, v, MP, A)
    u = array([Q] * nx)

    return u, [MP], tf, [Lx / nx], 'stick'


def lid_driven_cavity():
    """ n = 0.5 / 1.5

        N = 3
        cfl = 0.5 / 0.2
        SPLIT = True
        SOLVER = 'rusanov'
        contorted_tol = 0.03

        NO_CORNERS = true
        DESTRESS = true
    """
    n = 0.5

    tf = 40
    Lx = 1
    Ly = 1
    nx = 100
    ny = 100

    γ = 1.4
    μ = 1e-2

    ρ = 1
    p = 1
    v = zeros(3)
    A = eye(3)

    MP = material_params(EOS='sg', ρ0=ρ, cv=2.5, γ=γ, b0=1, μ=μ, n=n)

    Q = Cvec(ρ, p, v, MP, A)
    u = zeros([nx, ny, 14])
    for i in range(nx):
        for j in range(ny):
            u[i, j] = Q

    print("LID-DRIVEN CAVITY")
    return u, [MP], tf, [Lx / nx, Ly / ny], 'lid_driven'


def strain_relaxation(n=4, tf=0.00001):

    def f(Q, t, MP):
        return S_cons(Q, MP)

    MP = material_params('sg', 1, 1, 1, γ=1.4, b0=0.219, n=n, σY=9e-4, τ0=0.1)
    MPs = [MP]

    A = inv(array([[1, 0, 0],
                   [-0.01, 0.95, 0.02],
                   [-0.015, 0, 0.9]]))

    ρ = det(A) * MP.ρ0
    p = 1
    v = zeros(3)
    Q = Cvec(ρ, p, v, MP, A)

    N = 100
    t = linspace(0, tf, N)
    ua = array([Q for i in range(N)])
    un = zeros([N, 14])

    for i in range(N):
        dt = t[i]
        ode_solver_cons(ua[i], dt, MP)
        un[i] = odeint(f, Q.copy(), array([0, dt]), args=(MP,))[1]

    cm = colors(3)

    plot_energy(un, MPs, x=t, col=cm[0])
    plot_energy(ua, MPs, x=t, col=cm[1], style='x')

    for i in range(3):
        plot_distortion(un, MPs, i, i, x=t, col=cm[0], fig=10)
        plot_distortion(ua, MPs, i, i, x=t, col=cm[1], style='x', fig=10)
        plot_sigma(un, MPs, i, i, x=t, col=cm[0], fig=11)
        plot_sigma(ua, MPs, i, i, x=t, col=cm[1], style='x', fig=11)
        j = (i+1) // 3
        plot_distortion(un, MPs, i, j, x=t, col=cm[0], fig=12)
        plot_distortion(ua, MPs, i, j, x=t, col=cm[1], style='x', fig=12)
        plot_distortion(un, MPs, j, i, x=t, col=cm[0], fig=12)
        plot_distortion(ua, MPs, j, i, x=t, col=cm[1], style='x', fig=12)
        plot_sigma(un, MPs, i, j, x=t, col=cm[0], fig=13)
        plot_sigma(ua, MPs, i, j, x=t, col=cm[1], style='x', fig=13)

    return ua, un
