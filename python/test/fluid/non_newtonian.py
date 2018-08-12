from numpy import amax, array, concatenate, eye, flip, linspace, zeros
from scipy.integrate import odeint

from ader.etc.boundaries import standard_BC

from gpr.misc.objects import material_params
from gpr.misc.plot import colors, plot_energy, plot_distortion, plot_sigma
from gpr.misc.structures import Cvec
from gpr.sys.analytical import ode_solver_cons
from gpr.sys.conserved import S_cons

from test.boundaries import wall_BC


HP_n = 1.5
μ = 1e-2
dp = 0.48
Lx = 0.25


def poiseuille_exact(n, nx=400):

    ρ = 1
    d = Lx / (2 * nx)
    x = linspace(d, Lx-d, nx)[int(nx/2):]

    k = (n + 1) / n
    y = ρ / k * (dp / μ)**(1 / n) * ((Lx / 2)**k - (x - Lx / 2)**k)
    return concatenate([flip(y, axis=0), y])


def poiseuille_max(n):
    return amax(poiseuille_exact(n))


def poiseuille_average(n):
    ρ = 1
    k = (n + 1) / n
    return ρ / k * (dp / μ)**(1 / n) * 2**(-k) * k * Lx**k / (k+1)


def reynolds_number(n):
    ρ = 1
    return Lx * poiseuille_average(n) * ρ / μ


def poiseuille():
    """ N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'rusanov'
    """
    tf = 20
    nx = 400

    γ = 1.4
    ρ = 1
    p = 100 / γ
    #vy = poiseuille_max(HP_n)
    v = array([0, 0, 0])
    A = eye(3)
    δp = array([0, dp, 0])

    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ,
                         b0=1, μ=μ, n=HP_n, δp=δp)

    Q = Cvec(ρ, p, v, MP, A)
    u = array([Q] * nx)

    return u, [MP], tf, [Lx / nx]


def poiseuille_bc(u, N, *args):
    γ = 1.4
    δp = array([0, dp, 0])
    MP = material_params(EOS='sg', ρ0=1, cv=1, p0=100/γ, γ=γ,
                         b0=1, μ=μ, n=HP_n, δp=δp)
    return wall_BC(u, N, 1, [1], MP)


def lid_driven_cavity():

    tf = 2
    Lx = 1
    Ly = 1
    nx = 100
    ny = 100

    γ = 1.4

    ρ = 1
    p = 100 / γ
    v = zeros(3)
    A = eye(3)

    MP = material_params(EOS='sg', ρ0=ρ, cv=1, p0=p, γ=γ, b0=8, μ=1e-2, n=0.5)

    u = zeros([nx, ny, 14])
    Q = Cvec(ρ, p, v, MP, A)
    for i in range(nx):
        for j in range(ny):
            u[i, j] = Q

    print("LID-DRIVEN CAVITY")
    return u, [MP], tf, [Lx / nx, Ly / ny]


def lid_driven_cavity_bc(u, N, NDIM):

    ret = standard_BC(u, N, NDIM, wall=[True, True], reflectVars=[2, 3, 4])
    nx, ny = ret.shape[:2]

    for i in range(nx):
        for j in range(N):
            jj = 2 * N - 1 - j
            v = 2 - ret[i, jj, 2] / ret[i, jj, 0]
            ret[i, j, 2] = ret[i, j, 0] * v

    return ret


def strain_relaxation(n=4, tf = 0.00001):

    def f(Q, t, MP):
        return S_cons(Q, MP)

    MP = material_params('sg', 1, 1, 1, γ=1.4, b0=0.219, n=n, σY=9e-4, τ1=0.1)
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
