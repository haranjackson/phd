from numpy import arctan, array, cos, dot, exp, log, pi, prod, sqrt
from numpy.linalg import svd

from gpr.misc.functions import L2_1D
from gpr.opts import VISCOUS, THERMAL
from gpr.vars.eos import E_2A, E_3
from gpr.vars.shear import c_s2


### DISTORTION ###


def bound_f(x, l):
    tmp1 = log((x**2 + l * x + l**2) / (x - l)**2)
    tmp2 = 2 * sqrt(3) * arctan((2 * x + l) / (sqrt(3) * l))
    return tmp1 - tmp2


def pos(x):
    return max(0, x)


def nondimensionalized_time(ρ, detA3, m0, u0, dt, MP):

    τ1 = MP.τ1
    if hasattr(MP,'n') and MP.n != 0:
        n = MP.n
        σY = MP.σY
        cs2 = c_s2(ρ, MP)
        a = 9 * m0 - u0 - 9
        b = 6 * m0 - u0 - 6
        c = (108 * a - 324 * b + 108 * a**2 - 396 * a * b + 297 * b**2
             - 24 * (a**2 * b - 2 * a * b**2 + b**3) - 4 * (a - b)**4)
        if c <= 0:
            return 0

        λ = c / (18 * a - 36 * b + 9 * a**2 - 132 / 5 * a * b + 33 / 2 * b**2
                 - 8 / 7 * a**2 * b + 2 * a * b ** 2 - 8 / 9 * b**3 - a**4 / 6
                 + 16 / 27 * a**3 * b - 4 / 5 * a**2 * b**2 + 16 / 33 * a * b**3
                 - b**4 / 9)
        tmp = (sqrt(c) * ρ * cs2 / (6 * σY))**n
        return 2 / (n * λ) * log(n * λ / τ1 * detA3**(4 * n + 7) * tmp * dt + 1)
    else:
        return 2 * detA3**7 / τ1 * dt


def solver_distortion_analytic(A, dt, MP):

    U, s, V = svd(A)
    ρ_ρ0 = prod(s)
    ρ = ρ_ρ0 * MP.ρ0
    detA3 = ρ_ρ0**(1 / 3)

    s0 = (s / detA3)**2
    m0 = sum(s0) / 3
    u0 = ((s0[0] - s0[1])**2 + (s0[1] - s0[2])**2 + (s0[2] - s0[0])**2) / 3

    if u0 < 1e-12:
        return A

    τ = nondimensionalized_time(ρ, detA3, m0, u0, dt, MP)
    e_6τ = exp(-6 * τ)
    e_9τ = exp(-9 * τ)
    m = 1 + (3 * m0 - u0 / 3 - 3) * e_6τ - (2 * m0 - u0 / 3 - 2) * e_9τ
    u = pos((18 * m0 - 2 * u0 - 18) * e_6τ - (18 * m0 - 3 * u0 - 18) * e_9τ)

    Δ = -2 * m**3 + m * u + 2
    arg1 = pos(6 * u**3 - 81 * Δ**2)

    if arg1 < 1e-12:
        θ = 0
    elif abs(Δ) < 1e-12:
        θ = sign(Δ) * pi / 2
    else:
        θ = arctan(sqrt(arg1) / (9 * Δ))

    x1 = sqrt(6 * u) / 3 * cos(θ / 3) + m
    temp2 = 3 * m - x1
    arg2 = pos(x1 * temp2**2 - 4)
    x2 = 0.5 * (sqrt(arg2 / x1) + temp2)
    x3 = 1 / (x1 * x2)

    s1 = detA3 * sqrt(array([x1, x2, x3]))
    s1[::-1].sort()
    return dot(U * s1, V)


### THERMAL IMPULSE ###


def solver_thermal_analytic(ρ, E, A, J, v, dt, MP):
    """ Solves the thermal impulse ODE analytically in 3D
    """
    c1 = E - E_2A(ρ, A, MP) - E_3(v)
    c2 = MP.cα2 / 2
    k = 2 * MP.ρ0 / (MP.τ2 * MP.T0 * ρ * MP.cv)
    a = c1 * k
    b = c2 * k

    # To avoid NaNs if dt>>1
    ea = exp(-a * dt / 2)
    den = 1 - b / a * (1 - ea**2) * L2_1D(J)
    ret = J / sqrt(den)
    return ea * ret


### GENERAL ###


def ode_solver_cons(Q, dt, MP):
    """ Solves the ODE analytically by approximating the distortion equations
        and solving the thermal impulse equations
    """
    ρ = Q[0]
    A = Q[5:14].reshape([3, 3])

    if VISCOUS:
        A1 = solver_distortion_analytic(A, dt, MP)
        Q[5:14] = A1.ravel()

    if THERMAL:
        J = Q[14:17] / ρ
        E = Q[1] / ρ
        v = Q[2:5] / ρ
        A2 = (A + A1) / 2
        Q[14:17] = ρ * solver_thermal_analytic(ρ, E, A2, J, v, dt, MP)
