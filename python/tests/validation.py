from numpy import array, eye, zeros, arange, around, concatenate, exp, int64, ones, sqrt
from scipy.optimize import brentq
from scipy.special import erf

from auxiliary.common import material_parameters
from gpr.functions import conserved, primitive
from gpr.variables import c_0
from options import nx, ny, nz, Ms, dx, Rc, L


def first_stokes_problem_IC():
    """ tf = 1
        L = 1
    """
    y = 1.4
    mu = 1e-2 # 1e-3 # 1e-4

    r = 1
    p = 1 / y
    v = array([0, 0.1, 0])
    A = eye(3)
    J = zeros(3)

    params = material_parameters(y=y, pINF=0, cv=1, r0=r, p0=p, cs=1, alpha=1e-16, mu=mu, Pr=0.75)
    QL = conserved(r, p, -v, A, J, 0, params, 1, 0, 0)
    QR = conserved(r, p,  v, A, J, 0, params, 1, 0, 0)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i*dx < L/2:
            u[i,0,0] = QL
        else:
            u[i,0,0] = QR

    return u, [params]*1, []

def first_stokes_problem_exact(x, mu, v0=0.1, t=1):
    return v0 * erf(x / (2 * sqrt(mu * t)))

def viscous_shock_IC():
    CENTER = 1
    y = 1.4
    pINF = 0
    r0 = 1
    p0 = 1 / y
    mu = 2e-2

    params = material_parameters(y=y, pINF=0, cv=2.5, r0=r0, p0=p0,
                                 cs=5, alpha=5, mu=2e-2, Pr=0.75)

    if Ms==2:
        x0 = 0.07   # Position of center of shock for shock to start at x = 0
        l = 0.3
    elif Ms==3:
        x0 = 0.04   # Position of center of shock for shock to start at x = 0
        l = 0.13

    c0 = c_0(r0, p0, y, pINF)
    a = 2 / (Ms**2 * (y+1)) + (y-1)/(y+1)

    Re = r0 * c0 * Ms / mu
    c1 = ((1-a)/2)**(1-a)
    c2 = 3/4 * Re * (Ms**2-1) / (y*Ms**2)

    # Morduchow's formula
#    d = 0.35
#    k = 3/(8*d) * (y+1) * sqrt(pi/(8*y))
#    c1 = (1-sqrt(a)) / (sqrt(a)-a)**a
#    c2 = k * (1-a) * Ms

    x = around(arange(-l, l, dx), decimals=14)
    n = x.size
    vbar = zeros(n)
    for i in range(n):
        f = lambda v: (1-v)/(v-a)**a - c1 * exp(c2*-x[i])
        vbar[i] = brentq(f, a+1e-16, 1)

    # Dumbser's formula
#    pbar = 1-vbar + 1/(2*y) * (y+1)/(y-1) * (vbar-1)/vbar * (vbar-a)
#    p = r0 * c0**2 * Ms**2 * pbar + p0

    p = p0 / vbar * (1 + (y-1)/2 * Ms**2 * (1-vbar**2))
    r = r0 / vbar
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
        r = r.repeat(reps.astype(int64))
    else:
        x = x + x0
        p = p[x>=0]
        r = r[x>=0]
        v = v[x>=0]
        x = x[x>=0]

        n = x.size
        reps = ones(n)
        reps[-1] = nx - n + 1
        v = v.repeat(reps.astype(int64))
        p = p.repeat(reps.astype(int64))
        r = r.repeat(reps.astype(int64))
        x = concatenate((x, arange(l+x0, 1, dx)))

    u = zeros([nx, 1, 1, 18])
    for i in range(nx):
        A = (r[i])**(1/3) * eye(3)
        J = zeros(3)
        c = 1
        u[i,0,0] = conserved(r[i], p[i], array([v[i], 0, 0]), A, J, c, params, 1, 1, 0)

    return u, [params], []

def viscous_shock_exact_x(n, M=2, t=0.2):
    return arange(M*t-0.25, M*t+0.75, 1/n)

def heat_conduction_IC():
    rL = 2
    rR = 0.5
    p0 = 1
    v0 = zeros(3)
    AL = rL**(1/3) * eye(3)
    AR = rR**(1/3) * eye(3)
    J0 = zeros(3)

    params = material_parameters(y=1.4, pINF=0, cv=2.5, r0=1, p0=p0,
                                 cs=1, alpha=2, mu=1e-2, kappa=1e-2)
    QL = conserved(rL, p0, v0, AL, J0, 0, params, 1, 1, 0)
    QR = conserved(rR, p0, v0, AR, J0, 0, params, 1, 1, 0)
    u = zeros([nx, ny, nz, 18])
    x0 = L / 2
    for i in range(nx):
        if i*dx < x0:
            u[i,0,0] = QL
        else:
            u[i,0,0] = QR

    return u, [params]*2, [x0]

def semenov_IC():
    cv = 2.5
    T0 = 1
    Qc = 4
    eps = 1/20 # 1/15 # 1/10
    Ea = Rc * 1 / eps
    Bc = (cv * T0**2 * Rc) / (Ea * Qc) * exp(Ea/(Rc*T0))

    r = 1
    p = 1
    v = zeros(3)
    A = r**(1/3) * eye(3)
    J = zeros(3)
    c = 1

    params = material_parameters(y=1.4, pINF=0, cv=cv, r0=r, p0=p, Qc=Qc, epsilon=eps, Bc=Bc)
    Q = conserved(r, p, v, A, J, c, params, 0, 1, 1)
    u = zeros([nx, ny, nz, 18])
    for i in range(nx):
        u[i,0,0] = Q

    return u, [params], []

def semenov_temp(dataArray, params):
    states = [da[0, 0, 0] for da in dataArray]
    return [primitive(state, params, 0, 0, 1).T for state in states]
