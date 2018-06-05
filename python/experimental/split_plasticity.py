from numpy import array, diag, dot, eye, log, ones, sign, trace
from numpy.linalg import det, eigvals
from numpy.random import rand
from scipy.integrate import quad
from scipy.optimize import newton_krylov

from gpr.misc.functions import dev
from gpr.misc.objects import material_params
from gpr.misc.plot import plot_energy, plot_distortion, plot_sigma, colors
from gpr.vars.eos import E_2A


A = rand(3, 3)
A *= sign(det(A))
G = dot(A.T, A)
σ = dot(G, dev(G))

x1, x2, x3 = eigvals(G) / det(A)**(2 / 3)

a = x1 + x2 + x3
b = x1**2 + x2**2 + x3**2
c = x1**3 + x2**3 + x3**3

norm0 = sqrt(1 / 2 * ((σ[0, 0] - σ[1, 1])**2 + (σ[1, 1] - σ[2, 2])**2 +
                      (σ[2, 2] - σ[0, 0])**2 + 6 * (σ[0, 1]**2 + σ[1, 2]**2 + σ[2, 0]**2)))

norm1 = sqrt(3 / 2) * norm(dev(σ))

print(norm0 - norm1)

tmp = sqrt(7 / 54 * a**4 - 2 / 3 * a**2 * b + b**2 / 6 + 2 / 3 * a * c)
norm2 = sqrt(3 / 2) * det(A)**(4 / 3) * tmp

print(norm1 - norm2)

m = (x1 + x2 + x3) / 3
u = ((x1 - x2)**2 + (x2 - x3)**2 + (x3 - x1)**2) / 3

#print(a - 3 * m)
#print(b - (u + 3 * m**2))
#print(c - (9 / 2 * m * u + 3 * ρ))

tmp = sqrt(1 / 6 * u**2 + 4 * m**2 * u - 6 * m**4 + 6 * m)
norm3 = sqrt(3 / 2) * det(A)**(4 / 3) * tmp

print(norm2 - norm3)

a = 9 * m - u - 9
b = 6 * m - u - 6
τ = rand()
e_τ = exp(-τ)
m = 1 + a / 3 * e_τ**6 - b / 3 * e_τ**9
u = 2 * a * e_τ**6 - 3 * b * e_τ**9

norm4 = 1 / 6 * u**2 + 4 * m**2 * u - 6 * m**4 + 6 * m
norm5 = 1 / 54 * (108 * a * e_τ**6 - 324 * b * e_τ**9 + 108 * a**2 * e_τ**12
                  - 396 * a * b * e_τ**15 + 297 * b**2 * e_τ**18
                  - 24 * a**2 * b * e_τ**21
                  + (48 * a * b**2 - 4 * a**4) * e_τ**24
                  + (16 * a**3 * b - 24 * b**3) * e_τ**27
                  - 24 * a**2 * b**2 * e_τ**30 + 16 * a * b**3 * e_τ**33
                  - 4 * b**4 * e_τ**36)

print(norm4 - norm5)

m = 1 + rand() / 10
u = rand()
a = 9 * m - u - 9
b = 6 * m - u - 6


def f1(τ):
    e_τ = exp(-τ)
    return 108 * a * e_τ**6 - 324 * b * e_τ**9 + 108 * a**2 * e_τ**12 - 396 * a * b * e_τ**15 + 297 * b**2 * e_τ**18 - 24 * a**2 * b * e_τ**21 + (48 * a * b**2 - 4 * a**4) * e_τ**24 + (16 * a**3 * b - 24 * b**3) * e_τ**27 - 24 * a**2 * b**2 * e_τ**30 + 16 * a * b**3 * e_τ**33 - 4 * b**4 * e_τ**36


def f2(τ):
    c = 108 * a - 324 * b + 108 * a**2 - 396 * a * b + 297 * \
        b**2 - 24 * (a**2 * b - 2 * a * b**2 + b**3) - 4 * (a - b)**4
    λ = c / (18 * a - 36 * b + 9 * a**2 - 132 / 5 * a * b + 33 / 2 * b**2 - 8 / 7 * a**2 * b + 2 * a * b **
             2 - 8 / 9 * b**3 - a**4 / 6 + 16 / 27 * a**3 * b - 4 / 5 * a**2 * b**2 + 16 / 33 * a * b**3 - b**4 / 9)
    return c * exp(-λ * τ)


print(quad(f1, 0, inf)[0] - quad(f2, 0, inf)[0])
x = linspace(0, 1, 100)
#plot(x, f1(x))
#plot(x, f2(x))


def obj(b, σ0, x):
    a, c, d = x
    return array([(1 + 1 / d)**(d + 1) - a * c,
                  a / (1 + b)**d + σ0 - a,
                  (a - σ0) * b * c * d - (1 + b)])


def bingham(a=1e2, b=1e9, σ0=1):
    d = log(a / (a - σ0)) / log(1 + b)
    c = (1 + b) / ((a - 1) * b * d)
    x = linspace(0, 4, 400)
    plot(x, a / (1 + b * exp(-c * x))**d - (a - σ0))


def test():

    MP = material_params('sg', 1, 1, 1, γ=1.4, b0=0.219, n=4, σY=9e-4, τ1=0.1)

    n = 100
    tf = 0.00001

    A = inv(array([[1, 0, 0],
                   [-0.01, 0.95, 0.02],
                   [-0.015, 0, 0.9]]))
    J = zeros(3)
    v = zeros(3)
    p = 1
    ρ = det(A) * MP.ρ0

    Q = Cvec(ρ, p, v, A, J, MP)
    u = array([[[Q]]])
    t = linspace(0, tf, n)
    ua = zeros([n, 1, 1, 17])
    un = zeros([n, 1, 1, 17])

    E0 = Q[1] / ρ - E_2A(ρ, A, MP)

    for i in range(n):
        dt = t[i]
        utmpa = u.copy()
        utmpn = u.copy()
        ode_stepper_analytical(utmpa, dt, MP)
        ode_stepper_numerical(utmpn, dt, MP)

        Aa = utmpa[0, 0, 0, 5:14].reshape([3, 3])
        An = utmpn[0, 0, 0, 5:14].reshape([3, 3])
        utmpa[0, 0, 0, 1] = ρ * (E0 + E_2A(ρ, Aa, MP))
        utmpn[0, 0, 0, 1] = ρ * (E0 + E_2A(ρ, An, MP))

        ua[i, 0, 0] = utmpa[0, 0, 0]
        un[i, 0, 0] = utmpn[0, 0, 0]

    cm = colors(3)

    plot_energy(un, x=t, col=cm[0])
    plot_energy(ua, x=t, col=cm[1], style='x')

    for i in range(3):
        plot_distortion(un, i, i, x=t, col=cm[0], fig=10)
        plot_distortion(ua, i, i, x=t, col=cm[1], style='x', fig=10)
        plot_sigma(un, i, i, [MP], x=t, col=cm[0], fig=11)
        plot_sigma(ua, i, i, [MP], x=t, col=cm[1], style='x', fig=11)
        j = (i+1) // 3
        plot_distortion(un, i, j, x=t, col=cm[0], fig=12)
        plot_distortion(ua, i, j, x=t, col=cm[1], style='x', fig=12)
        plot_distortion(un, j, i, x=t, col=cm[0], fig=12)
        plot_distortion(ua, j, i, x=t, col=cm[1], style='x', fig=12)
        plot_sigma(un, i, j, [MP], x=t, col=cm[0], fig=13)
        plot_sigma(ua, i, j, [MP], x=t, col=cm[1], style='x', fig=13)

    return ua, un
