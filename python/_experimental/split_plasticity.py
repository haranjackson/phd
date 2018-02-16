from numpy import array, diag, dot, eye, log, trace
from numpy.random import rand


def dev(X):
    return X - trace(X) / 3 * eye(3)


x = rand()
y = rand()
z = rand()

a = x + y + z
b = x**2 + y**2 + z**2
c = x**3 + y**3 + z**3

G = diag(array([x, y, z]))
σ = dot(G, dev(G))

norm1 = norm(dev(σ))**2
norm2 = 7 / 54 * a**4 - 2 / 3 * a**2 * b + b**2 / 6 + 2 / 3 * a * c

print(norm1 - norm2)

m = (x + y + z) / 3
u = ((x - y)**2 + (y - z)**2 + (z - x)**2) / 3
ρ = x * y * z

print(a - 3 * m)
print(b - (u + 3 * m**2))
print(c - (9 / 2 * m * u + 3 * ρ))

norm3 = 1 / 6 * u**2 + 4 * m**2 * u - 6 * m**4 + 6 * m * ρ

print(norm1 - norm3)


def g(b, σ0, x):
    a, c, d = x
    return array([(1 + 1 / d)**(d + 1) - a * c,
                  a / (1 + b)**d + σ0 - a,
                  (a - σ0) * b * c * d - (1 + b)])

def f(a=2, b=1, σ0=1):

    from scipy.optimize import newton_krylov
    from numpy import ones

    d = log(a / (a - σ0)) / log(1 + b)
    c = (1 + b) / ((a - 1) * b * d)

    #g2 = lambda x: g(b, σ0, x)
    #a, c, d = newton_krylov(g2, array([a,c,d]))

    x = linspace(0, 4, 400)
    plot(x, a / (1 + b * exp(-c * x))**d - (a - 1))
    #plot(linspace(0, 1, 100), linspace(0, 1, 100))
    #plot(x, ones(400))
