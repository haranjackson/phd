from numpy import array, diag, dot, eye, trace
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
