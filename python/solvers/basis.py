from numpy import concatenate, eye, zeros
from numpy import polyder, polyint
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange

from options import N1


# The Legendre-Gauss nodes and weights, scaled to [0,1]
NODES, WGHTS = leggauss(N1)
NODES += 1
NODES /= 2
WGHTS /= 2

# The gaps between successive nodes
GAPS = NODES - concatenate(([0], NODES[:-1]))

# The basis polynomials
PSI = [lagrange(NODES, eye(N1)[i]) for i in range(N1)]

# The ith derivative of the jth basis polynomial
PSID = [[polyder(ψ, m=i) for ψ in PSI] for i in range(N1 + 1)]

# The integrals of the basis polynomials
PSII = [polyint(ψ) for ψ in PSI]

# The value of the ith basis function at j=0 and j=1
ENDVALS = zeros([N1, 2])
for i in range(N1):
    ENDVALS[i, 0] = PSI[i](0)
    ENDVALS[i, 1] = PSI[i](1)

# The value of the derivative of the jth basis function at the ith node
DERVALS = zeros([N1, N1])
for i in range(N1):
    for j in range(N1):
        DERVALS[i, j] = PSID[1][j](NODES[i])
