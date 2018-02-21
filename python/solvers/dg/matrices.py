from itertools import product

from numpy import concatenate, diag, eye, ones, zeros

from solvers.basis import NODES, WGHTS, PSI, PSID, DERVALS
from gpr.misc.functions import kron_prod
from options import NDIM, N, NT, NV


# Inner products required for Galerkin matrices
I11 = zeros([N, N])                           # I11[a,b] = ψ_a(1) * ψ_b(1)
I1 = zeros([N, N])                            # I1[a,b] = ψ_a • ψ_b
I2 = zeros([N, N])                            # I2[a,b] = ψ_a • ψ_b'
I = eye(N)
for a, b in product(range(N), range(N)):
    I11[a, b] = PSI[a](1) * PSI[b](1)
    if a == b:
        I1[a, b] = WGHTS[a]
        I2[a, b] = (PSI[a](1)**2 - PSI[a](0)**2) / 2
    else:
        I2[a, b] = WGHTS[a] * PSID[1][b](NODES[a])


# Galerkin matrices
DG_W = concatenate([PSI[a](0) * kron_prod([I1] * NDIM) for a in range(N)])

DG_U = kron_prod([I11 - I2.T] + [I1] * NDIM)

DG_V = zeros([NDIM, NT, NT])
for i in range(1, NDIM + 1):
    DG_V[i - 1] = kron_prod([I1] * i + [I2] + [I1] * (NDIM - i))

DG_Z = (diag(kron_prod([I1] * (NDIM + 1))) * ones([NV, NT])).T

DG_T = zeros([NDIM, NT, NT])
for i in range(1, NDIM + 1):
    DG_T[i - 1] = kron_prod([I] * i + [DERVALS] + [I] * (NDIM - i))
