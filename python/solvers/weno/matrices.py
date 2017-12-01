from numpy import ceil, floor, polyint, zeros

from solvers.basis import PSID, PSII
from options import N, N1


fHalfN = int(floor(N/2))
cHalfN = int(ceil(N/2))


# The linear systems governing the coefficients of the basis polynomials
WN_M = zeros([4, N1, N1])
for e in range(N1):
    for p in range(N1):
        ψ = PSII[p]
        WN_M[0,e,p] = ψ(e-N1+2) - ψ(e-N1+1)
        WN_M[1,e,p] = ψ(e+1) - ψ(e)
        WN_M[2,e,p] = ψ(e-cHalfN+1) - ψ(e-cHalfN)
        WN_M[3,e,p] = ψ(e-fHalfN+1) - ψ(e-fHalfN)

# The oscillation indicator
WN_Σ = zeros([N1, N1])
for p in range(N1):
    for m in range(N1):
        for a in range(1,N1):
            ψa = PSID[a]
            ψ = polyint(ψa[p] * ψa[m])
            WN_Σ[p,m] += ψ(1) - ψ(0)
