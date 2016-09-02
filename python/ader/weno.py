""" Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
"""
from numpy import concatenate, int64, multiply, ones, zeros, floor, ceil
from scipy.linalg import solve

from options import rc, lam, lams, eps, ndim, N, N1
from ader.weno_matrices import coefficient_matrices, oscillation_indicator


Mc = coefficient_matrices()
SIGMA = oscillation_indicator()

if N%2:
    nStencils = 4
    lamList = [lam, lam, lams, lams]
else:
    nStencils = 3
    lamList = [lam, lams, lams]


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension+1], ones(n-2), [extension+1])).astype(int64)
    return inarray.repeat(reps, axis=d)

def coeffs(uList):
    """ Calculate coefficients of basis polynomials and weights
    """
    wList = [solve(Mc[i], uList[i], overwrite_b=1, check_finite=0) for i in range(nStencils)]
    sigmaList = [((w.T).dot(SIGMA).dot(w)).diagonal() for w in wList]
    omegaList = [lamList[i]  / (abs(sigmaList[i]) + eps)**rc for i in range(nStencils)]
    omegaSum = zeros(18)
    numerator = zeros([N1, 18])
    for i in range(nStencils):
        omegaSum += omegaList[i]
        numerator += multiply(wList[i],omegaList[i])
    return numerator / omegaSum

def reconstruct(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]
    floorHalfN = int(floor(N/2))
    ceilHalfN = int(ceil(N/2))

    Wx = zeros([nx, ny, nz, N1, 18])
    tempu = extend(u, N, 0)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ii = i + N
                if nStencils==3:
                    u1 = tempu[ii-floorHalfN : ii+floorHalfN+1, j, k]
                    u2 = tempu[ii-N : ii+1, j, k]
                    u3 = tempu[ii : ii+N+1, j, k]
                    Wx[i, j, k] = coeffs([u1, u2, u3])
                else:
                    u1 = tempu[ii-floorHalfN : ii+ceilHalfN+1, j, k]
                    u2 = tempu[ii-ceilHalfN : ii+floorHalfN+1, j, k]
                    u3 = tempu[ii-N : ii+1, j, k]
                    u4 = tempu[ii : ii+N+1, j, k]
                    Wx[i, j, k] = coeffs([u1, u2, u3, u4])
    if ndim==1:
        return Wx

    Wxy = zeros([nx, ny, nz, N1, N1, 18])
    tempWx = extend(Wx, N, 1)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                jj = j + N
                for a in range(N1):
                    if nStencils==3:
                        w1 = tempWx[i, jj-floorHalfN : jj+floorHalfN+1, k, a]
                        w2 = tempWx[i, jj-N : jj+1, k, a]
                        w3 = tempWx[i, jj : jj+N+1, k, a]
                        Wxy[i, j, k, a] = coeffs([w1, w2, w3])
                    else:
                        w1 = tempWx[i, jj-floorHalfN : jj+ceilHalfN+1, k, a]
                        w2 = tempWx[i, jj-ceilHalfN : jj+floorHalfN+1, k, a]
                        w3 = tempWx[i, jj-N : jj+1, k, a]
                        w4 = tempWx[i, jj : jj+N+1, k, a]
                        Wxy[i, j, k, a] = coeffs([w1, w2, w3, w4])
    if ndim==2:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N1, N1, N1, 18])
    tempWxy = extend(Wxy, N, 2)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                kk = k + N
                for a in range(N1):
                    for b in range(N1):
                        if nStencils==3:
                            w1 = tempWxy[i, j, kk-floorHalfN : kk+floorHalfN+1, a, b]
                            w2 = tempWxy[i, j, kk-N : kk+1, a, b]
                            w3 = tempWxy[i, j, kk : kk+N+1, a, b]
                            Wxyz[i, j, k, a, b] = coeffs([w1, w2, w3])
                        else:
                            w1 = tempWxy[i, j, kk-floorHalfN : kk+ceilHalfN+1, a, b]
                            w2 = tempWxy[i, j, kk-ceilHalfN : kk+floorHalfN+1, a, b]
                            w3 = tempWxy[i, j, kk-N : kk+1, a, b]
                            w4 = tempWxy[i, j, kk : kk+N+1, a, b]
                            Wxyz[i, j, k, a, b] = coeffs([w1, w2, w3, w4])
    if ndim==3:
        return Wxyz
