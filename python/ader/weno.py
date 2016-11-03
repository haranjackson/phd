""" Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
"""
from itertools import product

from numpy import ceil, concatenate, dot, floor, int64, multiply, ones, tensordot, zeros
from scipy.linalg import solve

from options import rc, λc, λs, eps, ndim, N, N1
from ader.basis import mid_values, end_values, derivative_end_values
from ader.weno_matrices import coefficient_matrices, oscillation_indicator
from gpr.variables.vectors import Cvec_to_Pvec


Mc = coefficient_matrices()
Σ = oscillation_indicator()
midvals = mid_values()


if N%2:
    nStencils = 4
    lamList = [λc, λc, λs, λs]
else:
    nStencils = 3
    lamList = [λc, λs, λs]


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
    σList = [((w.T).dot(Σ).dot(w)).diagonal() for w in wList]
    oList = [lamList[i]  / (abs(σList[i]) + eps)**rc for i in range(nStencils)]
    oSum = zeros(18)
    numerator = zeros([N1, 18])
    for i in range(nStencils):
        oSum += oList[i]
        numerator += multiply(wList[i], oList[i])
    return numerator / oSum

def weno(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]
    floorHalfN = int(floor(N/2))
    ceilHalfN = int(ceil(N/2))

    Wx = zeros([nx, ny, nz, N1, 18])
    tempu = extend(u, N, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
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
    for i, j, k in product(range(nx), range(ny), range(nz)):
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
    for i, j, k in product(range(nx), range(ny), range(nz)):
        kk = k + N
        for a, b in product(range(N1), range(N1)):
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

def weno_primitive(q, PAR, SYS):
    """ Returns a WENO reconstruction in primitive variables, given the grid of conserved values.
        A reconstruction in conserved variables is performed. The midpoints of this reconstruction
        are then taken as the conserved cell averages (required for >2nd order). A primitive
        reconstruction is then performed with these averages.
    """
    qr = weno(q)
    qav = dot(midvals, qr)
    pav = zeros(qav.shape)
    nx, ny, nz = qav.shape[:3]
    for i, j, k in product(range(nx), range(ny), range(nz)):
        pav[i,j,k] = Cvec_to_Pvec(qav[i,j,k], PAR, SYS)
    return weno(pav)

def weno_endpoints(wh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    endVals = end_values()
    derEndVals = derivative_end_values()
    nx, ny, nz = wh.shape[:3]
    wh0 = wh.reshape([nx, ny, nz] + [N1]*ndim + [18])
    wEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*(ndim-1) + [18])
    wDerEnd = zeros([ndim, 2, nx, ny, nz] + [N1]*(ndim-1) + [18])
    for d in range(ndim):
        wEnd[d] = tensordot(endVals, wh0, (0,3+d))
        wDerEnd[d] = tensordot(derEndVals, wh0, (0,3+d))
    return wEnd, wDerEnd
