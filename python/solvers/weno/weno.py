""" Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
"""
from itertools import product

from numpy import ceil, concatenate, floor, int64, multiply, ones, zeros
from scipy.linalg import solve

from solvers.weno.matrices import coefficient_matrices, oscillation_indicator
from options import rc, λc, λs, eps, ndim, N, N1, nV


Mc = coefficient_matrices()
Σ = oscillation_indicator()

fHalfN = int(floor(N/2))
cHalfN = int(ceil(N/2))


LAMS = [λs, λs, λc, λc]

if N==1:
    nStencils = 2
elif N%2:
    nStencils = 4
else:
    nStencils = 3


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension+1], ones(n-2), [extension+1])).astype(int64)
    return inarray.repeat(reps, axis=d)

def calculate_coeffs(uList):
    """ Calculate coefficients of basis polynomials and weights
    """
    wList = [solve(Mc[i], uList[i], overwrite_b=1, check_finite=0) for i in range(nStencils)]
    σList = [((w.T).dot(Σ).dot(w)).diagonal() for w in wList]
    oList = [LAMS[i]  / (abs(σList[i]) + eps)**rc for i in range(nStencils)]
    oSum = zeros(nV)
    numerator = zeros([N1, nV])
    for i in range(nStencils):
        oSum += oList[i]
        numerator += multiply(wList[i], oList[i])
    return numerator / oSum

def coeffs(ret, u1, u2, u3, u4):
    if N==1:
        ret[:] = calculate_coeffs([u1, u2])
    elif nStencils==3:
        ret[:] = calculate_coeffs([u1, u2, u3])
    else:
        ret[:] = calculate_coeffs([u1, u2, u3, u4])

def extract_stencils(arrayRow, ind):
    u1 = arrayRow[ind-N : ind+1]
    u2 = arrayRow[ind : ind+N+1]
    u3 = arrayRow[ind-cHalfN : ind+fHalfN+1]
    u4 = arrayRow[ind-fHalfN : ind+cHalfN+1]
    return u1, u2, u3, u4

def weno_launcher(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]

    Wx = zeros([nx, ny, nz, N1, nV])
    tempu = extend(u, N, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        u1, u2, u3, u4 = extract_stencils(tempu[:,j,k], i+N)
        coeffs(Wx[i, j, k], u1, u2, u3, u4)

    if ny==1:
        return Wx

    Wxy = zeros([nx, ny, nz, N1, N1, nV])
    tempWx = extend(Wx, N, 1)
    for i, j, k, a in product(range(nx), range(ny), range(nz), range(N1)):
        u1, u2, u3, u4 = extract_stencils(tempWx[i,:,k,a], j+N)
        coeffs(Wxy[i, j, k, a], u1, u2, u3, u4)

    if nz==1:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N1, N1, N1, nV])
    tempWxy = extend(Wxy, N, 2)
    for i, j, k, a, b in product(range(nx), range(ny), range(nz), range(N1), range(N1)):
        u1, u2, u3, u4 = extract_stencils(tempWxy[i,j,:,a,b], k+N)
        coeffs(Wxyz[i, j, k, a, b], u1, u2, u3, u4)

    return Wxyz


######  EXPERIMENTAL  ######

from numpy import dot

from solvers.basis import mid_values
from system.gpr.misc.structures import Cvec_to_Pvec

MIDVALS = mid_values()

def weno_primitive(q, PAR):
    """ Returns a WENO reconstruction in primitive variables, given the grid of
        conserved values. A reconstruction in conserved variables is performed.
        The midpoints of this reconstruction are then taken as the conserved
        cell averages (required for >2nd order). A primitive reconstruction is
        then performed with these averages.
    """
    qr = weno(q)
    qav = dot(MIDVALS, qr)
    pav = zeros(qav.shape)
    nx, ny, nz = qav.shape[:3]
    for i, j, k in product(range(nx), range(ny), range(nz)):
        pav[i,j,k] = Cvec_to_Pvec(qav[i,j,k], PAR)
    return weno(pav)

def expand_weno(wh):
    nx, ny, _, N1, _, nvar = wh.shape
    ret = zeros([nx*N1, ny*N1, nvar])
    for i in range(nx):
        for j in range(ny):
            for ii in range(N1):
                for jj in range(N1):
                    indi = i*N1+ii
                    indj = j*N1+jj
                    ret[indi,indj] = wh[i,j,0,ii,jj]
    return ret
