""" Implements WENO method used in Dumbser et al, DOI 10.1016/j.cma.2013.09.022
"""
from itertools import product

from numpy import ceil, concatenate, floor, int64, multiply, ones, zeros
from scipy.linalg import solve

from solvers.weno.matrices import fHalfN, cHalfN, WN_M, WN_Σ
from options import rc, λc, λs, eps, ndim, N, N1, nV


LAMS = [λs, λs, λc, λc]


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension+1], ones(n-2), [extension+1])).astype(int64)
    return inarray.repeat(reps, axis=d)

def calculate_coeffs(uList, L=0, R=0):
    """ Calculate coefficients of basis polynomials and weights
    """
    n = len(uList)
    if L:
        wList = [solve(WN_M[0], uList[0], overwrite_b=1, check_finite=0)]
    elif R:
        wList = [solve(WN_M[1], uList[0], overwrite_b=1, check_finite=0)]
    else:
        wList = [solve(WN_M[i], uList[i], overwrite_b=1, check_finite=0)
                 for i in range(n)]

    σList = [((w.T).dot(WN_Σ).dot(w)).diagonal() for w in wList]
    oList = [LAMS[i]  / (abs(σList[i]) + eps)**rc for i in range(n)]
    oSum = zeros(nV)
    numerator = zeros([N1, nV])
    for i in range(n):
        oSum += oList[i]
        numerator += multiply(wList[i], oList[i])
    return numerator / oSum

def coeffs(ret, uL, uR, uCL, uCR, L=0, R=0):

    if L:
        ret[:] = calculate_coeffs([uL], L=1)
    elif R:
        ret[:] = calculate_coeffs([uR], R=1)

    if N==1:
        ret[:] = calculate_coeffs([uL, uR])
    elif not N%2:
        ret[:] = calculate_coeffs([uL, uR, uCL])
    else:
        ret[:] = calculate_coeffs([uL, uR, uCL, uCR])

def extract_stencils(arrayRow, ind):
    uL = arrayRow[ind-N : ind+1]
    uR = arrayRow[ind : ind+N+1]
    uCL = arrayRow[ind-cHalfN : ind+fHalfN+1]
    uCR = arrayRow[ind-fHalfN : ind+cHalfN+1]
    return uL, uR, uCL, uCR

def weno_launcher(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]

    Wx = zeros([nx, ny, nz, N1, nV])
    tempu = extend(u, N, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        uL, uR, uCL, uCR = extract_stencils(tempu[:,j,k], i+N)
        if i < 2*N-1:
            coeffs(Wx[i, j, k], uL, uR, uCL, uCR, R=1)
        elif i > nx:
            coeffs(Wx[i, j, k], uL, uR, uCL, uCR, L=1)
        else:
            coeffs(Wx[i, j, k], uL, uR, uCL, uCR)
    if ny==1:
        return Wx

    Wxy = zeros([nx, ny, nz, N1, N1, nV])
    tempWx = extend(Wx, N, 1)
    for i, j, k, a in product(range(nx), range(ny), range(nz), range(N1)):
        uL, uR, uCL, uCR = extract_stencils(tempWx[i,:,k,a], j+N)
        coeffs(Wxy[i, j, k, a], uL, uR, uCL, uCR)

    if nz==1:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N1, N1, N1, nV])
    tempWxy = extend(Wxy, N, 2)
    for i, j, k, a, b in product(range(nx), range(ny), range(nz), range(N1), range(N1)):
        uL, uR, uCL, uCR = extract_stencils(tempWxy[i,j,:,a,b], k+N)
        coeffs(Wxyz[i, j, k, a, b], uL, uR, uCL, uCR)

    return Wxyz


######  EXPERIMENTAL  ######


from numpy import dot

from solvers.basis import MIDVALS
from system.gpr.misc.structures import Cvec_to_Pvec


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
