""" Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
"""
from itertools import product

from numpy import ceil, concatenate, dot, einsum, floor, int64, multiply, ones, zeros
from scipy.linalg import solve

from options import rc, λc, λs, eps, ndim, N, N1, WENO_AVERAGE, RECONSTRUCT_PRIM
from solvers.basis import mid_values
from solvers.weno.matrices import coefficient_matrices, oscillation_indicator
from solvers.weno.matrices import inv_coeff_mats_1, inv_coeff_mats_2
from gpr.variables.vectors import Cvec_to_Pvec


Mc = coefficient_matrices()
McInv1L, McInv1R = inv_coeff_mats_1()
McInv2L, McInv2R, McInv2C = inv_coeff_mats_2()
Σ = oscillation_indicator()

midvals = mid_values()
floorHalfN = int(floor(N/2))
ceilHalfN = int(ceil(N/2))


lamList = [λs, λs, λc, λc]
if N%2:
    nStencils = 4
else:
    nStencils = 3


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension+1], ones(n-2), [extension+1])).astype(int64)
    return inarray.repeat(reps, axis=d)

def coeffs1(uL, uR):
    """ Calculate coefficients of basis polynomials and weights for N=1
    """
    wL = dot(McInv1L, uL)
    wR = dot(McInv1R, uR)
    ΣwL = dot(Σ, wL)
    ΣwR = dot(Σ, wR)
    σL = einsum('ki,ki->i', wL, ΣwL)
    σR = einsum('ki,ki->i', wR, ΣwR)
    oL = λs / (abs(σL) + eps)**rc
    oR = λs / (abs(σR) + eps)**rc
    return (multiply(wL,oL) + multiply(wR,oR)) / (oL+oR)

def coeffs2(uL, uR, uC):
    """ Calculate coefficients of basis polynomials and weights for N=2
    """
    wL = dot(McInv2L, uL)
    wR = dot(McInv2R, uR)
    wC = dot(McInv2C, uC)
    ΣwL = dot(Σ, wL)
    ΣwR = dot(Σ, wR)
    ΣwC = dot(Σ, wC)
    σL = einsum('ki,ki->i', wL, ΣwL)
    σR = einsum('ki,ki->i', wR, ΣwR)
    σC = einsum('ki,ki->i', wC, ΣwC)
    oL = λs / (abs(σL) + eps)**rc
    oR = λs / (abs(σR) + eps)**rc
    oC = λc / (abs(σC) + eps)**rc
    return (multiply(wL,oL) + multiply(wR,oR) + multiply(wC,oC)) / (oL+oR+oC)

def coeffsX(uList):
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

def coeffs(ret, u1, u2, u3, u4):
    if N==1:
        ret[:] = coeffs1(u1, u2)
    elif N==2:
        ret[:] = coeffs2(u1, u2, u3)
    elif nStencils==3:
        ret[:] = coeffsX([u1, u2, u3])
    elif nStencils==4:
        ret[:] = coeffsX([u1, u2, u3, u4])

def extract_stencils(arrayRow, ind):
    u1 = arrayRow[ind-N : ind+1]
    u2 = arrayRow[ind : ind+N+1]
    u3 = arrayRow[ind-ceilHalfN : ind+floorHalfN+1]
    u4 = arrayRow[ind-floorHalfN : ind+ceilHalfN+1]
    return u1, u2, u3, u4

def weno(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]

    Wx = zeros([nx, ny, nz, N1, 18])
    tempu = extend(u, N, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        u1, u2, u3, u4 = extract_stencils(tempu[:,j,k], i+N)
        coeffs(Wx[i, j, k], u1, u2, u3, u4)

    if ny==1:
        return Wx

    Wxy = zeros([nx, ny, nz, N1, N1, 18])
    tempWx = extend(Wx, N, 1)
    for i, j, k, a in product(range(nx), range(ny), range(nz), range(N1)):
        u1, u2, u3, u4 = extract_stencils(tempWx[i,:,k,a], j+N)
        coeffs(Wxy[i, j, k, a], u1, u2, u3, u4)

    if nz==1:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N1, N1, N1, 18])
    tempWxy = extend(Wxy, N, 2)
    for i, j, k, a, b in product(range(nx), range(ny), range(nz), range(N1), range(N1)):
        u1, u2, u3, u4 = extract_stencils(tempWxy[i,j,:,a,b], k+N)
        coeffs(Wxyz[i, j, k, a, b], u1, u2, u3, u4)

    return Wxyz

def weno_primitive(q, PAR):
    """ Returns a WENO reconstruction in primitive variables, given the grid of
        conserved values. A reconstruction in conserved variables is performed.
        The midpoints of this reconstruction are then taken as the conserved
        cell averages (required for >2nd order). A primitive reconstruction is
        then performed with these averages.
    """
    qr = weno(q)
    qav = dot(midvals, qr)
    pav = zeros(qav.shape)
    nx, ny, nz = qav.shape[:3]
    for i, j, k in product(range(nx), range(ny), range(nz)):
        pav[i,j,k] = Cvec_to_Pvec(qav[i,j,k], PAR)
    return weno(pav)

def weno_launcher(u):

    if RECONSTRUCT_PRIM:
        weno_func = weno_primitive
    else:
        weno_func = weno

    if ndim==2 and WENO_AVERAGE:
        wx = weno_func(u)
        wy = weno_func(u.swapaxes(0,1)).swapaxes(0,1).swapaxes(3,4)
        return (wx+wy)/2
    else:
        return weno_func(u)

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
