""" Implements WENO method used in Dumbser et al, DOI 10.1016/j.cma.2013.09.022
"""
from itertools import product

from numpy import multiply, prod, zeros
from scipy.linalg import solve

from solvers.weno.matrices import fHalfN, cHalfN, WN_M, WN_Σ
from options import rc, λc, λs, eps, N, NV, NDIM


LAMS = [λs, λs, λc, λc]


def calculate_coeffs(uList):
    """ Calculate coefficients of basis polynomials and weights
    """
    n = len(uList)
    wList = [solve(WN_M[i], uList[i], overwrite_b=1, check_finite=0)
             for i in range(n)]
    σList = [((w.T).dot(WN_Σ).dot(w)).diagonal() for w in wList]
    oList = [LAMS[i] / (abs(σList[i]) + eps)**rc for i in range(n)]

    oSum = zeros(NV)
    numerator = zeros([N, NV])
    for i in range(n):
        oSum += oList[i]
        numerator += multiply(wList[i], oList[i])

    return numerator / oSum


def coeffs(ret, uL, uR, uCL, uCR):

    if N == 2:
        ret[:] = calculate_coeffs([uL, uR])
    elif N % 2:
        ret[:] = calculate_coeffs([uL, uR, uCL])
    else:
        ret[:] = calculate_coeffs([uL, uR, uCL, uCR])


def stencils(strip, ind):
    """ Returns the set of stencils along strip at position ind
    """
    uL = strip[ind - N + 1: ind + 1]
    uR = strip[ind: ind + N]
    uCL = strip[ind - cHalfN: ind + fHalfN + 1]
    uCR = strip[ind - fHalfN: ind + cHalfN + 1]
    return uL, uR, uCL, uCR


def weno_launcher(uBC):
    """ Find reconstruction coefficients of uBC to order N
    """
    if N == 1:  # No reconstruction necessary
        return uBC.reshape(list(uBC.shape)[:NDIM] + [1] * NDIM + [NV])

    rec = uBC
    # The reconstruction is built up along each dimension. At each step, rec
    # holds the reconstruction so far, and tmp holds the new reconstruction
    # along dimension d.
    for d in range(NDIM):

        # We are reconstructing along dimension d and so we lose N-1 cells in
        # this dimension at each end. We group the remaining dimensions.
        shape = rec.shape
        shape0 = shape[:d]
        shape1 = shape[d + 1 : NDIM]

        n1 = int(prod(shape0))
        n2 = shape[d] - 2 * (N - 1)
        n3 = int(prod(shape1))
        n4 = N**d

        tmp = zeros([n1, n2, n3, n4, N, NV])
        rec_ = rec.reshape(n1, shape[d], n3, n4, NV)

        for i, j, k, a in product(range(n1), range(n2), range(n3), range(n4)):
            strip = rec_[i, :, k, a]
            uL, uR, uCL, uCR = stencils(strip, j + N - 1)
            coeffs(tmp[i, j, k, a], uL, uR, uCL, uCR)

        rec = tmp.reshape(shape0 + (n2,) + shape1 + (N,)*(d + 1) + (NV,))

    return rec
