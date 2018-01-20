from itertools import product

from numpy import dot, tensordot, zeros

from solvers.basis import DERVALS
from system import Bdot, flux, system
from options import nV, dx, dy, N1, ndim


def derivative(X, dim):
    """ Returns the derivative of polynomial coefficients X with respect to
        dimension dim. X can be of shape (N1,...) or (N1,N1,...)
    """
    if dim == 0:
        return tensordot(DERVALS, X, (1, 0)) / dx
    elif dim == 1:
        return tensordot(DERVALS, X, (1, 1)).swapaxes(0, 1) / dy


def weno_midstepper(wh, dt, MP):
    """ Steps the WENO reconstruction forwards by dt/2, under the homogeneous system
    """
    USE_JACOBIAN = 0

    nx, ny, nz = wh.shape[:3]

    F = zeros([N1] * ndim + [nV])
    G = zeros([N1] * ndim + [nV])
    Bdwdx = zeros(nV)
    Bdwdy = zeros(nV)

    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        dwdx = derivative(w, 0)

        if ndim == 1:

            if USE_JACOBIAN:
                for a in range(N1):
                    Mx = system(w[a], 0, MP)
                    w[a] -= dt / 2 * dot(Mx, dwdx[a])
            else:
                for a in range(N1):
                    F[a] = flux(w[a], 0, MP)
                dFdx = derivative(F, 0)
                for a in range(N1):
                    Bdot(Bdwdx, dwdx[a], w[a], 0, MP)
                    w[a] -= dt / 2 * (dFdx[a] + Bdwdx)

        elif ndim == 2:

            dwdy = derivative(w, 1)

            if USE_JACOBIAN:
                for a, b in product(range(N1), range(N1)):
                    Mx = system(w[a, b], 0, MP)
                    My = system(w[a, b], 1, MP)
                    w[a, b] -= dt / 2 * \
                        (dot(Mx, dwdx[a, b]) + dot(My, dwdy[a, b]))
            else:
                for a, b in product(range(N1), range(N1)):
                    F[a, b] = flux(w[a, b], 0, MP)
                    G[a, b] = flux(w[a, b], 1, MP)
                dFdx = derivative(F, 0)
                dGdy = derivative(G, 1)
                for a, b in product(range(N1), range(N1)):
                    Bdot(Bdwdx, dwdx[a, b], w[a, b], 0, MP)
                    Bdot(Bdwdy, dwdy[a, b], w[a, b], 1, MP)
                    w[a, b] -= dt / 2 * \
                        (dFdx[a, b] + dGdy[a, b] + Bdwdx + Bdwdy)
