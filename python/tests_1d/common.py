from itertools import product

from numpy import eye, zeros

from options import nx, ny, nz, nV, dx, RGFM
from system.gpr.misc.structures import Cvec


def riemann_IC(ρL, pL, vL, ρR, pR, vR, PARL, PARR=None, x0=0.5):
    """ constructs the riemann problem corresponding to the parameters given
    """
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    if PARR is None:
        PARR = PARL

    QL = Cvec(ρL, pL, vL, AL, JL, PARL)
    QR = Cvec(ρR, pR, vR, AR, JR, PARR)

    u = zeros([nx, ny, nz, nV])

    if RGFM:

        for i, j, k in product(range(nx), range(ny), range(nz)):

            if i*dx < x0:
                u[i, j, k] = QL
            else:
                u[i, j, k] = QR

            u[i, j, k, -1] = i*dx - x0

        return u, [PARL, PARR]

    else:

        for i, j, k in product(range(nx), range(ny), range(nz)):

            if i*dx < x0:
                u[i, j, k] = QL
            else:
                u[i, j, k] = QR

        return u, [PARL]
