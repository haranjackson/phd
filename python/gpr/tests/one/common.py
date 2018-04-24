from numpy import eye, zeros

from gpr.misc.structures import Cvec


def riemann_IC(tf, nx, dX, QL, QR, MPL, MPR, x0):

    u = zeros([nx, len(QL)])

    for i in range(nx):

        if i * dX[0] < x0:
            u[i] = QL
        else:
            u[i] = QR

        if MPR is not None:
            u[i, -1] = (i + 0.5) * dX[0] - x0

    if MPR is not None:
        MPs = [MPL, MPR]
    else:
        MPs = [MPL]

    return u, MPs, tf, dX


def fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MPL, MPR=None, x0=0.5):
    """ constructs the riemann problem corresponding to the parameters given
    """
    if MPR is None:
        MPR_ = MPL
    else:
        MPR_ = MPR

    AL = (ρL / MPL.ρ0)**(1 / 3) * eye(3)
    JL = zeros(3)
    QL = Cvec(ρL, pL, vL, AL, JL, MPL)

    AR = (ρR / MPR_.ρ0)**(1 / 3) * eye(3)
    JR = zeros(3)
    QR = Cvec(ρR, pR, vR, AR, JR, MPR_)

    return riemann_IC(tf, nx, dX, QL, QR, MPL, MPR, x0)
