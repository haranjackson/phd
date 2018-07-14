from numpy import eye, zeros

from gpr.misc.structures import Cvec
from gpr.opts import NV
from gpr.vars.hyp import Cvec_hyp


def riemann_IC(nx, dX, QL, QR, x0, isMulti):

    u = zeros([nx, NV])

    for i in range(nx):

        if i * dX[0] < x0:
            u[i] = QL

            if isMulti:
                u[i, -1] = -1

        else:
            u[i] = QR

            if isMulti:
                u[i, -1] = 1

    return u


def primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs, x0=0.5, λL=None, λR=None):

    isMulti = len(MPs) > 1
    MPL = MPs[0]
    MPR = MPs[1] if isMulti else MPL

    AL = (ρL / MPL.ρ0)**(1 / 3) * eye(3)
    JL = zeros(3)
    QL = Cvec(ρL, pL, vL, MPL, AL, JL, λL)

    AR = (ρR / MPR.ρ0)**(1 / 3) * eye(3)
    JR = zeros(3)
    QR = Cvec(ρR, pR, vR, MPR, AR, JR, λR)

    return riemann_IC(nx, dX, QL, QR, x0, isMulti)


def hyperelastic_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs, x0=0.5):

    isMulti = len(HYPs) > 1
    HYPL = HYPs[0]
    HYPR = HYPs[1] if isMulti else HYPL

    QL = Cvec_hyp(FL, SL, vL, HYPL)
    QR = Cvec_hyp(FR, SR, vR, HYPR)

    return riemann_IC(nx, dX, QL, QR, x0, isMulti)
