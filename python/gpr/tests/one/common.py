from numpy import eye, zeros

from gpr.misc.structures import Cvec
from gpr.opts import NV
from gpr.vars.hyp import Cvec_hyp


def riemann_IC(nx, dX, QL, QR, x0, isMulti):

    u = zeros([nx, NV])

    for i in range(nx):

        if i * dX[0] < x0:
            u[i] = QL
        else:
            u[i] = QR

        if isMulti:
            u[i, -1] = (i + 0.5) * dX[0] - x0

    return u


def fluids_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs, x0=0.5):

    isMulti = len(MPs) > 1
    MPL = MPs[0]
    MPR = MPs[1] if isMulti else MPL

    AL = (ρL / MPL.ρ0)**(1 / 3) * eye(3)
    JL = zeros(3)
    QL = Cvec(ρL, pL, vL, AL, JL, MPL)

    AR = (ρR / MPR.ρ0)**(1 / 3) * eye(3)
    JR = zeros(3)
    QR = Cvec(ρR, pR, vR, AR, JR, MPR)

    return riemann_IC(nx, dX, QL, QR, x0, isMulti)


def hyperelastic_solid_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs, x0=0.5):

    isMulti = len(HYPs) > 1
    HYPL = HYPs[0]
    HYPR = HYPs[1] if isMulti else HYPL

    QL = Cvec_hyp(FL, SL, vL, HYPL)
    QR = Cvec_hyp(FR, SR, vR, HYPR)

    return riemann_IC(nx, dX, QL, QR, x0, isMulti)
