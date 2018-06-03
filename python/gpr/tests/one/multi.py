from numpy import array, eye, sqrt, zeros

from gpr.misc.structures import Cvec
from gpr.vars.hyp import Cvec_hyp
from gpr.opts import NV

from gpr.tests.params import Air_SG_SI, He_SG_SI, H20_SG_SI, Cu_SMG_SI, \
    Cu_HYP_SI, Al_HYP_CGS, Al_GR_CGS, VAC
from gpr.tests.one.common import riemann_IC, primitive_IC
from gpr.tests.one.newtonian import heat_conduction_IC


def water_air_IC():

    tf = 2.3744e-4
    nx = 200
    Lx = 1
    MPs = [H20_SG_SI, Air_SG_SI]

    dX = [Lx / nx]

    ρL = 1000
    pL = 1e9
    vL = zeros(3)

    ρR = 50
    pR = 101325
    vR = zeros(3)

    u = primitive_IC(nx, dX, ρL, pL, vL, ρR, pR, vR, MPs, x0=0.7)
    return u, MPs, tf, dX


def helium_bubble_IC():

    tf = 14e-4
    nx = 200
    Lx = 1

    dX = [Lx / nx]
    dx = dX[0]

    ρL = 1.3333
    pL = 1.5e5
    vL = array([35.35 * sqrt(10), 0, 0])
    AL = ρL**(1 / 3) * eye(3)

    ρM = 1
    pM = 1e5
    vM = zeros(3)
    AM = ρM**(1 / 3) * eye(3)

    ρR = 0.1379
    pR = 1e5
    vR = zeros(3)
    AR = ρR**(1 / 3) * eye(3)

    J = zeros(3)

    u = zeros([nx, NV])
    Q1 = Cvec(ρL, pL, vL, AL, J, Air_SG_SI)
    Q2 = Cvec(ρM, pM, vM, AM, J, Air_SG_SI)
    Q3 = Cvec(ρR, pR, vR, AR, J, He_SG_SI)

    for i in range(nx):

        if i * dx < 0.05:
            u[i] = Q1
        elif i * dx < 0.4:
            u[i] = Q2
        elif i * dx < 0.6:
            u[i] = Q3
        else:
            u[i] = Q2

        if i * dx < 0.4 or i * dx >= 0.6:
            u[i, -1] = -1
        else:
            u[i, -1] = 1

    return u, [Air_SG_SI, He_SG_SI], tf, dX


def pbx_copper_IC(test):
    """ 10.1016/j.jcp.2011.07.008
        6.1 Initial value problems Initial
    """
    tf = 0.5e-6
    nx = 500
    Lx = 1
    MPs = [PBX_SG_SI, Cu_GR_SI]

    dX = [Lx / nx]

    if test == 1:

        pL = 18.9e9

        vR = zeros(3)
        FR = eye(3)

    elif test == 2:

        pL = 1e5

        vR = array([2, 0, 0.1])
        FR = array([[1, 0, 0],
                    [-0.01, 0.95, 0.02],
                    [-0.015, 0, 0.9]])

    vL = zeros(3)
    ρL = 1840
    AL = eye(3)
    JL = zeros(3)

    SR = 0

    QL = Cvec(ρL, pL, vL, AL, JL, PBX_SG_SI)
    QR = Cvec_hyp(FR, SR, vR, Cu_HYP_SI)

    u = riemann_IC(nx, dX, QL, QR, 0.5, True)
    return u, MPs, tf, dX


def aluminium_vacuum_IC():
    """ 10.1016/j.jcp.2010.04.012
        5.3 Solid/vacuum problem
    """
    tf = 0.06
    nx = 500
    Lx = 1
    MPs = [Al_GR_CGS, VAC]

    dX = [Lx / nx]

    vL = array([2, 0, 0.1])
    FL = array([[1, 0, 0],
                [-0.01, 0.95, 0.02],
                [-0.015, 0, 0.9]])
    SL = 0

    QL = Cvec_hyp(FL, SL, vL, Al_HYP_CGS)

    QR = zeros(NV)

    u = riemann_IC(nx, dX, QL, QR, 0.5, True)
    return u, MPs, tf, dX


def heat_conduction_multi_IC():
    return heat_conduction_IC(isMulti=True)
