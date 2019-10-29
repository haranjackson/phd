from numpy import array, concatenate, eye, pad, sqrt, zeros

from gpr.misc.structures import Cvec
from gpr.vars.hyp import Cvec_hyp

from tests.common import riemann_IC, primitive_IC
from tests.fluid.newtonian1 import heat_conduction
from tests.params.alt import VAC, Cu_HYP_SI, Al_HYP_CGS
from tests.params.fluids import Air_SG_SI, He_SG_SI, H20_SG_SI
from tests.params.reactive import PBX_SG_SI
from tests.params.solids import Cu_GR_SI, Al_GR_CGS


def water_air():
    """ 10.1016/j.jcp.2003.11.015
        7.1. Water–air shock tube

        N = 3
        cfl = 0.5
        SPLIT = True
        FLUX = 0
        RELAXATION = true
    """
    tf = 2.3744e-4
    nx = 3000
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
    return u, MPs, tf, dX, 'transitive'


def helium_bubble():
    """ 10.1016/j.jcp.2003.10.010
        5. Numerical experiments - Test B

        N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'rusanov'
    """
    #tf = 7e-4
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

    u = zeros([nx, 18])
    Q1 = Cvec(ρL, pL, vL, Air_SG_SI, AL)
    Q2 = Cvec(ρM, pM, vM, Air_SG_SI, AM)
    Q3 = Cvec(ρR, pR, vR, He_SG_SI, AR)

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

    return u, [Air_SG_SI, He_SG_SI], tf, dX, 'transitive'


def pbx_copper(test):
    """ 10.1016/j.jcp.2011.07.008
        6.1 Initial value problems

        N = 3
        cfl = 0.5
        SPLIT = True
        FLUX = 0
    """
    nx = 3000
    Lx = 1

    dX = [Lx / nx]

    if test == 1:

        pL = 18.9e9
        vR = zeros(3)
        FR = eye(3)
        tf = 5e-5

    elif test == 2:

        pL = 1e5
        vR = array([2, 0, 0.1])
        FR = array([[1, 0, 0], [-0.01, 0.95, 0.02], [-0.015, 0, 0.9]])
        tf = 9e-5

    vL = zeros(3)
    ρL = 1840
    AL = eye(3)

    SR = 0

    if test == 1:
        QL = Cvec(ρL, pL, vL, PBX_SG_SI, AL)
        QR = Cvec_hyp(FR, SR, vR, Cu_HYP_SI)
        MPs = [PBX_SG_SI, Cu_GR_SI]

    elif test == 2:
        QL = Cvec_hyp(FR, SR, vR, Cu_HYP_SI)
        QR = Cvec(ρL, pL, vL, PBX_SG_SI, AL)
        MPs = [Cu_GR_SI, PBX_SG_SI]

    QL = concatenate([QL, [0]])
    QR = concatenate([QR, [0]])

    u = riemann_IC(nx, dX, QL, QR, 0.5, True)
    return u, MPs, tf, dX, 'transitive'


def aluminium_vacuum(THERMAL=True):
    """ 10.1016/j.jcp.2010.04.012
        5.3 Solid/vacuum problem

        N = 3
        cfl = 0.5
        SPLIT = False
        FLUX = 0
        RELAXATION = false
    """
    tf = 0.06
    nx = 3000
    Lx = 1
    MPs = [Al_GR_CGS, VAC]

    dX = [Lx / nx]

    vL = array([2, 0, 0.1])
    FL = array([[1, 0, 0], [-0.01, 0.95, 0.02], [-0.015, 0, 0.9]])
    SL = 0

    NV = 15 + int(THERMAL) * 3
    QL = Cvec_hyp(FL, SL, vL, Al_HYP_CGS)
    QL = pad(QL, NV - 14, 'constant')[NV - 14:]
    QR = zeros(NV)

    u = riemann_IC(nx, dX, QL, QR, 0.5, True)
    return u, MPs, tf, dX, 'transitive'


def heat_conduction_multi():
    return heat_conduction(isMulti=True)
