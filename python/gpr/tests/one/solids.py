from numpy import array, eye, zeros

from ader.etc.boundaries import standard_BC

from gpr.misc.structures import Cvec
from gpr.tests.params import Cu_HYP_CGS, Cu_GR_CGS, Cu_SMGP_ALT
from gpr.tests.one.common import hyperelastic_IC


def barton_IC(test):
    """ 10.1016/j.jcp.2009.06.014
        5.1 Testcase 1, 5.2 Testcase 2
    """
    tf = 0.06
    nx = 500
    Lx = 1
    HYPs = [Cu_HYP_CGS]
    MPs = [Cu_GR_CGS]

    dX = [Lx / nx]

    if test == 1:

        vL = array([0, 0.5, 1])
        FL = array([[0.98, 0, 0],
                    [0.02, 1, 0.1],
                    [0, 0, 1]])
        SL = 0.001

        vR = array([0, 0, 0])
        FR = array([[1, 0, 0],
                    [0, 1, 0.1],
                    [0, 0, 1]])
        SR = 0

    elif test == 2:

        vL = array([2, 0, 0.1])
        FL = array([[1, 0, 0],
                    [-0.01, 0.95, 0.02],
                    [-0.015, 0, 0.9]])
        SL = 0

        vR = array([0, -0.03, -0.01])
        FR = array([[1, 0, 0],
                    [0.015, 0.95, 0],
                    [-0.01, 0, 0.9]])
        SR = 0

    u = hyperelastic_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    return u, MPs, tf, dX


def elastic_IC(test):
    """ 10.1016/j.compfluid.2016.05.004
        5.5 Purely elastic Riemann problems
    """
    tf = 0.06
    nx = 200
    Lx = 1
    HYPs = [Cu_HYP_CGS]
    MPs = [Cu_GR_CGS]

    dX = [Lx / nx]

    if test == 1:
        FL = array([[0.95, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
        vL = zeros(3)

    elif test == 2:
        FL = array([[0.95, 0, 0],
                    [0.05, 1, 0],
                    [0, 0, 1]])
        vL = array([0, 1, 0])

    SL = 0.001
    FR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    u = hyperelastic_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    return u, MPs, tf, dX


def piston_IC():
    """ 10.1016/j.compfluid.2016.05.004
        5.6 Elastic-plastic piston problem
    """
    tf = 1.5
    nx = 300
    Lx = 1.5

    dX = [Lx / nx]

    MP = Cu_SMGP_ALT
    ρ = MP.ρ0
    p = MP.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    Q = Cvec(ρ, p, v, A, J, MP)

    u = zeros([nx, 14])

    for i in range(nx):
        u[i] = Q

    return u, [MP], tf, dX


def piston_BC(u, N, NDIM):
    ret = standard_BC(u, N, NDIM)
    ret[:N, 2:5] = ret[N, 0] * array([0.002, 0, 0])
    return ret
