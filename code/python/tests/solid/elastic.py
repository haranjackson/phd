from numpy import array, zeros

from tests.params.alt import Cu_HYP_CGS
from tests.params.solids import Cu_GR_CGS
from tests.common import hyperelastic_IC


def barton(test):
    """ 10.1016/j.jcp.2009.06.014
        5.1 Testcase 1, 5.2 Testcase 2

        N = 3
        cfl = 0.6
        SPLIT = True
        SOLVER = 'roe'
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

    def f():
        return u, MPs, tf, dX, 'transitive'

    return f


def pure_elastic(test):
    """ 10.1016/j.compfluid.2016.05.004
        5.5 Purely elastic Riemann problems

        N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'rusanov'
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

    def f():
        return u, MPs, tf, dX, 'transitive'

    return f
