from numpy import array, eye, zeros

from ader.etc.boundaries import standard_BC

from gpr.misc.structures import Cvec
from gpr.tests.one.common import riemann_IC, hyperelastic_solid_IC
from gpr.tests.one.params import HYP_Cu, MP_Cu_GR, MP_Cu_SMG_P, MP_Al_SG


def barton1_IC(isMulti=False):

    tf = 0.06
    nx = 500
    Lx = 1
    HYPs = [HYP_Cu]
    MPs = [MP_Cu_GR]

    if isMulti:
        HYPs = 2 * HYPs
        MPs = 2 * MPs

    dX = [Lx / nx]

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

    u = hyperelastic_solid_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    print("BARTON1")
    return u, MPs, tf, dX


def barton2_IC(isMulti=False):

    tf = 0.06
    nx = 500
    Lx = 1
    HYPs = [HYP_Cu]
    MPs = [MP_Cu_GR]

    if isMulti:
        HYPs = 2 * HYPs
        MPs = 2 * MPs

    dX = [Lx / nx]

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

    u = hyperelastic_solid_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    print("BARTON1")
    return u, MPs, tf, dX


def elastic1_IC(isMulti=False):

    tf = 0.06
    nx = 200
    Lx = 1
    HYPs = [HYP_Cu]
    MPs = [MP_Cu_GR]

    if isMulti:
        HYPs = 2 * HYPs
        MPs = 2 * MPs

    dX = [Lx / nx]

    FL = array([[0.95, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vL = zeros(3)
    SL = 0.001

    FR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    u = hyperelastic_solid_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    print("BARTON1")
    return u, MPs, tf, dX


def elastic2_IC(isMulti=False):

    tf = 0.06
    nx = 200
    Lx = 1
    HYPs = [HYP_Cu]
    MPs = [MP_Cu_GR]

    if isMulti:
        HYPs = 2 * HYPs
        MPs = 2 * MPs

    dX = [Lx / nx]

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

    u = hyperelastic_solid_IC(nx, dX, vL, vR, FL, FR, SL, SR, HYPs)
    print("BARTON1")
    return u, MPs, tf, dX


def piston_IC():

    tf = 1.5
    nx = 300
    Lx = 1.5

    dX = [Lx / nx]

    MP = MP_Cu_SMG_P
    ρ = MP.ρ0
    p = MP.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)

    Q = Cvec(ρ, p, v, A, J, MP)

    u = zeros([nx, 17])

    for i in range(nx):
        u[i] = Q

    print("ELASTO-PLASTIC PISTON")
    return u, [MP], tf, dX


def piston_BC(u, N):
    ret = standard_BC(u)
    ret[:N, 2:5] = ret[N, 0] * array([0.002, 0, 0])
    return ret


def favrie1_IC():

    tf = 50e-6
    nx = 200
    Lx = 1

    dX = [Lx / nx]

    MP = MP_Al_SG

    ρ = MP.ρ0
    p = MP.p0
    A = eye(3)
    J = zeros(3)

    vL = array([0, 500, 0])
    vR = array([0, -500, 0])

    QL = Cvec(ρ, p, vL, A, J, MP)
    QR = Cvec(ρ, p, vR, A, J, MP)

    u = riemann_IC(nx, dX, QL, QR, 0.5, False)
    print("FAVRIE1")
    return u, [MP], tf, dX


def favrie2_IC():

    tf = 50e-6
    nx = 200
    Lx = 1

    dX = [Lx / nx]

    MP = MP_Al_SG

    ρ = MP.ρ0
    p = MP.p0
    A = eye(3)
    J = zeros(3)

    vL = array([100, 500, 0])
    vR = array([-100, -500, 0])

    QL = Cvec(ρ, p, vL, A, J, MP)
    QR = Cvec(ρ, p, vR, A, J, MP)

    u = riemann_IC(nx, dX, QL, QR, 0.5, False)
    print("FAVRIE2")
    return u, [MP], tf, dX
