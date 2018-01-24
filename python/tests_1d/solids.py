from itertools import product

from numpy import array, eye, zeros

from etc.boundaries import standard_BC
from gpr.misc.structures import Cvec
from gpr.variables.hyp import Cvec_hyp
from tests_1d.common import HYP_COP, MP_COP_GR, MP_COP_SMG, MP_COP_SMG_P, MP_ALU_SG
from options import nx, ny, nz, nV, dx


def solid_IC(tf, vL, vR, FL, FR, SL, SR, HYP, MP):

    QL = Cvec_hyp(FL, SL, vL, HYP)
    QR = Cvec_hyp(FR, SR, vR, HYP)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i * dx < 0.5:
            u[i, j, k] = QL
        else:
            u[i, j, k] = QR

    return u, [MP], tf


def barton1_IC():

    tf = 0.06

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

    return solid_IC(tf, vL, vR, FL, FR, SL, SR, HYP_COP, MP_COP_GR)


def barton2_IC():

    tf = 0.06

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

    return solid_IC(tf, vL, vR, FL, FR, SL, SR, HYP_COP, MP_COP_GR)


def elastic1_IC():

    tf = 0.06

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

    return solid_IC(tf, vL, vR, FL, FR, SL, SR, HYP_COP, MP_COP_SMG)


def elastic2_IC():

    tf = 0.06

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

    return solid_IC(tf, vL, vR, FL, FR, SL, SR, HYP_COP, MP_COP_SMG)


def piston_IC():

    tf = 1.5

    MP = MP_COP_SMG_P
    ρ = MP.ρ0
    p = MP.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    Q = Cvec(ρ, p, v, A, J, MP)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        u[i, j, k] = Q

    return u, [MP], tf


def piston_BC(u):
    ret = standard_BC(u)
    for j, k in product(range(ny), range(nz)):
        ret[0, j, k, 2:5] = ret[0, j, k, 0] * array([0.002, 0, 0])
    return ret


def favrie1_IC():

    tf = 50e-6

    MP = MP_ALU_SG

    ρ = MP.ρ0
    p = MP.p0
    A = eye(3)
    J = zeros(3)

    vL = array([0, 500, 0])
    vR = array([0, -500, 0])

    QL = Cvec(ρ, p, vL, A, J, MP)
    QR = Cvec(ρ, p, vR, A, J, MP)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i * dx < 0.5:
            u[i, j, k] = QL
        else:
            u[i, j, k] = QR

    return u, [MP], tf


def favrie2_IC():

    tf = 50e-6

    MP = MP_ALU_SG

    ρ = MP.ρ0
    p = MP.p0
    A = eye(3)
    J = zeros(3)

    vL = array([100, 500, 0])
    vR = array([-100, -500, 0])

    QL = Cvec(ρ, p, vL, A, J, MP)
    QR = Cvec(ρ, p, vR, A, J, MP)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i * dx < 0.5:
            u[i, j, k] = QL
        else:
            u[i, j, k] = QR

    return u, [MP], tf
