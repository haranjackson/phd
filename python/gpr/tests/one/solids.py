from numpy import array, eye, zeros

from ader.etc.boundaries import standard_BC

from ...variables.hyp import Cvec_hyp
from ...systems.conserved import SystemConserved

from .common import riemann_IC
from .params import HYP_Cu, MP_Cu_GR, MP_Cu_SMG_P, MP_Al_SG


def solid_IC(tf, nx, dX, vL, vR, FL, FR, SL, SR, HYP, MP, x0=0.5):

    QL = Cvec_hyp(FL, SL, vL, HYP)
    QR = Cvec_hyp(FR, SR, vR, HYP)

    sys = SystemConserved(VISCOUS=True, THERMAL=False)

    return riemann_IC(sys, tf, nx, dX, QL, QR, MP, MP, x0)


def barton1_IC():

    tf = 0.06
    nx = 500
    Lx = 1

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

    print("BARTON1")
    return solid_IC(tf, nx, dX, vL, vR, FL, FR, SL, SR, HYP_Cu, MP_Cu_GR)


def barton2_IC():

    tf = 0.06
    nx = 500
    Lx = 1

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

    print("BARTON2")
    return solid_IC(tf, nx, dX, vL, vR, FL, FR, SL, SR, HYP_Cu, MP_Cu_GR)


def elastic1_IC():

    tf = 0.06
    nx = 200
    Lx = 1

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

    print("ELASTIC1")
    return solid_IC(tf, nx, dX, vL, vR, FL, FR, SL, SR, HYP_Cu, MP_Cu_GR)


def elastic2_IC():

    tf = 0.06
    nx = 200
    Lx = 1

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

    print("ELASTIC2")
    return solid_IC(tf, nx, dX, vL, vR, FL, FR, SL, SR, HYP_Cu, MP_Cu_GR)


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

    sys = SystemConserved(VISCOUS=True, THERMAL=False)

    Q = sys.Cvec(ρ, p, v, A, J, MP)

    u = zeros([nx, 17])

    for i in range(nx):
        u[i] = Q

    print("ELASTO-PLASTIC PISTON")
    return u, [MP], tf, dX, sys


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

    sys = SystemConserved(VISCOUS=True, THERMAL=False)

    QL = sys.Cvec(ρ, p, vL, A, J, MP)
    QR = sys.Cvec(ρ, p, vR, A, J, MP)

    print("FAVRIE1")
    return riemann_IC(tf, nx, dX, QL, QR, MP, MP, 0.5)


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

    sys = SystemConserved(VISCOUS=True, THERMAL=False)

    QL = sys.Cvec(ρ, p, vL, A, J, MP)
    QR = sys.Cvec(ρ, p, vR, A, J, MP)

    print("FAVRIE2")
    return riemann_IC(tf, nx, dX, QL, QR, MP, MP, 0.5)
