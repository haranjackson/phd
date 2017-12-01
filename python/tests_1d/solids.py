from itertools import product

from numpy import array, eye, zeros
from numpy.linalg import inv

from auxiliary.boundaries import standard_BC
from system.gpr.misc.structures import Cvec
from system.gpr.variables.eos_hyp import total_energy_hyp, density_hyp
from tests_1d.common import HYP_COP, PAR_COP, PAR_COP2
from options import nx, ny, nz, nV, dx


def hyperelastic_to_gpr(v, A, S, HYP):
    Q = zeros(nV)
    ρ = density_hyp(A, HYP)
    Q[0] = ρ
    Q[1] = ρ * total_energy_hyp(A, S, v, HYP)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    return Q

def barton_IC():
    """ tf = 0.6
        L = 1
    """
    vL = array([2e3, 0, 100])
    FL = array([[1,      0,    0   ],
                [-0.01,  0.95, 0.02],
                [-0.015, 0,    0.9 ]])
    AL = inv(FL)
    SL = 0

    vR = array([0, -30, -10])
    FR = array([[1,     0,    0  ],
                [0.015, 0.95, 0  ],
                [-0.01, 0,    0.9]])
    AR = inv(FR)
    SR = 0

    QL = hyperelastic_to_gpr(vL, AL, SL, HYP_COP)
    QR = hyperelastic_to_gpr(vR, AR, SR, HYP_COP)
    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR_COP]

def elastic1_IC():
    """ tf = 0.06
        L = 1
    """
    AL = array([[0.95, 0, 0],
                [0,    1, 0],
                [0,    0, 1]])
    vL = zeros(3)
    SL = 0.001

    AR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    QL = hyperelastic_to_gpr(vL, AL, SL, HYP_COP)
    QR = hyperelastic_to_gpr(vR, AR, SR, HYP_COP)
    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR_COP]

def elastic2_IC():
    """ tf = 0.06
        L = 1
    """
    AL = array([[0.95, 0, 0],
                [0.05, 1, 0],
                [0,    0, 1]])
    vL = array([0,1,0])
    SL = 0.001

    AR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    QL = hyperelastic_to_gpr(vL, AL, SL, HYP_COP)
    QR = hyperelastic_to_gpr(vR, AR, SR, HYP_COP)
    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR_COP]

def piston_IC():
    """ tf = 1.5
        L = 1.5
    """
    ρ = PAR_COP2.ρ0
    p = PAR_COP2.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    Q = Cvec(ρ, p, v, A, J, PAR_COP2)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        u[i,j,k] = Q

    return u, [PAR_COP]

def piston_BC(u):
    ret = standard_BC(u)
    for j, k in product(range(ny), range(nz)):
        ret[0,j,k,2:5] = ret[0,j,k,0] * array([0.002,0,0])
    return ret
