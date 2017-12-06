from itertools import product

from numpy import array, eye, zeros
from numpy.linalg import det, inv

from auxiliary.boundaries import standard_BC
from system.gpr.misc.structures import Cvec
from system.gpr.variables.eos_hyp import total_energy_hyp, temperature_hyp
from system.gpr.variables.state import pressure2
from tests_1d.common import HYP_COP, PAR_COP_SMG, PAR_COP_SMG_P, PAR_COP_CC
from options import nx, ny, nz, nV, dx


def hyperelastic_to_gpr(v, A, S, HYP):
    Q = zeros(nV)
    ρ = HYP.ρ0 * det(A)
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

    ρL = HYP.ρ0 * det(AL)
    ρR = HYP.ρ0 * det(AR)

    QL = hyperelastic_to_gpr(vL, AL, SL, HYP_COP)
    QR = hyperelastic_to_gpr(vR, AR, SR, HYP_COP)
    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR_COP_SMG]

def elastic1_IC():
    """ tf = 0.06
        L = 1
    """
    AL = inv(array([[0.95, 0, 0],
                    [0,    1, 0],
                    [0,    0, 1]]))
    vL = zeros(3)
    SL = 0.001

    AR = inv(array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]))
    vR = zeros(3)
    SR = 0

    J = zeros(3)

    PAR = PAR_COP_CC
    HYP = HYP_COP

    ρL = PAR.ρ0 * det(AL)
    ρR = PAR.ρ0 * det(AR)
    TL = temperature_hyp(SL, AL, HYP)
    TR = temperature_hyp(SR, AR, HYP)
    pL = pressure2(ρL, TL, PAR)
    pR = pressure2(ρR, TR, PAR)

    pL = 37.41
    pR = 0

    QL = Cvec(ρL, pL, vL, AL, J, PAR)
    QR = Cvec(ρR, pR, vR, AR, J, PAR)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR]

def elastic2_IC():
    """ tf = 0.06
        L = 1
    """
    AL = inv(array([[0.95, 0, 0],
                    [0.05, 1, 0],
                    [0,    0, 1]]))
    vL = array([0,1,0])
    SL = 0.001

    AR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    J = zeros(3)

    PAR = PAR_COP_CC
    HYP = HYP_COP

    ρL = PAR.ρ0 * det(AL)
    ρR = PAR.ρ0 * det(AR)
    TL = temperature_hyp(SL, AL, HYP)
    TR = temperature_hyp(SR, AR, HYP)
    pL = pressure2(ρL, TL, PAR)
    pR = pressure2(ρR, TR, PAR)

    QL = Cvec(ρL, pL, vL, AL, J, PAR)
    QR = Cvec(ρR, pR, vR, AR, J, PAR)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR]

def piston_IC():
    """ tf = 1.5
        L = 1.5
    """
    PAR = PAR_COP_SMG_P
    ρ = PAR.ρ0
    p = PAR.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    Q = Cvec(ρ, p, v, A, J, PAR)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        u[i,j,k] = Q

    return u, [PAR]

def piston_BC(u):
    ret = standard_BC(u)
    for j, k in product(range(ny), range(nz)):
        ret[0,j,k,2:5] = ret[0,j,k,0] * array([0.002,0,0])
    return ret
