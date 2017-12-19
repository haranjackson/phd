from itertools import product

from numpy import array, eye, trace, zeros
from numpy.linalg import det, inv

from etc.boundaries import standard_BC
from system.gpr.misc.structures import Cvec
from system.gpr.variables.eos_hyp import Sigma_hyp
from system.gpr.variables.state import sigma
from tests_1d.common import HYP_COP, PAR_COP_SMG, PAR_COP_SMG_P
from options import nx, ny, nz, nV, dx


def hyperelastic_vars(F, S, HYP, PAR):
    """ Returns the GPR variables corresponding to the hyperelastic variables
    """
    A = inv(F)
    ρ = HYP.ρ0 * det(A)
    Σ = Sigma_hyp(ρ, A, S, HYP)

    σ = sigma(ρ, A, PAR)
    p = trace(σ-Σ)/3
    return ρ, p, A

def solid_IC(vL, vR, FL, FR, SL, SR, HYP, PAR):

    ρL, pL, AL = hyperelastic_vars(FL, SL, HYP, PAR)
    ρR, pR, AR = hyperelastic_vars(FR, SR, HYP, PAR)

    J = zeros(3)

    QL = Cvec(ρL, pL, vL, AL, J, PAR)
    QR = Cvec(ρR, pR, vR, AR, J, PAR)

    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR]

def bartonA_IC():
    """ tf = 0.06
        L = 1
    """
    vL = array([0, 0.5, 1])
    FL = array([[0.98, 0, 0  ],
                [0.02, 1, 0.1],
                [0,    0, 1  ]])
    SL = 0.001

    vR = array([0, 0, 0])
    FR = array([[1, 0, 0  ],
                [0, 1, 0.1],
                [0, 0, 1  ]])
    SR = 0

    return solid_IC(vL, vR, FL, FR, SL, SR, HYP_COP, PAR_COP_SMG)

def bartonB_IC():
    """ tf = 0.06
        L = 1
    """
    vL = array([2, 0, 0.1])
    FL = array([[1,      0,    0   ],
                [-0.01,  0.95, 0.02],
                [-0.015, 0,    0.9 ]])
    SL = 0

    vR = array([0, -0.03, -0.01])
    FR = array([[1,     0,    0  ],
                [0.015, 0.95, 0  ],
                [-0.01, 0,    0.9]])
    SR = 0

    return solid_IC(vL, vR, FL, FR, SL, SR, HYP_COP, PAR_COP_SMG)

def elastic1_IC():
    """ tf = 0.06
        L = 1
    """
    FL = array([[0.95, 0, 0],
                [0,    1, 0],
                [0,    0, 1]])
    vL = zeros(3)
    SL = 0.001

    FR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    return solid_IC(vL, vR, FL, FR, SL, SR, HYP_COP, PAR_COP_SMG)

def elastic2_IC():
    """ tf = 0.06
        L = 1
    """
    FL = array([[0.95, 0, 0],
                [0.05, 1, 0],
                [0,    0, 1]])
    vL = array([0,1,0])
    SL = 0.001

    FR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0

    return solid_IC(vL, vR, FL, FR, SL, SR, HYP_COP, PAR_COP_SMG)

def piston_IC():
    """ tf = 1.5
        L = 1
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
