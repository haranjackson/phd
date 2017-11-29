from itertools import product

from numpy import array, zeros
from numpy.linalg import inv

from system.gpr.misc.objects import hyperelastic_params, material_parameters
from system.gpr.variables.eos_hyp import total_energy_hyp, density_hyp
from tests_1d.common import PAR_COP, HYP_COP
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
    """ tf = 0.6e-6
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

def purely_elastic1_IC():
    """ tf = 0.06
        L = 1
    """
    ρL = 8.93
    AL = array([[0.95, 0, 0],
                [0,    0, 0],
                [0,    0, 1]])
    vL = zeros(3)
    SL = 0.001

    ρR = 8.93
    AR = array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    vR = zeros(3)
    SR = 0
