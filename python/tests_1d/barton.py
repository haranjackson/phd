from itertools import product

from numpy import zeros

from options import nx, ny, nz, nV
from system.gpr.misc.objects import hyperelastic_params, material_parameters
from system.gpr.variables.eos_hyp import total_energy_hyp, density_hyp


def hyperelastic_to_gpr(v, F, S, HYP):
    Q = zeros(nV)
    A = inv(array(F))
    ρ = density_hyp(A)
    Q[0] = ρ
    Q[1] = ρ * total_energy_hyp(A, S, v)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    return Q

def barton_IC():

    HYP = hyperelastic_params(ρ0=8.93, α=1, β=3, γ=2, cv=4e-4, T0=300,
                              b0=2.1, c0=4.6)

    PAR = material_parameters(EOS='smg', ρ0=8.93, cv=4e-4,
                              c0=0.394, Γ0=2, s=1.48, cs=0.219, τ1=inf)

    vL = array([2e3, 0, 100])
    FL = array([[1,      0,    0   ],
                [-0.01,  0.95, 0.02],
                [-0.015, 0,    0.9 ]])

    vR = array([0, -30, -10])
    FR = array([[1,     0,    0  ],
                [0.015, 0.95, 0  ],
                [-0.01, 0,    0.9]])

    QL = hyperelastic_to_gpr(vL, FL, 0, HYP)
    QR = hyperelastic_to_gpr(vR, FR, 0, HYP)
    u = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i*dx < 0.5:
            u[i,j,k] = QL
        else:
            u[i,j,k] = QR

    return u, [PAR]

def purely_elastic1_IC():
    """ tf = 0.06
        L = 1
    """
    ρL = 8.93
    pL = 1
    vL = zeros(3)

    ρR = 8.93
    pR = 1
    vR = zeros(3)

    PAR = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0,
                              cs=1, α=2, μ=1e-2, κ=1e-2)