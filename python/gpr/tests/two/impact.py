from numpy import array, eye, zeros

from gpr.misc.structures import Cvec
from gpr.opts import VISCOUS, THERMAL, REACTIVE, MULTI, LSET, NV
from gpr.tests.params.alt import VAC
from gpr.tests.params.solids import Al_GRP_SI


def aluminium_plates_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 2)

    MP = Al_GRP_SI

    Lx = 0.03
    Ly = 0.04
    nx = 300  # 6000
    ny = 400  # 8000
    tf = 5e-6  # 2.919e-06

    ρ = MP.ρ0
    p = 0
    v0 = array([400., 0., 0.])
    v1 = zeros(3)
    A = eye(3)
    J = zeros(3)

    MPs = [VAC, MP, MP]
    dX = [Lx / nx, Ly / ny]

    Q0 = Cvec(ρ, p, v0, A, J, MP)
    Q1 = Cvec(ρ, p, v1, A, J, MP)

    u = zeros([nx, ny, NV])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # projectile
            if 0.001 <= x <= 0.006 and 0.014 <= y <= 0.026:
                u[i, j] = Q0
                u[i, j, -2] = 1
                u[i, j, -1] = -1

            # plate
            elif 0.006 <= x <= 0.028 and 0.003 <= y <= 0.037:
                u[i, j] = Q1
                u[i, j, -2] = 1
                u[i, j, -1] = 1

            # vacuum
            else:
                u[i, j, -2] = -1
                u[i, j, -1] = -1

    return u, MPs, tf, dX


def rod_penetration_IC():

    D = 0.029
    # D = 0.0495
    V = 1250
    # V = 1700

    ny = 200

    Lx = 0.05 + 1.5 * D
    Ly = 0.4

    nx = ny * Lx / Ly
    dX = [Lx / nx, Ly / ny]
