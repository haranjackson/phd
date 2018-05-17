from numpy import array, eye, zeros

from gpr.misc.structures import Cvec
from gpr.opts import VISCOUS, THERMAL, REACTIVE, MULTI, LSET, NV
from gpr.tests.params import MP_VAC, MP_Al_GR_SI


def aluminium_plate_impact_IC():

    assert(VISCOUS)
    assert(not THERMAL)
    assert(not REACTIVE)
    assert(not MULTI)
    assert(LSET == 2)

    Lx = 0.03
    Ly = 0.04
    nx = 60
    ny = 80
    tf = 5e-6

    ρ = MP_Al_GR_SI.ρ0
    p = 0
    v0 = array([400., 0., 0.])
    v1 = zeros(3)
    A = eye(3)
    J = zeros(3)

    MPs = [MP_VAC, MP_Al_GR_SI, MP_Al_GR_SI]
    dX = [Lx / nx, Ly / ny]

    Q0 = Cvec(ρ, p, v0, A, J, MP_Al_GR_SI)
    Q1 = Cvec(ρ, p, v1, A, J, MP_Al_GR_SI)

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
