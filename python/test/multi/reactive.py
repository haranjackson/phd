from numpy import array, eye, pad, zeros

from gpr.misc.structures import Cvec

from test.params.alt import VAC
from test.params.reactive import NM_JWL_SI
from test.params.solids import Steel_SMGP_SI


def confined_explosive():
    """ LSET = 3
    """
    Lx = 0.06271
    Ly = 0.1
    nx = 200
    tf = 4.9e-6

    ny = int(nx * Ly / Lx)
    dX = [Lx / nx, Ly / ny]

    MPm = Steel_SMGP_SI
    MPe = NM_JWL_SI

    pm = 0
    pe = 1e5
    v1 = array([1800, 0, 0])
    v = zeros(3)
    A = eye(3)

    MPs = [VAC, MPe, MPm, MPm]
    dX = [Lx / nx, Ly / ny]

    Qm1 = pad(Cvec(MPm.ρ0, pm, v1, MPm, A), (0, 3), 'constant')
    Qm2 = pad(Cvec(MPm.ρ0, pm, v, MPm, A), (0, 3), 'constant')
    Qe = pad(Cvec(MPe.ρ0, pe, v, MPe, A), (0, 3), 'constant')

    u = zeros([nx, ny, 17])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # projectile
            if x <= 0.05 and 0.041 <= y <= 0.059:
                u[i, j] = Qm1
                u[i, j, -3] = 1
                u[i, j, -2] = 1
                u[i, j, -1] = 1

            # plates
            elif 0.05 <= x <= 0.05318 or 0.05953 <= x:
                u[i, j] = Qm2
                u[i, j, -3] = 1
                u[i, j, -2] = 1
                u[i, j, -1] = -1

            # explosive
            elif 0.05318 <= x <= 0.05953:
                u[i, j] = Qe
                u[i, j, -3] = 1
                u[i, j, -2] = -1
                u[i, j, -1] = -1

            # vacuum
            else:
                u[i, j, -3] = -1
                u[i, j, -2] = -1
                u[i, j, -1] = -1

    #u = u[int(nx/2):]

    return u, MPs, tf, dX, 'transitive'
