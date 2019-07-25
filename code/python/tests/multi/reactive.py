from numpy import array, eye, pad, zeros

from gpr.misc.structures import Cvec

from tests.params.alt import VAC
from tests.params.fluids import Air_SG_SI
from tests.params.reactive import NM_CC_SI, C4_JWL_SI
from tests.params.solids import Steel_SMGP_SI, Cu_GRP_SI2


def confined_explosive():
    """ N = 2
        cfl = 0.8
        SPLIT = True
        FLUX = 1

        LSET = 3 / 4
        RIEMANN_STICK = false
        RIEMANN_RELAXATION = true
        STAR_TOL = 1e-8
        PRIM_RECONSTRUCT = true
    """
    BACK_PLATE = False
    AIR_GAP = False
    NITRO = False

    Lx = 0.051
    Ly = 0.09

    nx = 200
    tf = 4.9e-6

    o = 0.0015 if AIR_GAP else 0

    ny = int(nx * Ly / Lx)
    dX = [Lx / nx, Ly / ny]

    MPm = Steel_SMGP_SI
    MPa = Air_SG_SI

    if NITRO:
        MPe = NM_CC_SI
    else:
        MPe = C4_JWL_SI

    pm = 1e5
    pe = 1e5
    pa = 1e5
    v1 = array([700, 0, 0])
    v = zeros(3)
    A = eye(3)

    if AIR_GAP:
        MPs = [MPa, MPe, MPm, MPm, VAC]
        u = zeros([nx, ny, 19])
        u[:, :, -4:] = 1
    else:
        MPs = [MPe, MPm, MPm, VAC]
        u = zeros([nx, ny, 18])
        u[:, :, -3:] = 1

    dX = [Lx / nx, Ly / ny]

    Qm1 = Cvec(MPm.ρ0, pm, v1, MPm, A, λ=0)
    Qm2 = Cvec(MPm.ρ0, pm, v, MPm, A, λ=0)
    Qe = Cvec(MPe.ρ0, pe, v, MPe, A, λ=1)
    Qa = Cvec(MPa.ρ0, pa, v, MPa, A, λ=0)

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # air gap
            if 0.033 <= x <= 0.033 + o:
                u[i, j, :15] = Qa
                u[i, j, -4:] = -1

            # projectile
            elif x <= 0.03 and 0.036 <= y <= 0.054:
                u[i, j, :15] = Qm1
                u[i, j, -1] = -1

            # front plate
            elif 0.03 <= x <= 0.033:
                u[i, j, :15] = Qm2
                u[i, j, -2:] = -1

            # explosive
            elif 0.033 + o <= x <= 0.039 + o:
                u[i, j, :15] = Qe
                u[i, j, -3:] = -1

            # back plate
            elif 0.039 + o <= x:
                if BACK_PLATE:
                    u[i, j, :15] = Qm2
                    u[i, j, -2:] = -1
                else:
                    u[i, j, :15] = Qe
                    u[i, j, -3:] = -1

    #u = u[:, int(ny/2):]

    return u, MPs, tf, dX, 'transitive'


def rod_impact():
    """ N = 2
        cfl = 0.8
        SPLIT = True
        FLUX = 0

        LSET = 3
        RIEMANN_STICK = false
        RIEMANN_RELAXATION = true
        STAR_TOL = 1e-8
        PRIM_RECONSTRUCT = true
    """
    Lx = 0.55
    Ly = 0.18
    nx = 2000
    tf = 60e-6

    nx = int(nx)
    ny = int(nx * Ly / Lx)
    dX = [Lx / nx, Ly / ny]

    MPa = Air_SG_SI
    MPm = Cu_GRP_SI2
    MPe = NM_CC_SI

    pm = 1e5
    pe = 1e5
    pa = 1e5
    v1 = array([2000, 0, 0])
    v = zeros(3)
    A = eye(3)

    MPs = [MPa, MPe, MPm, MPm]
    dX = [Lx / nx, Ly / ny]

    Qm1 = pad(Cvec(MPm.ρ0, pm, v1, MPm, A, λ=0), (0, 3), 'constant')
    Qm2 = pad(Cvec(MPm.ρ0, pm, v, MPm, A, λ=0), (0, 3), 'constant')
    Qe = pad(Cvec(MPe.ρ0, pe, v, MPe, A, λ=1), (0, 3), 'constant')
    Qa = pad(Cvec(MPa.ρ0, pa, v, MPa, A, λ=0), (0, 3), 'constant')

    u = zeros([nx, ny, 18])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # projectile
            if x <= 0.06 and 0.04 <= y <= 0.14:
                u[i, j] = Qm1
                u[i, j, -3] = 1
                u[i, j, -2] = 1
                u[i, j, -1] = 1

            # casing
            elif ((0.06 <= x <= 0.08 or 0.53 <= x <= 0.55) and
                  0.02 <= y <= 0.16) or \
                  (0.08 <= x <= 0.53 and
                   (0.02 <= y <= 0.04 or 0.14 <= y <= 0.16)):
                u[i, j] = Qm2
                u[i, j, -3] = 1
                u[i, j, -2] = 1
                u[i, j, -1] = -1

            # explosive
            elif 0.08 <= x <= 0.53 and 0.04 <= y <= 0.14:
                u[i, j] = Qe
                u[i, j, -3] = 1
                u[i, j, -2] = -1
                u[i, j, -1] = -1

            # air
            else:
                u[i, j] = Qa
                u[i, j, -3] = -1
                u[i, j, -2] = -1
                u[i, j, -1] = -1

    u = u[:, int(ny/2):]

    return u, MPs, tf, dX, 'halfy'
