from matplotlib.pyplot import figure, plot
from numpy import array, eye, pad, zeros

from gpr.misc.structures import Cvec, State

from test.params.alt import VAC
from test.params.solids import Al_GRP_SI, W_SMGP_SI, Steel_SMGP_SI, Al_SMGP_CGS


def gauge_plot(uList, MPs):

    n = len(uList)
    nx, ny = uList[0].shape[:2]

    ρy = zeros([5, n])
    py = zeros([5, n])

    j0 = int(ny / 2)
    for i in range(5):
        i0 = int((0.0018125 + 0.006 + 0.003625 * i) / 0.03 * nx)
        for j in range(n):
            Q = uList[j][i0, j0]
            P = State(Q, MPs[1])
            ρy[i, j] = P.ρ
            py[i, j] = P.p()

    figure(1)
    for i in range(5):
        plot(ρy[i])

    figure(2)
    for i in range(5):
        plot(py[i])


def aluminium_plates():
    """ N = 2
        cfl = 0.5
        SPLIT = True
        FLUX = 0

        LSET = 2
        RIEMANN_RELAXATION = false
    """
    MP = Al_GRP_SI

    Lx = 0.03
    Ly = 0.04
    nx = 600
    ny = 800
    tf = 5e-6

    ρ = MP.ρ0
    p = 0
    v0 = array([400., 0., 0.])
    v1 = zeros(3)
    A = eye(3)

    MPs = [VAC, MP, MP]
    dX = [Lx / nx, Ly / ny]

    Q0 = pad(Cvec(ρ, p, v0, MP, A), (0, 2), 'constant')
    Q1 = pad(Cvec(ρ, p, v1, MP, A), (0, 2), 'constant')

    u = zeros([nx, ny, 16])

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

    return u, MPs, tf, dX, 'transitive'


def rod_penetration():
    """ N = 2
        cfl = 0.5
        SPLIT = True
        FLUX = 0
    """
    D = 0.029
    V = -1250
    # D = 0.0495
    # V = -1700

    o = 0.01
    nx = 200

    Lx = 0.06
    Ly = 0.05 + 1.5 * D

    ny = int(nx * Ly / Lx)
    dX = [Lx / nx, Ly / ny]
    tf = 20e-6 # 80e-6

    MP1 = W_SMGP_SI
    MP2 = Steel_SMGP_SI

    p = 0
    v1 = array([0, V, 0])
    v2 = zeros(3)
    A = eye(3)

    MPs = [VAC, MP1, MP2]
    dX = [Lx / nx, Ly / ny]

    Q1 = pad(Cvec(MP1.ρ0, p, v1, MP1, A), (0, 2), 'constant')
    Q2 = pad(Cvec(MP2.ρ0, p, v2, MP2, A), (0, 2), 'constant')

    u = zeros([nx, ny, 16])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # projectile
            if 0.028 <= x <= 0.032 and D + o <= y <= D + 0.05 + o:
                u[i, j] = Q1
                u[i, j, -2] = 1
                u[i, j, -1] = -1

            # plate
            elif o <= y <= D + o:
                u[i, j] = Q2
                u[i, j, -2] = 1
                u[i, j, -1] = 1

            # vacuum
            else:
                u[i, j, -2] = -1
                u[i, j, -1] = -1

    #u = u[int(nx/2):]

    return u, MPs, tf, dX, 'transitive'


def taylor_bar():
    """ N = 2
        cfl = 0.8
        SPLIT = True
        FLUX = 0

        LSET = 1
        DESTRESS = false
        RIEMANN_RELAXATION = true
    """
    Lx = 200
    Ly = 550
    tf = 5e3
    nx = 200
    ny = int(nx * Ly / Lx)
    dX = [Lx / nx, Ly / ny]

    p = 0
    v = array([0, -0.015, 0])
    A = eye(3)

    MP = Al_SMGP_CGS
    MPs = [VAC, MP]

    Q = pad(Cvec(MP.ρ0, p, v, MP, A), (0, 1), 'constant')

    u = zeros([nx, ny, 15])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]

            # projectile
            if Lx / 4 <= x <= 3 * Lx / 4 and y <= 5.5 * Ly / 6:
                u[i, j] = Q
                u[i, j, -1] = 1

            # vacuum
            else:
                u[i, j, -1] = -1

    #u = u[int(nx/2):]

    return u, MPs, tf, dX, 'slip'