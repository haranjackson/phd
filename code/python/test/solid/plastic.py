from numpy import eye, sqrt, zeros

from gpr.misc.structures import Cvec

from test.params.solids import Cu_SMGP_SI, Cu_GRP_SI


def piston_exact(nx, var):

    eWaveSpeed = 4722
    pWaveSpeed = 3977
    tf = 1.5e-4

    if var == 'density':
        e = 8938.9
        p = 8973.5
        u = 8930
    elif var == 'velocity':
        e = 4.72
        p = 20
        u = 0
    elif var == 'pressure':
        e = 139.03e6
        p = 681.59e6
        u = 0
    elif var == 'stress':
        e = 199.03e6
        p = 741.59e6
        u = 0

    ei = int(eWaveSpeed * tf * nx)
    pi = int(pWaveSpeed * tf * nx)
    y = zeros(nx)
    y[:pi] = p
    y[pi:ei] = e
    y[ei:] = u
    return y


def piston():
    """ http://arxiv.org/abs/1806.00706
        6.1 Elasto-plastic piston

        N = 3
        cfl = 0.5
        SPLIT = True
        SOLVER = 'roe'
    """
    tf = 1.5e-4
    nx = 400
    Lx = 1

    dX = [Lx / nx]

    MP = Cu_SMGP_SI
    ρ = MP.ρ0
    p = 0
    v = zeros(3)
    A = eye(3)

    Q = Cvec(ρ, p, v, MP, A)

    u = zeros([nx, 14])

    for i in range(nx):
        u[i] = Q

    return u, [MP], tf, dX, 'piston_bc'


def cylindrical_shock():
    """ 10.1002/nme.2695
        6.2. Two-dimensional test case

        N = 4
        cfl = 0.8
        SPLIT = True
        SOLVER = 'roe'
    """
    tf = 10e-6
    nx = 500
    ny = 500
    Lx = 0.2
    Ly = 0.2

    dX = [Lx / nx, Ly / ny]

    MP = Cu_GRP_SI

    ρi = MP.ρ0
    pi = 0
    Ai = eye(3)

    ρo = 9375
    po = 10e9
    Ao = (ρo / MP.ρ0)**(1/3) * eye(3)

    v = zeros(3)

    Qi = Cvec(ρi, pi, v, MP, Ai)
    Qo = Cvec(ρo, po, v, MP, Ao)

    u = zeros([nx, ny, 14])

    for i in range(nx):
        for j in range(ny):
            x = (i+0.5) * dX[0]
            y = (j+0.5) * dX[1]
            r = sqrt((x - Lx / 2)**2 + (y - Ly / 2)**2)

            if r > 0.02:
                u[i, j] = Qo
            else:
                u[i, j] = Qi

    return u[:, 250:], [MP], tf, dX, 'slip'
