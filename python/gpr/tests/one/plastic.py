from numpy import array, eye, zeros

from ader.etc.boundaries import standard_BC

from gpr.misc.structures import Cvec
from gpr.tests.params.solids import Cu_SMGP_SI


def piston_IC():
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
    p = MP.p0
    v = zeros(3)
    A = eye(3)

    Q = Cvec(ρ, p, v, MP, A)

    u = zeros([nx, 14])

    for i in range(nx):
        u[i] = Q

    return u, [MP], tf, dX


def piston_BC(u, N, NDIM):
    ret = standard_BC(u, N, NDIM)
    ret[:N, 2:5] = ret[N, 0] * array([20, 0, 0])
    return ret


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
