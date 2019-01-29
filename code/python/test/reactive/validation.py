from numpy import eye, zeros

from gpr.misc.structures import Cvec
from test.params.reactive import NM_CC_SI, C4_JWL_SI


def pop_plot():
    """ 10.1016/j.jcp.2018.03.037
        5.3. C4 Pop-plot
    """
    tf = 1e-7
    nx = 2000
    Lx = 6.4e-2
    L0 = 0.04e-2

    MP = C4_JWL_SI

    ρ = 1590
    p = 1e5
    pb = 30e9
    v = zeros(3)
    A = eye(3)

    Q = Cvec(ρ, p, v, MP, A, λ=1)
    Qb = Cvec(ρ, pb, v, MP, A, λ=0)

    dX = [Lx / nx]

    u = zeros([nx, 15])
    for i in range(nx):
        x = (i+0.5) * dX[0]
        if x < L0:
            u[i] = Qb
        else:
            u[i] = Q

    return u, [MP], tf, dX, 'transitive'
