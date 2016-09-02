from numpy import eye, zeros

from auxiliary.common import material_parameters
from gpr.functions import conserved
from options import nx, ny, nz, y, pINF


defaultParams = material_parameters()


def standard_IC():
    u = zeros([nx, ny, nz, 18])
    r = 1
    v = zeros(3)
    p = 1
    A = eye(3)
    J = zeros(3)
    c = 1
    Q = conserved(r, p, v, A, J, c, y, pINF)
    for i in range(nx):
        u[i,0,0] = Q
    return u, [defaultParams], []
