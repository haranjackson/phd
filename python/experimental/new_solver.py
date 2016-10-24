from numpy import array, dot, sqrt, zeros
from numpy.polynomial.legendre import leggauss

from gpr.functions import primitive_to_conserved

from options import dx


nodes, weights = leggauss(2)
nodes += 1
nodes /= 2
weights /= 2


c1 = nodes[0]
c2 = nodes[1]

U = sqrt(3) * array([[c2, -c1],
                     [-1,   1]])

Uinv = array([[1, c1],
              [1, c2]])

V = 3 * array([[c2**2, -c1*c2, -c1*c2, c1**2],
               [-c2,       c2,     c1,   -c1],
               [-c2,       c1,     c2,   -c1],
               [  1,       -1,     -1,     1]])

Vinv = array([[1, c1, c1, c1**2],
              [1, c2, c1, c1*c2],
              [1, c1, c2, c1*c2],
              [1, c2, c2, c2**2]])

def new_predictor(wh, params, dt, subsystems):
    """ wh must be reconstruction of primitive variables
    """
    nx = wh.shape[0]
    wh1 = dot(U, wh)
    ret = zeros([nx, 1, 1, 4, 18])
    dtdx = dt/dx
    for i in range(nx):
        q0 = wh1[0,i,0,0]
        qx = wh1[1,i,0,0]
        qt = zeros(18)
        qxt = zeros(18)

        ρ0 = q0[0]
        ρx = qx[0]
        p0 = q0[1]
        px = qx[1]
        u0 = q0[2]
        ux = qx[2]
        γ = params.γ

        qt[0] = -(ρx*u0 + ρ0*ux)
        qxt[0] = -2*ρx*ux
        qt[1] = -(px*u0 + γ*p0*ux)
        qxt[1] = -(1+γ)*px*ux
        qt[2] = -(u0*ux + px/ρ0)
        qxt[2] = -(ρx*qt[2] + ρ0*ux**2 + ρx*u0*ux) / ρ0

        qt *= dtdx
        qxt *= dtdx

        Q = array([q0, qx, qt, qxt])
        Q1 = dot(Vinv, Q)
        for j in range(4):
            Q1[j] = primitive_to_conserved(Q1[j], params, subsystems)

        ret[i,0,0] = Q1

    return ret
