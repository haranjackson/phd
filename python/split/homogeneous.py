from numpy import dot, zeros

from options import dx, N1, nx
from ader.basis import derivative_values
from gpr.matrices.conserved import B0dot, flux, system_conserved


derivs = derivative_values()

def weno_midstepper(wh, dt, PAR, SYS):
    """ Steps the WENO reconstruction forwards by dt/2, under the homogeneous system
    """
    F = zeros([N1, 18])
    Bdwdx = zeros(18)
    for i in range(nx):
        w = wh[i,0,0]
        dwdx = dot(derivs, w)

        for j in range(N1):
            F[j] = flux(w[j], 0, PAR, SYS)
        dFdx = dot(derivs, F)
        for j in range(N1):
            v = w[j,2:5] / w[j,0]
            B0dot(Bdwdx, dwdx[j], v)
            w[j] -= dt/(2*dx) * (dFdx[j] + Bdwdx)

#        for j in range(N1):
#            w[j] -= dt/2 * dot(system_conserved(w[j],0,PAR,SYS), dwdx[j]/dx)
