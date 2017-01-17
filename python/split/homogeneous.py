from numpy import dot

from options import dx, N1, nx
from ader.basis import derivative_values
from gpr.matrices.conserved import system_conserved


derivs = derivative_values()

def weno_midstepper(wh, dt, PAR, SYS):
    """ Steps the WENO reconstruction forwards by dt/2, under the homogeneous system
    """
    for i in range(nx):
        w = wh[i,0,0]
        dwdx = dot(derivs, w)
        for j in range(N1):
            w[j] -= dt/(2*dx) * dot(system_conserved(w[j], 0, PAR, SYS), dwdx[j])
