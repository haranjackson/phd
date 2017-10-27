from numpy import dot
from numpy.linalg import solve

from system.gpr.misc.structures import Cvec_to_Pclass
from system.gpr.systems.conserved import block
from system.gpr.systems.primitive import system_primitive
from system.gpr.systems.jacobians import dQdP, dPdQ, dFdP, jacobian_variables


def systems(Q, d, PAR):
    """ Returns the Jacobian in the dth direction
    """
    P = Cvec_to_Pclass(Q, PAR)
    jacVars = jacobian_variables(P, PAR)
    DFDP = dFdP(P, d, jacVars, PAR)
    DPDQ = dPdQ(P, jacVars, PAR)
    DQDP = dQdP(P, PAR)

    M0 = dot(DFDP,DPDQ)+block(Q, d)
    M1 = solve(DQDP, dot(M0, DQDP))
    M2 = system_primitive(Q, d, PAR)
    M3 = system_primitive(Q, d, PAR, pForm=0)
    return M0, M1, M2, M3