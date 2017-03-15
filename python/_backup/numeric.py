from numpy import dot

from gpr.matrices.conserved import block, source
from gpr.matrices.jacobians import jacobian_variables, dFdP, dPdQ, dQdP
from gpr.variables.vectors import primitive


def system_primitive_numeric(Q, d, params, subsystems):
    """ Returns the systems matrix in the dth direction for the system of primitive variables, using
        the constituent Jacobian matrices
    """
    P = primitive(Q, params, subsystems)
    jacVars = jacobian_variables(P, params)
    ret = dot(block(P.v, d, subsystems.viscous), dQdP(P, params, jacVars, subsystems))
    ret += dFdP(P, d, params, jacVars, subsystems)
    return dot(dPdQ(P, params, jacVars, subsystems), ret)

def source_primitive_numeric(Q, params):

    S = source(Q, params)
    P = primitive(Q, params)
    jacVars = jacobian_variables(P, params)
    DPDQ = dPdQ(P, params, jacVars)
    return dot(DPDQ, S)
