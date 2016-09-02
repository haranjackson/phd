from numpy import dot, eye, zeros

from auxiliary.funcs import L2_1D, L2_2D
from gpr.functions import primitive, theta_1, theta_2
from gpr.matrices import jacobian_variables, block, dFdP, dPdQ, source
from gpr.variables import sigma, E_1, E_A, E_J, sigma_A


def dQdP(P, params, jacVars, viscous, thermal, reactive):
    """ Returns the Jacobian of the conserved variables with respect to the primitive variables
    """
    r = P.r; p = P.p; A = P.A; J = P.J; v = P.v; c = P.c; E = P.E
    psi = E_A(A)
    ret = eye(18)
    Y = jacVars.Y

    ret[1, 0] = E - E_1(r, p, params.y, params.pINF)
    ret[1, 1] /= Y
    ret[1, 2:5] = r * v
    ret[2:5, 0] = v
    ret[2:5, 2:5] *= r

    if viscous:
        ret[1, 5:14] = r * psi.ravel(order='F')

    if thermal:
        ret[1, 14:17] = params.alpha2 * r * J
        ret[14:17, 0] = J
        ret[14:17, 14:17] *= r

    if reactive:
        ret[1, 17] = params.Qc * r
        ret[17, 0] = c
        ret[17, 17] *= r

    return ret

def jacobian_primitive(Q, d, params, viscous, thermal, reactive):
    """ Returns the Jacobian in the dth direction for the system of primitive variables, using the
        constituent Jacobian matrices
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    jacVars = jacobian_variables(P, params)
    ret = dot(block(P.v, d, viscous), dQdP(P, params, jacVars, viscous, thermal, reactive))
    ret += dFdP(P, d, params, jacVars, viscous, thermal, reactive)
    return dot(dPdQ(P, params, jacVars, viscous, thermal, reactive), ret)

def jacobian_primitive_direct(Q, d, params, viscous, thermal, reactive):
    """ Returns the Jacobian in the dth direction for the system of primitive variables, calculated
        directly
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    r = P.r; p = P.p; A = P.A; v = P.v; T = P.T
    y = params.y; pINF = params.pINF
    sig = sigma(r, A)
    dsdA = sigma_A(r, A)

    ret = v[d] * eye(18)
    ret[0, 2+d] = r
    ret[1, 2+d] = y * p
    ret[1, 14+d] = (y-1) * params.alpha2 * T
    ret[2+d, 1] = 1 / r

    for i in range(3):
        ret[2+i, 0] = -sig[i, d] / r**2
        ret[2+i, 5:14] = -1 / r * dsdA[i,d].ravel(order='F')
    ret[5+3*d:5+3*(d+1), 2:5] = A

    ret[14+d, 0] = -T / r**2
    ret[14+d, 1] = T / (r * (p + pINF))

    if not reactive:
        ret[17, 17] = 0

    return ret

def jacobian_primitive_reordered(Q, d, params, viscous, thermal, reactive):
    """ Returns the Jacobian in the dth direction for the system of primitive variables, calculated
        directly
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    r = P.r; p = P.p; A = P.A; v = P.v; T = P.T
    y = params.y; pINF = params.pINF
    sig = sigma(r, A)
    dsdA = sigma_A(r, A)

    ret = v[d] * eye(18)
    ret[0, 11+d] = r
    ret[1, 11+d] = y * p
    ret[1, 14+d] = (y-1) * params.alpha2 * T
    ret[2+3*d:5+3*d, 11:14] = A
    ret[11:14, 0] = -sig[d] / r**2
    ret[11+d, 1] = 1 / r

    for i in range(3):
        ret[11+i, 2:11] = -1 / r * dsdA[i,d].ravel(order='F')

    ret[14+d, 0] = -T / r**2
    ret[14+d, 1] = T / (r * (p + pINF))

    if not reactive:
        ret[17, 17] = 0

    return ret

def source_primitive(Q, params):

    S = source(Q, params)
    P = primitive(Q, params)
    jacVars = jacobian_variables(P, params)
    DPDQ = dPdQ(P, params, jacVars)
    return dot(DPDQ, S)

def source_primitive_direct(Q, params, viscous, thermal):

    ret = zeros(18)
    P = primitive(Q, params)
    r = P.r; A = P.A; J = P.J; T = P.T
    y = params.y

    if viscous:
        psi = E_A(A)
        theta1 = theta_1(A)
        ret[1] += (y-1) * r * L2_2D(psi) / theta1
        ret[5:14] = -psi.ravel(order='F') / theta1

    if thermal:
        H = E_J(J)
        theta2 = theta_2(r, T)
        ret[1] += (y-1) * r * L2_2D(H) / theta2
        ret[14:17] = -H / theta2

    return ret

def source_primitive_reordered(Q, params, viscous, thermal):

    ret = zeros(18)
    P = primitive(Q, params)
    r = P.r; A = P.A; J = P.J; T = P.T
    y = params.y

    if viscous:
        psi = E_A(A)
        theta1 = theta_1(A)
        ret[1] += (y-1) * r * L2_2D(psi) / theta1
        ret[2:11] = -psi.ravel(order='F') / theta1

    if thermal:
        H = E_J(J)
        theta2 = theta_2(r, T)
        ret[1] += (y-1) * r * L2_1D(H) / theta2
        ret[14:17] = -H / theta2

    return ret
