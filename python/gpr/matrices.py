from numba import jit
from numpy import dot, eye, outer, tensordot, zeros

from auxiliary.funcs import L2_1D, dot3
from gpr.functions import primitive, theta_1, theta_2
from gpr.functions import arrhenius_reaction_rate, discrete_ignition_temperature_reaction_rate
from gpr.variables import heat_flux, sigma, E_1, E_A, sigma_A
from options import reactionType, reactiveEOS


def flux_ref(ret, P, d, params, mechanical, viscous, thermal, reactive):

    r = P.r; p = P.p; v = P.v
    rvd = r*v[d]

    ret[1] = rvd * P.E + p * v[d]

    if mechanical:
        ret[0] = rvd
        ret[2:5] = rvd * v
        ret[2+d] += p

    if viscous:
        A = P.A
        sigd = sigma(r, A, params.cs2)[d]
        ret[1] -= dot3(sigd, v)
        ret[2:5] -= sigd
        ret[5+3*d:8+3*d] = dot(A, v)

    if thermal:
        T = P.T
        J = P.J
        ret[1] += params.alpha2 * T * J[d]
        ret[14:17] = rvd * J
        ret[14+d] += T

    if reactive:
        ret[17] = rvd * P.c

@jit
def block_ref(ret, v, d, viscous):

    if viscous:
        vd = v[d]
        for i in range(5,14):
            ret[i,i] = vd
        k1 = 5+3*d
        for i in range(3):
            vi = v[i]
            k2 = 5+3*i
            for j in range(3):
                ret[k1+j, k2+j] -= vi

def source_ref(ret, P, params, viscous, thermal, reactive):

    r = P.r

    if viscous:
        r = P.r
        A = P.A
        Asource = - E_A(A, params.cs2) / theta_1(A, params)
        ret[5:14] = Asource.ravel(order='F')

    if thermal:
        T = P.T
        Jsource = - r * params.alpha2 * P.J / theta_2(r, T, params)
        ret[14:17] = Jsource

    if reactive:
        if reactionType == 'a':
            K = arrhenius_reaction_rate(r, P.c, T, params)
        elif reactionType == 'd':
            K = discrete_ignition_temperature_reaction_rate(r, P.c, T, params)
        ret[17] = -K

        if not reactiveEOS:
            ret[1] = params.Qc * K


def flux(Q, d, params, mechanical, viscous, thermal, reactive):
    """ Returns the flux matrix in the kth direction
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    ret = zeros(18)
    flux_ref(ret, P, d, params, mechanical, viscous, thermal, reactive)
    return ret

def block(v, d, viscous):
    """ Returns the nonconvervative matrix in the kth direction
    """
    ret = zeros([18, 18])
    block_ref(ret, v, d, viscous)
    return ret

def source(Q, params, viscous, thermal, reactive):
    """ Returns the source vector
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    ret = zeros(18)
    source_ref(ret, P, params, viscous, thermal, reactive)
    return ret

@jit
def B0dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[5] = - v1 * Q[8] - v2 * Q[11]
        ret[6] = - v1 * Q[9] - v2 * Q[12]
        ret[7] = - v1 * Q[10] - v2 * Q[13]
        ret[8] = v0 * Q[8]
        ret[9] = v0 * Q[9]
        ret[10] = v0 * Q[10]
        ret[11] = v0 * Q[11]
        ret[12] = v0 * Q[12]
        ret[13] = v0 * Q[13]

@jit
def B1dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[8] = - v0 * Q[5] - v2 * Q[11]
        ret[9] = - v0 * Q[6] - v2 * Q[12]
        ret[10] = - v0 * Q[7] - v2 * Q[13]
        ret[5] = v1 * Q[5]
        ret[6] = v1 * Q[6]
        ret[7] = v1 * Q[7]
        ret[11] = v1 * Q[11]
        ret[12] = v1 * Q[12]
        ret[13] = v1 * Q[13]

@jit
def B2dot(ret, Q, v, viscous):
    if viscous:
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        ret[11] = - v0 * Q[5] - v1 * Q[8]
        ret[12] = - v0 * Q[6] - v1 * Q[9]
        ret[13] = - v0 * Q[7] - v1 * Q[10]
        ret[5] = v2 * Q[5]
        ret[6] = v2 * Q[6]
        ret[7] = v2 * Q[7]
        ret[8] = v2 * Q[8]
        ret[9] = v2 * Q[9]
        ret[10] = v2 * Q[10]

def Bdot(ret, Q, d, v, viscous):
    if d==0:
        B0dot(ret, Q, v, viscous)
    elif d==1:
        B1dot(ret, Q, v, viscous)
    else:
        B2dot(ret, Q, v, viscous)


class jacobian_variables():

    def __init__(self, prims, params):
        r = prims.r; p = prims.p; A = prims.A; J = prims.J; v = prims.v; E = prims.E
        y = params.y; pINF = params.pINF; cs2 = params.cs2; alpha2 = params.alpha2

        q = heat_flux(prims.T, J, alpha2)
        sig = sigma(r, A, cs2)
        dsdA = sigma_A(r, A, cs2)
        psi = E_A(A, cs2)

        self.Y = params.y - 1
        self.Psi = r * outer(v, v) - sig
        self.Omega = (E - E_1(r, p, y, pINF)) * v - (dot(sig, v) + q) / r
        self.Upsilon = self.Y * (L2_1D(v) + alpha2 * L2_1D(J) + E_1(r, p, y, pINF) - E)
        self.Phi = r * outer(v, psi).reshape([3,3,3])
        self.Phi -= tensordot(v, dsdA, axes=(0,0))

def dPdQ(P, params, jacVars, viscous, thermal, reactive):
    """ Returns the Jacobian of the primitive variables with respect to the conserved variables
    """
    r = P.r; A = P.A; J = P.J; v = P.v; c = P.c
    r_1 = 1 / r
    psi = E_A(A, params.cs2)
    ret = eye(18)
    Y, Upsilon = jacVars.Y, jacVars.Upsilon

    ret[1, 0] = Upsilon
    ret[1, 1] = Y
    ret[1, 2:5] = -Y * v
    ret[2:5, 0] = -v / r
    for i in range(2,5):
        ret[i, i] = r_1

    if viscous:
        ret[1, 5:14] = -Y * r * psi.ravel(order='F')

    if thermal:
        ret[1, 14:17] = -Y * params.alpha2 * J
        ret[14:17, 0] = -J / r
        for i in range(14,17):
            ret[i, i] = r_1

    if reactive:
        ret[17, 0] = -c / r
        ret[17, 17] /= r

        if reactiveEOS:
            ret[1, 0] += Y * params.Qc * c
            ret[1, 17] -= Y * params.Qc

    return ret

def dFdP(P, d, params, jacVars, viscous, thermal, reactive):
    """ Returns the Jacobian of the flux vector with respect to the primitive variables
    """
    r = P.r; p = P.p; A = P.A; J = P.J; v = P.v; c = P.c; E = P.E; T = P.T
    y = params.y; pINF = params.pINF; cs2 = params.cs2; alpha2 = params.alpha2
    rvd = r * v[d]

    q = heat_flux(T, J, alpha2)
    dsdA = sigma_A(r, A, cs2)
    Psi, Phi, Omega, Y = jacVars.Psi, jacVars.Phi, jacVars.Omega, jacVars.Y

    ret = zeros([18, 18])
    ret[0, 0] = v[d]
    ret[0, 2+d] = r
    ret[1, 0] = Omega[d]
    ret[1, 1] = y * v[d] / Y + q[d] / (p + pINF)
    ret[1, 2:5] = Psi[d]
    ret[1, 2+d] += r * E + p
    ret[2:5, 0] = Psi[d] / r
    for i in range(2,5):
        ret[i, i] = rvd
    ret[2:5, 2+d] += r * v
    ret[2+d, 1] = 1

    if viscous:
        ret[1, 5:14] = Phi[d].ravel(order='F')
        ret[2:5, 5:14] = -dsdA[d].reshape([3,9], order='F')
        k1 = 5+3*d
        ret[k1:k1+3, 2:5] = A
        for i in range(3):
            vi = v[i]
            k2 = 5+3*i
            for j in range(3):
                ret[k1+j, k2+j] = vi

    if thermal:
        ret[1, 14:17] = alpha2 * rvd * J
        ret[1, 14+d] += alpha2 * T
        ret[14:17, 0] = v[d] * J
        ret[14+d, 0] -= T / r
        ret[14+d, 1] = T / (p+pINF)
        ret[14:17, 2+d] = r * J
        for i in range(14,17):
            ret[i, i] = rvd

    if reactive:
        ret[17, 0] = v[d] * c
        ret[17, 2+d] = r * c
        ret[17, 17] = rvd

        if reactiveEOS:
            ret[1, 17] += params.Qc * rvd

    return ret

def jacobian(Q, d, params, viscous, thermal, reactive):
    """ Returns the Jacobian in the dth direction
    """
    P = primitive(Q, params, viscous, thermal, reactive)
    jacVars = jacobian_variables(P, params)
    DFDP = dFdP(P, d, params, jacVars, viscous, thermal, reactive)
    DPDQ = dPdQ(P, params, jacVars, viscous, thermal, reactive)
    return dot(DFDP, DPDQ) + block(P.v, d, viscous)
