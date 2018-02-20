from itertools import product

from numpy import array, zeros
from scipy.integrate import odeint

from gpr.variables import mg
from gpr.variables.derivatives import dEdA_s, dEdJ
from gpr.variables.eos import E_2A, E_2J, E_3
from gpr.variables.sources import theta1inv, theta2inv
from gpr.misc.functions import det3
from gpr.misc.structures import Cvec_to_Pclass


### DISTORTION ###


def f_A(A, MP):
    ρ = det3(A)
    return - dEdA_s(ρ, A, MP).ravel() * theta1inv(ρ, A, MP)


def jac_A(A, MP):
    """ DOESN'T WORK FOR PLASTIC MATERIALS
    """
    τ1 = MP.τ1
    G = gram(A)
    Grev = gram_rev(A)
    A_devG = AdevG(A, G)
    AinvT = inv(A).T
    AA = einsum('ij,mn', A, A)
    L2A = L2_2D(A)

    ret = 5 / 3 * einsum('ij,mn', A_devG, AinvT) - \
        2 / 3 * AA + AA.swapaxes(1, 3)

    for i in range(3):
        for j in range(3):
            ret[i, j, i, j] -= L2A / 3

    for k in range(3):
        ret[k, :, k, :] += G
        ret[:, k, :, k] += Grev

    ret *= -3 / τ1 * det3(A)**(5 / 3)
    return ret.reshape([9, 9])


### THERMAL IMPULSE ###


def f_J(ρ, E, A, J, v, MP):
    e = E - E_2A(ρ, A, MP) - E_2J(J, MP) - E_3(v)
    T = mg.temperature2(ρ, e, MP)
    return - dEdJ(J, MP) * theta2inv(ρ, T, MP)


def jac_J(ρ, E, A, J, v, MP):
    c1 = E - E_2A(ρ, A, MP) - E_3(v)
    c2 = MP.cα2 / 2
    k = 2 * MP.ρ0 / (MP.τ2 * MP.T0 * ρ * MP.cv)
    a = c1 * k
    b = c2 * k
    return (b * norm(J)**2 - a) / 2 * eye(3) + b * outer(J, J)


### GENERAL ###


def f(y, t0, ρ, E, v, MP):

    ret = zeros(12)
    A = y[:9].reshape([3, 3])

    if MP.VISCOUS:
        ret[:9] = f_A(A, MP)

    if MP.THERMAL:
        J = y[9:]
        ret[9:] = f_J(ρ, E, A, J, v, MP)

    return ret


def jac(y, t0, ρ, E, v, MP):

    ret = zeros([12, 12])
    A = y[:9].reshape([3, 3])

    if MP.VISCOUS:
        ret[:9, :9] = jac_A(A, MP)

    if MP.THERMAL:
        J = y[9:]
        ret[9:, 9:] = jac_J(ρ, E, A, J, v, MP)

    return ret


def ode_stepper_numerical(u, dt, MP, useJac=0):
    """ Full numerical solver for the ODE system
    """
    nx, ny, nz = u.shape[:3]
    y0 = zeros([12])
    for i, j, k in product(range(nx), range(ny), range(nz)):
        Q = u[i, j, k]
        P = Cvec_to_Pclass(Q, MP)
        ρ = P.ρ
        E = P.E
        v = P.v

        y0[:9] = Q[5:14]
        y0[9:] = Q[14:17] / ρ
        t = array([0, dt])

        if useJac:
            y1 = odeint(f, y0, t, args=(ρ, E, v, MP), Dfun=jac)[1]
        else:
            y1 = odeint(f, y0, t, args=(ρ, E, v, MP))[1]
        Q[5:14] = y1[:9]
        Q[14:17] = ρ * y1[9:]
