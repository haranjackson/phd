from itertools import product

from numpy import array, zeros
from scipy.integrate import odeint

from gpr.variables.eos import E_3
from gpr.variables.vectors import primitive
from split.distortion import f_A, jac_A, solver_distortion_reduced
from split.thermal import f_J, jac_J, solver_thermal_analytic_ideal


def f(y, t0, ρ, E, PAR, SYS):

    ret = zeros(12)
    A = y[:9].reshape([3,3])

    if SYS.viscous:
        ret[:9] = f_A(A, PAR)

    if SYS.thermal:
        J = y[9:]
        ret[9:] = f_J(ρ, E, A, J, PAR)

    return ret

def jac(y, t0, ρ, E, PAR, SYS):

    ret = zeros([12, 12])
    A = y[:9].reshape([3,3])

    if SYS.viscous:
        ret[:9,:9] = jac_A(A, PAR.τ1)

    if SYS.thermal:
        J = y[9:]
        ret[9:,9:] = jac_J(ρ, E, A, J, PAR)

    return ret

def ode_stepper_full(u, dt, PAR, SYS, useJac=0):
    """ Full numerical solver for the ODE system
    """
    nx,ny,nz = u.shape[:3]
    y0 = zeros([12])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        Q = u[i,j,k]
        P0 = primitive(Q, PAR, SYS)
        ρ = P0.ρ
        E = P0.E - E_3(P0.v)

        y0[:9] = Q[5:14]
        y0[9:] = Q[14:17] / ρ
        t = array([0, dt])

        if useJac:
            y1 = odeint(f, y0, t, args=(ρ,E,PAR,SYS), Dfun=jac)[1]
        else:
            y1 = odeint(f, y0, t, args=(ρ,E,PAR,SYS))[1]
        Q[5:14] = y1[:9]
        Q[14:17] = ρ * y1[9:]

def ode_stepper_fast(u, dt, PAR, SYS):
    """ Solves the ODE analytically by linearising the distortion equations and providing an
        analytic approximation to the thermal impulse evolution
    """
    nx,ny,nz = u.shape[:3]
    for i,j,k in product(range(nx), range(ny), range(nz)):
        Q = u[i,j,k]
        ρ = Q[0]
        A = Q[5:14].reshape([3,3])

        if SYS.viscous:
            A1 = solver_distortion_reduced(A, dt, PAR)
            Q[5:14] = A1.ravel()

        if SYS.thermal:
            J = Q[14:17] / ρ
            E = Q[1] / ρ
            v = Q[2:5] / ρ
            A2 = (A+A1)/2
            Q[14:17] = ρ * solver_thermal_analytic_ideal(ρ, E, A2, J, v, dt, PAR)
