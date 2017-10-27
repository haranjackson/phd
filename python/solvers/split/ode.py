from itertools import product

from numpy import array, zeros
from scipy.integrate import odeint

from options import NUM_ODE, VISCOUS, THERMAL
from system.gpr.variables.eos import E_3
from system.gpr.misc.structures import Cvec_to_Pclass
from solvers.split.distortion import f_A, jac_A, solver_approximate_analytic
from solvers.split.thermal import f_J, jac_J, solver_thermal_analytic_ideal


def f(y, t0, ρ, E, PAR):

    ret = zeros(12)
    A = y[:9].reshape([3,3])

    if VISCOUS:
        ret[:9] = f_A(A, PAR)

    if THERMAL:
        J = y[9:]
        ret[9:] = f_J(ρ, E, A, J, PAR)

    return ret

def jac(y, t0, ρ, E, PAR):

    ret = zeros([12, 12])
    A = y[:9].reshape([3,3])

    if VISCOUS:
        ret[:9,:9] = jac_A(A, PAR.τ1)

    if THERMAL:
        J = y[9:]
        ret[9:,9:] = jac_J(ρ, E, A, J, PAR)

    return ret

def ode_stepper_numerical(u, dt, PAR, useJac=0):
    """ Full numerical solver for the ODE system
    """
    nx,ny,nz = u.shape[:3]
    y0 = zeros([12])
    for i,j,k in product(range(nx), range(ny), range(nz)):
        Q = u[i,j,k]
        P0 = Cvec_to_Pclass(Q, PAR)
        ρ = P0.ρ
        E = P0.E - E_3(P0.v)

        y0[:9] = Q[5:14]
        y0[9:] = Q[14:17] / ρ
        t = array([0, dt])

        if useJac:
            y1 = odeint(f, y0, t, args=(ρ,E,PAR), Dfun=jac)[1]
        else:
            y1 = odeint(f, y0, t, args=(ρ,E,PAR))[1]
        Q[5:14] = y1[:9]
        Q[14:17] = ρ * y1[9:]

def ode_stepper_analytical(u, dt, PAR):
    """ Solves the ODE analytically by linearising the distortion equations and providing an
        analytic approximation to the thermal impulse evolution
    """
    nx,ny,nz = u.shape[:3]
    for i,j,k in product(range(nx), range(ny), range(nz)):
        Q = u[i,j,k]
        ρ = Q[0]
        A = Q[5:14].reshape([3,3])

        if VISCOUS:
            A1 = solver_approximate_analytic(A, dt, PAR)
            Q[5:14] = A1.ravel()

        if THERMAL:
            J = Q[14:17] / ρ
            E = Q[1] / ρ
            v = Q[2:5] / ρ
            A2 = (A+A1)/2
            Q[14:17] = ρ * solver_thermal_analytic_ideal(ρ, E, A2, J, v, dt, PAR)

def ode_launcher(u, dt, PAR, useJac=0):
    if NUM_ODE:
        ode_stepper_numerical(u, dt, PAR, useJac=useJac)
    else:
        ode_stepper_analytical(u, dt, PAR)
