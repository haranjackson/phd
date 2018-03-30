from itertools import product

from numpy import array
from scipy.integrate import odeint

from system import source, ode_solver_analytical, source_jacobian
from options import NUM_ODE


def f(y, t0, *args):
    return source(y, *args)


def jac(y, t0, *args):
    return source_jacobian(y, *args)


def ode_solver_numerical(Q, dt, *args):
    """ Full numerical solver for the ODE system
    """
    USE_JACOBIAN = False

    y0 = Q.copy()
    t = array([0, dt])

    if USE_JACOBIAN:
        Q[:] = odeint(f, y0, t, args=args, Dfun=jac)[1]
    else:
        Q[:] = odeint(f, y0, t, args=args)[1]


def ode_launcher(u, dt, *args):

    for coords in product(*[range(s) for s in u.shape[:-1]]):
        if NUM_ODE:
            ode_solver_numerical(u[coords], dt, *args)
        else:
            ode_solver_analytical(u[coords], dt, *args)
