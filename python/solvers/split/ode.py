from itertools import product

from numpy import array, zeros
from scipy.integrate import odeint

from system import source
from options import NV, NUM_ODE

from models.gpr.systems.jacobians import dSdQ
from models.gpr.systems.analytical import ode_solver_analytical


def f(y, t0, *args):
    ret = zeros(NV)
    source(ret, y, *args)
    return ret


def jac(y, t0, *args):
    return dSdQ(y, *args)


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
