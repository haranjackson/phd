from numpy import array, zeros
from scipy.integrate import odeint

import auxiliary
from auxiliary.funcs import det3
from split.ode import linearised_distortion, jac, f


def compare_solvers(A, dt):
    PAR = auxiliary.classes.material_parameters(γ=1.4, pINF=0, cv=1, ρ0=1, p0=1, cs=1, α=1e-16, μ=1e-3, Pr=0.75)
    SYS = auxiliary.classes.active_subsystems(1,1,0,0)
    ρ = det3(A)

    A1 = linearised_distortion(ρ, A, dt, PAR)

    y0 = zeros([12])
    y0[:9] = A.ravel()
    t = array([0, dt])
    y1 = odeint(f, y0, t, args=(ρ,0,PAR,SYS), Dfun=jac)[1]
    A2 = y1[:9].reshape([3,3])

    return A1, A2
