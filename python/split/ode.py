from numpy import array, einsum, exp, eye, log, sqrt, zeros
from scipy.integrate import odeint

from auxiliary.funcs import AdevG, det3, gram, gram_rev, inv3, L2_2D, tr
from gpr.variables.eos import E_2A, E_3, E_A, E_J, energy_to_temperature
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.vectors import primitive


def A_jac(A, τ1):
    G = gram(A)
    Grev = gram_rev(A)
    A_devG = AdevG(A,G)
    AinvT = inv3(A).T
    AA = einsum('ij,mn', A, A)
    L2A = L2_2D(A)

    ret = 5/3 * einsum('ij,mn', A_devG, AinvT) - 2/3 * AA + AA.swapaxes(1,3)

    for i in range(3):
        for j in range(3):
            ret[i,j,i,j] -= L2A / 3

    for k in range(3):
        ret[k,:,k,:] += G
        ret[:,k,:,k] += Grev

    ret *= -3/τ1 * det3(A)**(5/3)
    return ret.reshape([9,9])

def jac(y, t0, ρ, E, PAR, SYS):

    ret = zeros([12, 12])
    A = y[:9].reshape([3,3])

    if SYS.viscous:
        ret[:9,:9] = A_jac(A, PAR.τ1)

    if SYS.thermal:
        J = y[9:]
        T = energy_to_temperature(E, A, J, PAR)
        ret[9:,9:] = -(T * PAR.ρ0) / (PAR.T0 * ρ * PAR.τ1) * eye(3)

    return ret

def f(y, t0, ρ, E, PAR, SYS):

    ret = zeros(12)
    A = y[:9].reshape([3,3])

    if SYS.viscous:
        Asource = - E_A(A, PAR.cs2) / theta_1(A, PAR.cs2, PAR.τ1)
        ret[:9] = Asource.ravel()

    if SYS.thermal:
        J = y[9:]
        T = energy_to_temperature(E, A, J, PAR)
        ret[9:] = - E_J(J, PAR.α2) / theta_2(ρ, T, PAR.ρ0, PAR.T0, PAR.α2, PAR.τ2)

    return ret

def ode_stepper_full(u, dt, PAR, SYS):
    """ Full numerical solver for the ODE system
    """
    for i in range(len(u)):
        Q = u[i,0,0]
        P0 = primitive(Q, PAR, SYS)
        ρ = P0.ρ
        E = P0.E - E_3(P0.v)

        y0 = zeros([12])
        y0[:9] = Q[5:14]
        y0[9:] = Q[14:17] / ρ
        t = array([0, dt])

        y1 = odeint(f, y0, t, args=(ρ,E,PAR,SYS))[1]
#        y1 = odeint(f, y0, t, args=(ρ,E,PAR,SYS), Dfun=jac)[1]
        u[i,0,0,5:14] = y1[:9]
        u[i,0,0,14:17] = ρ * y1[9:]


def linearised_distortion(ρ, A, dt, PAR):
    """ A linearised solver for the distortion ODE
    """
    diff = tr(A)/3 * eye(3)
    ret1 = 0.5 * (A - A.T) + diff
    ret2 = 0.5 * (A + A.T) - diff
    return ret1 + exp(-6*dt/PAR.τ1 * (ρ/PAR.ρ0)**(7/3)) * ret2

def simple_analytical_thermal_solver(ρ, Q, dt, PAR, SYS):
    """ Returns the analytic solution to the thermal impulse ODE, assuming pressure is constant
        over the timescale of the ODE.
        NB This may not be a good assumption.
    """
    P0 = primitive(Q, PAR, SYS)
    return ρ * exp(-(P0.T * PAR.ρ0 * dt)/(PAR.T0 * ρ * PAR.τ2)) * P0.J

def analytic_thermal_solver(ρ, E, A, Ji, v, dt, PAR):
    """ Solves the thermal impulse ODE analytically in 1D for the ideal gas EOS
    """
    cv = PAR.cv
    c1 = (E - E_2A(A, PAR.cs2) - E_3(v)) / cv
    c2 = PAR.α2 / (2 * cv)
    k = PAR.ρ0 / (PAR.τ2 * PAR.T0 * ρ)
    c1 *= k
    c2 *= k
    c = log(c1/Ji**2 - c2) / (2 * c1)
    if Ji > 0:
        return sqrt(c1 / (exp(2*c1*(c+dt)) + c2))
    else:
        return -sqrt(c1 / (exp(2*c1*(c+dt)) + c2))

def ode_stepper(u, dt, PAR, SYS):
    """ Solves the ODE analytically by linearising the distortion equations and providing an
        analytic approximation to the thermal impulse evolution
    """
    for i in range(len(u)):
        Q = u[i,0,0]
        ρ = Q[0]
        A = Q[5:14].reshape([3,3])

        if SYS.viscous:
            A1 = linearised_distortion(ρ, A, dt, PAR)
            u[i,0,0,5:14] = A1.ravel()

        if SYS.thermal:
            u[i,0,0,14:17] = simple_analytical_thermal_solver(ρ, Q, dt, PAR, SYS)
#            E = Q[1] / ρ
#            J1 = Q[14] / ρ
#            v = Q[2:5] / ρ
#            A2 = (A+A1)/2
#            u[i,0,0,14] = ρ * analytic_thermal_solver(ρ, E, A2, J1, v, dt, PAR)
