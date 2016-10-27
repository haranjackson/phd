from numpy import array, einsum, eye, zeros
from scipy.integrate import odeint

from auxiliary.funcs import AdevG, det3, gram, gram_rev, inv3, L2_2D
from gpr.variables.eos import E_A
from gpr.variables.material_functions import theta_1, theta_2
from gpr.variables.vectors import conserved, primitive


def A_jac(A, t1):
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

    ret *= -3/t1 * det3(A)**(5/3)
    return ret.swapaxes(0,1).swapaxes(2,3).reshape([9,9])

def jac(y, t0, P0, params):
    ret = zeros([12, 12])
    A = y[:9].reshape([3,3])
    ret[:9,:9] = A_jac(A, params.t1)
    ret[9:,9:] = (P0.T * params.ρ0) / (params.T0 * P0.ρ * params.t1) * eye(3)
    return ret

def f(y, t0, P0, params):

    A = y[:9].reshape([3,3])
    Asource = - E_A(A, params.cs2) / theta_1(A, params)
    J = y[9:]
    Jsource = - P0.ρ * params.α2 * J / theta_2(P0.ρ, P0.T, params)

    ret = zeros(12)
    ret[:9] = Asource.ravel()
    ret[9:] = Jsource

    return ret

def ode_stepper(u, params, subsystems, dt):
    for i in range(len(u)):
        Q = u[i,0,0]
        P0 = primitive(Q, params, subsystems)
        y0 = zeros([12])
        y0[:9] = P0.A.ravel()
        y0[9:] = P0.J
        t = array([0, dt])
        y1 = odeint(f, y0, t, args=(P0,params), Dfun=jac)[1]

        A1 = y1[:9].reshape([3,3])
        J1 = y1[9:]
        u[i,0,0] = conserved(P0.ρ, P0.p, P0.v, A1, J1, P0.λ, params, subsystems)
