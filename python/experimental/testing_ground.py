from numba import jit
from numpy import array, dot, einsum, prod, sqrt, zeros
from scipy.integrate import odeint
from scipy.linalg import svd

import auxiliary
from auxiliary.funcs import AdevG, det3, gram, gram_rev, inv3, L2_2D
from gpr.variables.eos import E_A
from gpr.variables.material_functions import theta_1
from split.ode import linearised_distortion


def jac(y, t0, PAR):
    A = y.reshape([3,3])
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

    ret *= -3/PAR.τ1 * det3(A)**(5/3)
    return ret.reshape([9,9])

def f(y, t0, PAR):
    A = y.reshape([3,3])
    Asource = - E_A(A, PAR.cs2) / theta_1(A, PAR.cs2, PAR.τ1)
    return Asource.ravel()

def numerical(A, dt, PAR):
    y0 = A.ravel()
    t = array([0, dt])
    y1 = odeint(f, y0, t, args=(PAR,), Dfun=jac)[1]
    return y1[:9].reshape([3,3])

def compare_solvers(A, dt):
    PAR = auxiliary.classes.material_parameters(γ=1.4,pINF=0,cv=1,ρ0=1,p0=1,cs=1,α=1e-16,μ=1e-3,Pr=3/4)
    ρ = det3(A)

    A1 = linearised_distortion(ρ, A, dt, PAR)
    A2 = numerical(A, dt, PAR)
    return A1, A2

@jit
def stretch_f(y, t0, k):
    ret = zeros(3)
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    ret[0] = k * y0 * (2*y0 - y1 - y2)
    ret[1] = k * y1 * (2*y1 - y2 - y0)
    ret[2] = k * y2 * (2*y2 - y0 - y1)
    return ret

def stretch_solver(A, dt, PAR):
    U, s, V = svd(A)
    s0 = s**2
    t = array([0, dt])
    k = -2 * prod(s)**(5/3) / PAR.τ1
    s2 = odeint(stretch_f, s0, t, args=(k,))[1]
    return dot(U*sqrt(s2),V)
