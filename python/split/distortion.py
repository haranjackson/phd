from numba import jit
from numpy import arctan, array, dot, einsum, log, prod, sqrt, zeros
from scipy.integrate import odeint
from scipy.linalg import svd

from auxiliary.funcs import AdevG, det3, gram, gram_rev, inv3, L2_2D
from gpr.variables.eos import E_A
from gpr.variables.material_functions import theta_1


def f_A(A, PAR):
    return - E_A(A, PAR.cs2).ravel() / theta_1(A, PAR.cs2, PAR.τ1)

def jac_A(A, τ1):
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

@jit
def f_reduced(y, t0, k, c):
    ret = zeros(2)
    y0 = y[0]
    y1 = y[1]
    ret[0] = k * y0 * (2*y0 - y1 - c/(y0*y1))
    ret[1] = k * y1 * (2*y1 - y0 - c/(y0*y1))
    return ret

def solver_distortion_reduced(A, dt, PAR):
    U, s, V = svd(A)
    s0 = s**2
    c = prod(s0)
    t = array([0, dt])
    k = -2 * prod(s)**(5/3) / PAR.τ1
    s2 = odeint(f_reduced, s0[:2], t, args=(k,c))[1]
    s = array([s2[0], s2[1], c/(s2[0]*s2[1])])
    return dot(U*sqrt(s),V)

def bound_f(x, l):
    return log((x**2+l*x+l**2) / (x-l)**2) - 2*sqrt(3)*arctan((2*x+l) / (sqrt(3)*l))
