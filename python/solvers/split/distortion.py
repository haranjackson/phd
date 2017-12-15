from numpy import arctan, argsort, array, dot, cos, einsum, exp, log, prod, sort, sqrt, zeros
from scipy.integrate import odeint
from scipy.linalg import svd, inv

from system.gpr.misc.functions import AdevG, det3, gram, gram_rev, L2_2D
from system.gpr.variables.eos import dEdA
from system.gpr.variables.sources import theta1inv


def f_A(A, PAR):
    return - dEdA(A, PAR).ravel() * theta1inv(A, PAR)

def jac_A(A, τ1):
    G = gram(A)
    Grev = gram_rev(A)
    A_devG = AdevG(A,G)
    AinvT = inv(A).T
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

def pos(x):
    return max(0,x)

def solver_approximate_analytic(A, dt, PAR):
    U, s, V = svd(A)
    detA3 = prod(s)**(1/3)
    s0 = (s/detA3)**2
    m0 = sum(s0) / 3
    u0 = ((s0[0]-s0[1])**2 + (s0[1]-s0[2])**2 + (s0[2]-s0[0])**2) / 3

    if u0 == 0:
        return A

    k = 2 * detA3**7 / PAR.τ1
    τ = k * dt
    e_6τ = exp(-6*τ)
    e_9τ = exp(-9*τ)
    m = 1 + (3*m0-u0/3-3) * e_6τ - (2*m0-u0/3-2) * e_9τ
    u = pos((18*m0-2*u0-18) * e_6τ - (18*m0-3*u0-18) * e_9τ)

    Δ = -2*m**3 + m*u + 2
    arg1 = pos(6*u**3-81*Δ**2)
    θ = arctan(sqrt(arg1)/max(1e-8,9*Δ))

    x1 = sqrt(6*u)/3 * cos(θ/3) + m
    temp2 = 3*m - x1
    arg2 = pos(x1*temp2**2-4)
    x2 = 0.5 * (sqrt(arg2/x1) + temp2)
    x3 = 1/(x1*x2)

    s1 = sort(detA3 * sqrt(array([x1,x2,x3])))
    s = s1[argsort(s)]
    return dot(U*s,V)
