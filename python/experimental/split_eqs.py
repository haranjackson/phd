from numpy import array, eye
from scipy.integrate import odeint
from scipy.linalg import svd

from gpr.variables.eos import E_A


ε = array([[0.62,  0.40,  1.14],
           [-0.28, -1.41, 0.59],
           [-0.19, -0.72, -1.28]])

τ = 1.45e-9

A0 = eye(3)

includeSources = 0


def f(y, t):
    A = y.reshape([3,3])
    ret = -dot(A,ε)
    if includeSources:
        ret -= E_A(A,1)/τ
    return ret.ravel()

def ode_stepper(A, dt):
    t = array([0, dt])
    y0 = A.ravel()
    return odeint(f, y0, t, rtol=1e-12, atol=1e-12)[1].reshape([3,3])

def test():
    n = 500
    A = zeros([n,3,3])
    U = zeros([n,3,3])
    V = zeros([n,3,3])
    Σ = zeros([n,3])
    for i in range(n):
        dt = i*5*1e-9/n
        A[i] = ode_stepper(A0,dt)
        U[i], Σ[i], V[i] = svd(Amats[i])

    return A, U, Σ, V
