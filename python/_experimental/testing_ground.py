from numpy import arange, meshgrid
import matplotlib.pyplot as plt

from numba import jit
from numpy import array, dot, einsum, prod, sqrt, zeros
from scipy.integrate import odeint
from scipy.linalg import svd

from system.gpr.misc.objects import material_parameters
from system.gpr.misc.functions import AdevG, det3, gram, gram_rev, inv3, L2_2D
from system.gpr.variables.eos import dEdA, total_energy
from system.gpr.variables.material_functions import theta_1
from system.gpr.misc.structures import Cvec, Cvec_to_Pvec


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
    Asource = - dEdA(A, PAR.cs2) / theta_1(A, PAR.cs2, PAR.τ1)
    return Asource.ravel()

def numerical(A, dt, PAR):
    y0 = A.ravel()
    t = array([0, dt])
    y1 = odeint(f, y0, t, args=(PAR,), Dfun=jac)[1]
    return y1[:9].reshape([3,3])

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

@jit
def stretch_f2(y, t0, k, c):
    ret = zeros(2)
    y0 = y[0]
    y1 = y[1]
    ret[0] = k * y0 * (2*y0 - y1 - c/(y0*y1))
    ret[1] = k * y1 * (2*y1 - y0 - c/(y0*y1))
    return ret

def stretch_solver2(A, dt, PAR):
    U, s, V = svd(A)
    s0 = s**2
    c = prod(s0)
    t = array([0, dt])
    k = -2 * prod(s)**(5/3) / PAR.τ1
    s2 = odeint(stretch_f2, s0[:2], t, args=(k,c))[1]
    s = array([s2[0], s2[1], c/(s2[0]*s2[1])])
    return dot(U*sqrt(s),V)


def plot_surfaces():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = arange(0.5, 2, 0.01)
    Y = arange(0.5, 2, 0.01)
    X, Y = meshgrid(X, Y)
    Z = 1/(X*Y)
    Z2 = 3 - X - Y

    ax.plot_surface(X,Y,Z, color = 'b')
    ax.plot_surface(X,Y,Z2, color='g')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_zlim(0,2)


def generate_pars():
    γ = 1.4
    cv = 2.5
    pinf = 0
    ρ0 = 1
    p0 = 1
    T0 = temperature(ρ0, p0, γ, pinf, cv)
    cs = 2
    μ = 1e-3
    τ1 = 6 * μ / (ρ0 * cs**2)
    α = 1.5
    κ = 1e-4
    τ2 = κ * ρ0 / (T0 * α**2)

    PAR = material_parameters(γ=γ, pINF=pinf, cv=cv, ρ0=ρ0, p0=p0, cs=cs, α=α,
                              μ=μ, κ=κ)
    return PAR

def generate_vecs(PAR):
    A = rand(3,3)
    A /= sign(det(A))
    ρ = det(A)
    p = rand()
    v = rand(3)
    J = rand(3)
    E = total_energy(ρ, p, v, A, J, 0, PAR)

    Q = Cvec(ρ, p, v, A, J, 0, PAR)
    P = Cvec_to_Pvec(Q, PAR)
    return Q, P
