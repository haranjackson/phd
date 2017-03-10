from numpy import array, cos, diag, eye, mod, outer, sin, tan
from scipy.integrate import odeint
from scipy.linalg import svd

from auxiliary.funcs import dev
from gpr.variables.eos import E_A


ε = array([[0.62,  0.40,  1.14],
           [-0.28, -1.41, 0.59],
           [-0.19, -0.72, -1.28]])

τ = 1.45e-9

tScale = 1
includeSources = 1
n = 50


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
    A = zeros([n,3,3])
    U = zeros([n,3,3])
    V = zeros([n,3,3])
    Σ = zeros([n,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        A[i] = ode_stepper(eye(3),dt)
        U[i], Σ[i], V[i] = svd(A[i])
        V[i] = V[i].T

    return A, U, Σ, V

def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1

def lim(x):
    TOL = 1e-11
    if abs(x) < TOL:
        return sgn(x) * TOL
    else:
        return x

def f2(y, t):
    λ1 = y[0]
    λ2 = y[1]
    λ3 = y[2]
    v1 = y[3:6]
    v2 = y[6:9]
    v3 = y[9:12]

    b1 = -dot(v3, dot(λ3*ε + λ2*ε.T, v2)) / lim(λ2-λ3)
    b2 = -dot(v1, dot(λ1*ε + λ3*ε.T, v3)) / lim(λ3-λ1)
    b3 = -dot(v2, dot(λ2*ε + λ1*ε.T, v1)) / lim(λ1-λ2)

    ret = zeros(12)
    ret[0] = -2 * dot(v1,dot(ε,v1))
    ret[1] = -2 * dot(v2,dot(ε,v2))
    ret[2] = -2 * dot(v3,dot(ε,v3))
    ret[3:6]  = b3*v2 - b2*v3
    ret[6:9]  = b1*v3 - b3*v1
    ret[9:12] = b2*v1 - b1*v2

    if includeSources:
        Λ = diag(y[:3])
        ret[:3] -= 2/τ * diag(dot(Λ, dev(Λ)))

    return ret

def rhs1(λ,V):
    ret = zeros(9)
    λ1 = λ[0]
    λ2 = λ[1]
    λ3 = λ[2]
    v1 = V[:,0]
    v2 = V[:,1]
    v3 = V[:,2]

    b1 = -dot(v3, dot(λ3*ε + λ2*ε.T, v2)) / lim(λ2-λ3)
    b2 = -dot(v1, dot(λ1*ε + λ3*ε.T, v3)) / lim(λ3-λ1)
    b3 = -dot(v2, dot(λ2*ε + λ1*ε.T, v1)) / lim(λ1-λ2)

    ret[:3]  = b3*v2 - b2*v3
    ret[3:6] = b1*v3 - b3*v1
    ret[6:9] = b2*v1 - b1*v2
    return ret

def mat(λ,V,i,a):
    j = mod(i+1,3)
    k = mod(i+2,3)
    Vi = V[:,i]
    Vj = V[:,j]
    Vk = V[:,k]
    λi = λ[i]
    λj = λ[j]
    λk = λ[k]
    Mki = λk * Vi[a] * outer(Vk,Vk) + λi * Vk[a] * outer(Vk,Vi)
    Mij = λi * Vj[a] * outer(Vj,Vi) + λj * Vi[a] * outer(Vj,Vj)
    return 1/(λk-λi) * Mki - 1/(λi-λj) * Mij

def rhs2(λ,V):
    ret = zeros(9)
    for i in range(3):
        for a in range(3):
            ret[3*i:3*(i+1)] += dot(mat(λ,V,i,a), ε[:,a])
    return ret


def ode_stepper2(A0, dt):
    t = array([0, dt])
    G = dot(A0.T,A0)
    l,V = eig(G)
    y0 = zeros(12)
    y0[:3] = l
    y0[3:] = V.ravel(order='F')
    ret = odeint(f2, y0, t)[1]
    return ret[:3], ret[3:].reshape([3,3], order='F')

def test2(start=100):
    A, U, Σ, V = test()
    A0 = A[start]
    m = n - start
    l = zeros([m,3])
    V = zeros([m,3,3])
    for i in range(m):
        dt = i*5e-9/n * tScale
        l[i], V[i] = ode_stepper2(A0,dt)
    return l, V

def test3():
    l = zeros([n,3])
    V = zeros([n,3,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        l[i],V[i] = ode_stepper2(eye(3),dt)
    return l, V

def plot_eigenvec(a, Vold, Vnew):
    for i in range(3):
        if i==0:
            plot(x,Vold[:,i,a],label='Original ODEs',color=cm[0])
            plot(x,Vnew[:,i,a],label='Eigendecomposition ODEs',marker='x',color=cm[2],linestyle='None')
        else:
            plot(x,Vold[:,i,a],color=cm[0])
            plot(x,Vnew[:,i,a],marker='x',color=cm[2],linestyle='None')

def rotmat(x,y,z):
    Rx = array([[1,0,0],[0,cos(x),-sin(x)],[0,sin(x),cos(x)]])
    Ry = array([[cos(y),0,sin(y)],[0,1,0],[-sin(y),0,cos(y)]])
    Rz = array([[cos(z),-sin(z),0],[sin(z),cos(z),0],[0,0,1]])
    return dot(Rz,dot(Ry,Rx))

def sec(x):
    return 1/lim(cos(x))

def Minv(x,y,z,λ1,λ2,λ3):
    return array([[cos(x)*tan(y)/lim(λ1-λ2), -sin(x)*tan(y)/lim(λ1-λ3), 1/lim(λ2-λ3)],
                  [-sin(x)/lim(λ1-λ2),       -cos(x)/lim(λ1-λ3),        0],
                  [cos(x)*sec(y)/lim(λ1-λ2), -sec(y)*sin(x)/lim(λ1-λ3), 0]])

def b(x,y,z,λ1,λ2,λ3):
    R = rotmat(x,y,z)
    Λ = diag([λ1,λ2,λ3])
    RHS = -(dot(Λ,dot(R.T,dot(ε,R))) + dot(dot(R.T,dot(ε.T,R)),Λ))
    return array([RHS[0,1], RHS[0,2], RHS[1,2]])

def f_angles(y0, t):
    λ1 = y0[0]
    λ2 = y0[1]
    λ3 = y0[2]
    x = y0[3]
    y = y0[4]
    z = y0[5]

    V = rotmat(x,y,z)
    v1 = V[:,0]
    v2 = V[:,1]
    v3 = V[:,2]

    ret = zeros(6)
    ret[0] = -2 * dot(v1,dot(ε,v1))
    ret[1] = -2 * dot(v2,dot(ε,v2))
    ret[2] = -2 * dot(v3,dot(ε,v3))
    ret[3:] = dot(Minv(x,y,z,λ1,λ2,λ3), b(x,y,z,λ1,λ2,λ3))

    if includeSources:
        Λ = diag(y0[:3])
        ret[:3] -= 2/τ * diag(dot(Λ, dev(Λ)))

    return ret

def ode_stepper_angles(dt):
    t = array([0, dt])
    y0 = zeros(6)
    y0[:3] = ones(3)
    y0[3:] = zeros(3)
    ret = odeint(f_angles, y0, t, atol=1e-6,rtol=1e-6)[1]
    return ret[:3], ret[3:]

def test_angles():
    l = zeros([n,3])
    θ = zeros([n,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        l[i],θ[i] = ode_stepper_angles(dt)
    return l, θ

def angles2vecs(θ):
    return array([rotmat(θ[i,0],θ[i,1],θ[i,2]) for i in range(len(θ))])
