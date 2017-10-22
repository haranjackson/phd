from numpy import array, cos, diag, eye, mod, outer, sin, tan
from scipy.integrate import odeint
from scipy.linalg import svd
import matplotlib.pyplot as plt

from auxiliary.funcs import dev
from gpr.variables.eos import E_A


### Options ###

RAND = True

if RAND:
    ε = rand(3,3)
    ε -= 0.5
    ε *= 2
    Λ = rand(3)
else:
    ε = array([[0.62,  0.40,  1.14],
               [-0.28, -1.41, 0.59],
               [-0.19, -0.72, -1.28]])
    Λ = ones(3)


τ = 1.45e-9

tScale = 1
includeSources = 1
n = 50


### Auxiliary Functions ###

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

def sec(x):
    return 1/lim(cos(x))

def plot_eigenvec(a, Vold, Vnew):
    for i in range(3):
        if i==0:
            plot(x,Vold[:,i,a], label='Original ODEs', color=cm[0])
            plot(x,Vnew[:,i,a], label='Eigendecomp ODEs', color=cm[2],
                 marker='x', linestyle='None')
        else:
            plot(x,Vold[:,i,a], color=cm[0])
            plot(x,Vnew[:,i,a], color=cm[2], marker='x', linestyle='None')

### Standard Formulation ###

def f_standard(y, t):
    A = y.reshape([3,3])
    ret = -dot(A,ε)
    if includeSources:
        ret -= E_A(A,1)/τ
    return ret.ravel()

def solver_standard(A, dt):
    t = array([0, dt])
    y0 = A.ravel()
    return odeint(f_standard, y0, t, rtol=1e-12, atol=1e-12)[1].reshape([3,3])


### Vectors Formulaiton ###

def f_vectors(y, t):
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

def solver_vectors(A0, dt):
    t = array([0, dt])
    G = dot(A0.T,A0)
    l,V = eig(G)
    y0 = zeros(12)
    y0[:3] = l
    y0[3:] = V.ravel(order='F')
    ret = odeint(f_vectors, y0, t)[1]
    return ret[:3], ret[3:].reshape([3,3], order='F')


### Angles Formulation ###

def rotmat_angles(x,y,z):
    Rx = array([[1,0,0],[0,cos(x),-sin(x)],[0,sin(x),cos(x)]])
    Ry = array([[cos(y),0,sin(y)],[0,1,0],[-sin(y),0,cos(y)]])
    Rz = array([[cos(z),-sin(z),0],[sin(z),cos(z),0],[0,0,1]])
    return dot(Rz,dot(Ry,Rx))

def Minv(x,y,z,λ1,λ2,λ3):
    return array([[cos(x)*tan(y)/lim(λ1-λ2), -sin(x)*tan(y)/lim(λ1-λ3), 1/lim(λ2-λ3)],
                  [-sin(x)/lim(λ1-λ2),       -cos(x)/lim(λ1-λ3),        0],
                  [cos(x)*sec(y)/lim(λ1-λ2), -sec(y)*sin(x)/lim(λ1-λ3), 0]])

def b_angles(x,y,z,λ1,λ2,λ3):
    R = rotmat_angles(x,y,z)
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

    V = rotmat_angles(x,y,z)
    v1 = V[:,0]
    v2 = V[:,1]
    v3 = V[:,2]

    ret = zeros(6)
    ret[0] = -2 * λ1 * dot(v1,dot(ε,v1))
    ret[1] = -2 * λ2 * dot(v2,dot(ε,v2))
    ret[2] = -2 * λ3 * dot(v3,dot(ε,v3))
    ret[3:] = dot(Minv(x,y,z,λ1,λ2,λ3), b_angles(x,y,z,λ1,λ2,λ3))

    if includeSources:
        Λ = diag(y0[:3])
        ret[:3] -= 2/τ * diag(dot(Λ, dev(Λ)))

    return ret

def solver_angles(Λ, dt):
    t = array([0, dt])
    y0 = zeros(6)
    y0[:3] = Λ
    y0[3:] = zeros(3)
    ret = odeint(f_angles, y0, t, atol=1e-6, rtol=1e-6)[1]
    return ret[:3], ret[3:]


### Quaternions Formulation ###

def rotmat_quaternions(x, y, z, s_sign):
    ret = zeros([3,3])
    x2 = x*x
    y2 = y*y
    z2 = z*z
    s = s_sign * sqrt(1-x2-y2-z2)
    xy = x*y
    yz = y*z
    zx = z*x
    sx = s*x
    sy = s*y
    sz = s*z
    ret[0,0] = 0.5 - y2 - z2
    ret[1,1] = 0.5 - z2 - x2
    ret[2,2] = 0.5 - x2 - y2
    ret[0,1] = xy - sz
    ret[1,0] = xy + sz
    ret[0,2] = zx + sy
    ret[2,0] = zx - sy
    ret[1,2] = yz - sx
    ret[2,1] = yz + sx
    ret *= 2
    return ret

def qaxis(x,y,z):
    """ Returns the axis of rotation corresponding to quaternion components
        x, y, z
    """
    mod = sqrt(x*x + y*y + z*z)
    return array([x,y,z]) / mod

def qangle(x,y,z):
    """ Returns the angle of rotation corresponding to quaternion components
        x, y, z
    """
    s = sqrt(1 - x*x + y*y + z*z)
    mod = sqrt(x*x + y*y + z*z)
    return 2 * arctan2(s, mod)

def b_quaternions(V, λ):
    Λ = diag(λ)
    RHS = dot(Λ, dot(V.T, dot(ε, V))) + dot( dot(V.T, dot(ε.T, V)), Λ)
    return array([RHS[1,2], RHS[0,2], RHS[0,1]])

def f_quaternions(y0, t):
    λ1 = y0[0]
    λ2 = y0[1]
    λ3 = y0[2]
    x = y0[3]
    y = y0[4]
    z = y0[5]

    S_SIGN = 1

    V = rotmat_quaternions(x, y, z, S_SIGN)
    v1 = V[:,0]
    v2 = V[:,1]
    v3 = V[:,2]
    s = - S_SIGN * sqrt(1 - x*x - y*y - z*z)

    M = -0.5 * array([[s, -z, y],
                      [z, s, -x],
                      [-y, x, s]])

    ret = zeros(6)
    ret[0] = -2 * λ1 * dot(v1,dot(ε,v1))
    ret[1] = -2 * λ2 * dot(v2,dot(ε,v2))
    ret[2] = -2 * λ3 * dot(v3,dot(ε,v3))
    ret[3:] = - b_quaternions(V, y0[:3])
    ret[3] /= lim(λ2 - λ3)
    ret[4] /= lim(λ3 - λ1)
    ret[5] /= lim(λ1 - λ2)
    ret[3:] = dot(M, ret[3:])

    if includeSources:
        Λ = diag(y0[:3])
        ret[:3] -= 2/τ * diag(dot(Λ, dev(Λ)))

    return ret

def solver_quaternions(Λ, dt):
    t = array([0, dt])
    y0 = zeros(6)
    y0[:3] = Λ
    y0[3:] = zeros(3)
    ret = odeint(f_quaternions, y0, t, atol=1e-6, rtol=1e-6)[1]
    return ret[:3], ret[3:]


### Tests ###

def test_standard(Λ):
    A0 = diag(sqrt(Λ))

    l = zeros([n,3])
    V = zeros([n,3,3])
    A = zeros([n,3,3])
    U = zeros([n,3,3])
    Σ = zeros([n,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        A[i] = solver_standard(A0, dt)
        U[i], Σ[i], V[i] = svd(A[i])
        V[i] = V[i].T
        l[i] = Σ[i]**2

    return l, V, A, U, Σ

def test_vectors(Λ, start=1):
    _, _, A, _, _ = test_standard(Λ)
    A0 = A[start]
    m = n - start
    l = zeros([m,3])
    V = zeros([m,3,3])
    for i in range(m):
        dt = i*5e-9/n * tScale
        l[i], V[i] = solver_vectors(A0, dt)
    return l, V

def test_angles(Λ):
    l = zeros([n,3])
    V = zeros([n,3,3])
    θ = zeros([n,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        l[i], θ[i] = solver_angles(Λ, dt)
        V[i] = rotmat_angles(θ[i,0],θ[i,1],θ[i,2])
    return l, V, θ

def test_quaternions(Λ):
    l = zeros([n,3])
    V = zeros([n,3,3])
    θ = zeros([n,3])
    for i in range(n):
        dt = i*5e-9/n * tScale
        l[i], θ[i] = solver_quaternions(Λ, dt)
        V[i] = rotmat_quaternions(θ[i,0], θ[i,1], θ[i,2], 1)
    return l, V, θ


### Main ###

if __name__ == "__main__":

    print('ε =', ε)
    print('Λ =', Λ)

    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    l_stan, V_stan, _, _ , _ = test_standard(Λ)
    ax1.plot(l_stan)
    ax1.set_title('Standard')

    l_vec, V_vec = test_vectors(Λ, 1)
    ax2.plot(l_vec)
    ax2.set_title('Vectors')

    l_ang, V_ang, _ = test_angles(Λ)
    ax3.plot(l_ang)
    ax3.set_title('Angles')

    l_quat, V_quat, _ = test_quaternions(Λ)
    ax4.plot(l_quat)
    ax4.set_title('Quaternions')

    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    V_stan_end = V_stan[:, :, argmin(l_stan[-1])]
    V_vec_end = V_vec[:, :, argmin(l_vec[-1])]
    V_ang_end = V_ang[:, :, argmin(l_ang[-1])]
    V_quat_end = V_quat[:, :, argmin(l_quat[-1])]

    V_stan_end *= sgn(V_stan_end[-1,0])
    V_vec_end *= sgn(V_vec_end[-1,0])
    V_ang_end *= sgn(V_ang_end[-1,0])
    V_quat_end *= sgn(V_quat_end[-1,0])

    ax1.plot(V_stan_end)
    ax1.set_title('Standard')
    ax2.plot(V_vec_end)
    ax2.set_title('Vectors')
    ax3.plot(V_ang_end)
    ax3.set_title('Angles')
    ax4.plot(V_quat_end)
    ax4.set_title('Quaternions')

    ax1.set_ylim([-1.1, 1.1])
    ax3.set_ylim([-1.1, 1.1])


"""

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

"""
