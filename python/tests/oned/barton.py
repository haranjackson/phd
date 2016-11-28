from numpy import array, dot, exp, zeros, trace as tr, transpose as trans
from numpy.linalg import det, inv

from options import nx, ny, nz

TEST = 1

ρ0 = 8.93e3
c0 = 4.6e3
cv = 3.9e2
T0 = 300
b0 = 2.1e3
α = 1
β = 3
γ = 2

K0 = c0**2 - 4/3 * b0**2
B0 = b0**2


def initial_vector(u, F, S):
    U = zeros(18)
    v = array(u)
    A = inv(array(F))
    ρ = density(A)
    U[0] = ρ
    U[1] = ρ * total_energy(A, S, v)
    U[2:5] = ρ * v
    U[5:14] = A.ravel()
    return U

def total_energy(A, S, v):
    """ Returns the total energy, given the distortion tensor and entropy
    """
    G = dot(trans(A), A)
    return U_1(G) + U_2(G, S) + W(G) + dot(v,v) / 2

def density(A):
    """ Returns the density, given the distortion tensor
    """
    return ρ0 * det(A)

def I_1(G):
    """ Returns the first invariant of G
    """
    return tr(G)

def I_2(G):
    """ Returns the second invariant of G
    """
    return 1/2 * (tr(G)**2 - tr(dot(G,G)))

def I_3(G):
    """ Returns the third invariant of G
    """
    return det(G)

def U_1(G):
    """ Returns the first component of the thermal energy density, given G
    """
    return K0/(2*α**2) * (I_3(G)**(α/2) - 1)**2

def U_2(G, S):
    """ Returns the second component of the thermal energy density, given G and entropy
    """
    return cv * T0 * I_3(G)**(γ/2) * (exp(S/cv) - 1)

def W(G):
    """ Returns the internal energy due to shear deformations, given the distortion tensor
    """
    return B0/2 * I_3(G)**(β/2) * (I_1(G)**2 / 3 - I_2(G))

def barton_IC():
    if TEST==1:
        UL0 = initial_vector([0, 0.5e3, 1e3], [[0.98,0,0],[0.02,1,0.1],[0,0,1]], 1e3)
        UR0 = initial_vector([0, 0,     0  ], [[1,   0,0],[0,   1,0.1],[0,0,1]], 0)
    elif TEST==2:
        UL0 = initial_vector([2e3, 0, 100], [[1,0,0],[-0.01,0.95,0.02],[-0.015,0,0.9]], 0)
        UR0 = initial_vector([0, -30, -10], [[1,0,0],[0.015,0.95,0],[-0.01,0,0.9]], 0)
    ret = zeros([nx, ny, nz, 18])
    for i in range(nx):
        if i < int(nx/2):
            ret[i,0,0] = UL0
        else:
            ret[i,0,0] = UR0
    return ret
