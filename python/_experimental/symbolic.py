import numpy as np
from numpy import array, dot, float64
from numpy.linalg import det, inv
from numpy.random import rand
from sympy import eye, Matrix, symbols

from auxiliary.common import material_parameters
from gpr.variables.vectors import Cvec
from options import reactiveEOS, Qc


defaultParams = material_parameters()
r0 = rand()
p0 = rand()
v0 = 2*rand(3) - 1
A0 = np.eye(3) + rand(3,3)
A0 *= (r0/det(A0))**(1/3)
J0 = rand(3)
c0 = rand()
Q0 = Cvec(r0, p0, v0, A0, J0, c0, defaultParams.y, defaultParams.pINF)

r, p, c, T = symbols('r p c T')
v1, v2, v3 = symbols('v1 v2 v3')
A11, A21, A31, A12, A22, A32, A13, A23, A33 = symbols('A11 A21 A31 A12 A22 A32 A13 A23 A33')
J1, J2, J3 = symbols('J1 J2 J3')

y, a = symbols('y a')

s1, s2, s3 = symbols('s1 s2 s3')
S11_11, S11_21, S11_31, S11_12, S11_22, S11_32, S11_13, S11_23, S11_33 = symbols('S11_11 S11_21 S11_31 S11_12 S11_22 S11_32 S11_13 S11_23 S11_33')
S21_11, S21_21, S21_31, S21_12, S21_22, S21_32, S21_13, S21_23, S21_33 = symbols('S21_11 S21_21 S21_31 S21_12 S21_22 S21_32 S21_13 S21_23 S21_33')
S31_11, S31_21, S31_31, S31_12, S31_22, S31_32, S31_13, S31_23, S31_33 = symbols('S31_11 S31_21 S31_31 S31_12 S31_22 S31_32 S31_13 S31_23 S31_33')

A = Matrix([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
v = Matrix([v1, v2, v3])
J = Matrix([J1, J2, J3])

def tr(X):
    return X[0,0] + X[1,1] + X[2,2]

def gram(A):
    return A.T * A

def dev(G):
    return G - tr(G)/3 * eye(3)

def E_1(r, p):
    return p / ((y-1) * r)

def E_1r(c):
    return Qc * c + minE

def E_2(A, J):
    G = gram(A)
    devG = dev(G)
    ret = cs**2/4 * sum(devG.multiply_elementwise(devG))
    ret += alpha**2/2 * sum(J.multiply_elementwise(J))
    return ret

def E_3(v):
    return 1/2 * sum(v.multiply_elementwise(v))

def E(r, p, c, A, J, v):
    return E_1(r, p) + E_2(A, J) + E_3(v) + int(reactiveEOS) * E_1r(c)

def pressure(E1, r):
    return E1 * (y-1) * r

def temperature(r,p):
    return p / ((y-1) * r * cv)

def sigma(r, A):
    G = gram(A)
    return -r * cs**2 * G * dev(G)

def sigma_A(r, A, i, j):
    G = gram(A)
    AdevG = A * dev(G)
    ret = -2/3 * G[i,j] * A
    for m in range(3):
        for n in range(3):
            ret[m, n] += A[m, i] * G[j, n] + A[m, j] * G[i, n]
            if n==i:
                ret[m,n] += AdevG[m,j]
            if n==j:
                ret[m,n] += AdevG[m,i]
    return -r * cs2 * ret

def heat_flux(r,p,J):
    return alpha**2 * temperature(r, p) * J

def flux_matrix():
    F = Matrix([r*v1,
                v1*r*E(r,p,c,A,J,v) + v1*p - (sigma(r,A).T*v)[0] + heat_flux(r,p,J)[0],
                r*v1*v1 + p - sigma(r,A)[0,0],
                r*v2*v1 - sigma(r,A)[1,0],
                r*v3*v1 - sigma(r,A)[2,0],
                (A*v)[0],
                (A*v)[1],
                (A*v)[2],
                0,
                0,
                0,
                0,
                0,
                0,
                r*J1*v1 + temperature(r,p),
                r*J2*v1,
                r*J3*v1,
                r*c*v1])
    return F

def block_matrix():
    B = Matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -v2, 0,   0,   -v3, 0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   -v2, 0,   0,   -v3, 0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   -v2, 0,   0,   -v3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, v1,  0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   v1,  0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   v1,  0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   v1,  0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   v1,  0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   v1,  0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,   0,   0,   0,   0, 0, 0, 0]])
    return B

def conservative_variables_matrix():
    Q = Matrix([r, r*E(r,p,c,A,J,v), r*v1, r*v2, r*v3, A11, A21, A31, A12, A22, A32, A13, A23, A33,
                r*J1, r*J2, r*J3, r*c])
    return Q

def primitive_variables_matrix():
    P = Matrix([r, p, v1, v2, v3, A11, A21, A31, A12, A22, A32, A13, A23, A33, J1, J2, J3, c])
    return P

def jacobian_primitive_matrix():
    J = Matrix([[v1,      0,       r,   0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,            0,  0],
                [0,       v1,      y*p, 0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      (y-1)*a**2*T, 0,  0],
                [s1,      1/r,     v1,  0,   0,   S11_11, S11_21, S11_31, S11_12, S11_22, S11_32, S11_13, S11_23, S11_33, 0,            0,  0],
                [s2,      0,       0,   v1,  0,   S21_11, S21_21, S21_31, S21_12, S21_22, S21_32, S21_13, S21_23, S21_33, 0,            0,  0],
                [s3,      0,       0,   0,   v1,  S31_11, S31_21, S31_31, S31_12, S31_22, S31_32, S31_13, S31_23, S31_33, 0,            0,  0],
                [0,       0,       A11, A12, A13, v1,     0,      0,      0,      0,      0,      0,      0,      0,      0,            0,  0],
                [0,       0,       A21, A22, A23, 0,      v1,     0,      0,      0,      0,      0,      0,      0,      0,            0,  0],
                [0,       0,       A31, A32, A33, 0,      0,      v1,     0,      0,      0,      0,      0,      0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      v1,     0,      0,      0,      0,      0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      v1,     0,      0,      0,      0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      v1,     0,      0,      0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      0,      v1,     0,      0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      0,      0,      v1,     0,      0,            0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      v1,     0,            0,  0],
                [-T/r**2, T/(r*p), 0,   0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      v1,           0,  0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,            v1, 0],
                [0,       0,       0,   0,   0,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,            0,  v1]])
    return J

def evaluate(M, Q0):
    r0 = Q0[0]
    E0 = Q0[1] / r0
    v0_1 = Q0[2] / r0
    v0_2 = Q0[3] / r0
    v0_3 = Q0[4] / r0
    A0_11 = Q0[5]
    A0_21 = Q0[6]
    A0_31 = Q0[7]
    A0_12 = Q0[8]
    A0_22 = Q0[9]
    A0_32 = Q0[10]
    A0_13 = Q0[11]
    A0_23 = Q0[12]
    A0_33 = Q0[13]
    J0_1 = Q0[14] / r0
    J0_2 = Q0[15] / r0
    J0_3 = Q0[16] / r0
    c0 = Q0[17] / r0
    A0 = Matrix([[A0_11,A0_12,A0_13],[A0_21,A0_22,A0_23],[A0_31,A0_32,A0_33]])
    v0 = Matrix([v0_1, v0_2, v0_3])
    J0 = Matrix([J0_1, J0_2, J0_3])
    E10 = E0 - E_2(A0,J0) - E_3(v0) - reactiveEOS * E_1r(c0)
    p0 = pressure(E10, r0)

    ret = M.subs([(r,r0), (p,p0), (v1,v0_1), (v2,v0_2), (v3,v0_3),
                  (A11,A0_11), (A12,A0_12), (A13,A0_13), (A21,A0_21), (A22,A0_22), (A23,A0_23),
                  (A31,A0_31), (A32,A0_32), (A33,A0_33), (J1,J0_1), (J2,J0_2), (J3,J0_3), (c,c0)])
    return array(ret.tolist()).astype(float64)


Q = conservative_variables_matrix()
P = primitive_variables_matrix()
F = flux_matrix()
B = block_matrix()
DQDP = Q.jacobian(P)
DFDP = F.jacobian(P)


def flux(Q0, d):
    return evaluate(F, Q0)

def jacobian(Q0, d):
    dPdQ = inv(evaluate(DQDP, Q0))
    dFdP = evaluate(DFDP, Q0)
    return dot(dFdP, dPdQ) + evaluate(B, Q0)
