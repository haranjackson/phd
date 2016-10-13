import numpy as np
import sympy as sp
from matplotlib.pyplot import spy
from numpy import array, einsum, float64
from numpy.linalg import det, inv
from numpy.random import rand
from sympy import Matrix, symbols


A11, A21, A31, A12, A22, A32, A13, A23, A33 = symbols('A11 A21 A31 A12 A22 A32 A13 A23 A33')
J1, J2, J3 = symbols('J1 J2 J3')
Asym = Matrix([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])


def A_num():
    ret = rand(3,3)
    if det(ret) > 0:
        return ret
    else:
        return -ret

def L2_2D(X):
    return ( X[0,0]**2 + X[0,1]**2 + X[0,2]**2
           + X[1,0]**2 + X[1,1]**2 + X[1,2]**2
           + X[2,0]**2 + X[2,1]**2 + X[2,2]**2)

def det3(X):
    return (X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2])
           -X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2])
           +X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2]))

def tr(X):
    return X[0,0] + X[1,1] + X[2,2]

def gram_num(A):
    return np.dot(A.T, A)

def gram_rev_num(A):
    return np.dot(A, A.T)

def gram_sym(A):
    return A.T * A

def dev_num(G):
    return G - tr(G)/3 * np.eye(3)

def dev_sym(G):
    return G - tr(G)/3 * sp.eye(3)

def S_num(A, t1):
    G = gram_num(A)
    return -3/t1 * det3(A)**(5/3) * np.dot(A, dev_num(G))

def S_sym(A, t1):
    G = gram_sym(A)
    return -3/t1 * det3(A)**(5/3) * (A * dev_sym(G))

def J_num(A, t1):
    G = gram_num(A)
    Grev = gram_rev_num(A)
    AdevG = np.dot(A, dev_num(G))
    AinvT = inv(A).T
    AA = einsum('ij,mn', A, A)
    L2A = L2_2D(A)

    ret = 5/3 * einsum('ij,mn', AdevG, AinvT) - 2/3 * AA + AA.swapaxes(1,3)

    for i in range(3):
        for j in range(3):
            ret[i,j,i,j] -= L2A / 3

    for k in range(3):
        ret[k,:,k,:] += G
        ret[:,k,:,k] += Grev

    ret *= -3/t1 * det3(A)**(5/3)
    return ret.reshape([9,9])

def J_sym(A, t1):
    Svec = S_sym(A, t1).reshape(9,1)
    Avec = A.reshape(9,1)
    return Svec.jacobian(Avec)

def evaluate(M):
    global Anum
    A = Anum
    ret = M.subs([(A11,A[0,0]), (A12,A[0,1]), (A13,A[0,2]),
                  (A21,A[1,0]), (A22,A[1,1]), (A23,A[1,2]),
                  (A31,A[2,0]), (A32,A[2,1]), (A33,A[2,2])])
    return array(ret.tolist()).astype(float64)

if __name__ == "__main__":
    Anum = A_num()
    Jnum = J_num(Anum,1)
    Jsym = J_sym(Asym,1)
    Jnum2 = evaluate(Jsym)
    diff = Jnum-Jnum2
    spy(abs(diff)>1e-14)

    G = gram_num(Anum)
    err = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            err[i,j] = diff[3*i,3*j]
