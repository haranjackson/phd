from numba import jit
from numpy import dot, empty, eye, kron


@jit
def tr(X):
    return X[0,0] + X[1,1] + X[2,2]

@jit
def det3(X):
    return (X[0][0] * (X[1][1] * X[2][2] - X[2][1] * X[1][2])
           -X[1][0] * (X[0][1] * X[2][2] - X[2][1] * X[0][2])
           +X[2][0] * (X[0][1] * X[1][2] - X[1][1] * X[0][2]))

@jit
def dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@jit
def L2_1D(x):
    return x[0]**2 + x[1]**2 + x[2]**2

@jit
def L2_2D(X):
    """ Returns sum(Xij^2)
    """
    return ( X[0,0]**2 + X[0,1]**2 + X[0,2]**2
           + X[1,0]**2 + X[1,1]**2 + X[1,2]**2
           + X[2,0]**2 + X[2,1]**2 + X[2,2]**2)

@jit
def dev(G):
    """ Returns the deviator of G
    """
    return G - tr(G)/3 * eye(3)

@jit
def GdevG(G):
    return dot(G,G) - tr(G)/3 * G

@jit
def AdevG(A,G):
    return dot(A,G) - tr(G)/3 * A

@jit
def gram(A):
    """ Returns the Gram matrix for A
    """
    return dot(A.T, A)

@jit
def outer3self(x):
    """ Returns the outer product of x with itself
    """
    ret = empty([3,3])
    for i in range(3):
        for j in range(3):
            ret[i,j] = x[i]*x[j]
    return ret

def kron_prod(matList):
    """ Returns the kronecker product of the matrices in matList
    """
    ret = matList[0]
    for i in range(1, len(matList)):
        ret = kron(ret, matList[i])
    return ret
