from numba import jit
from numpy import array, cross, dot, empty, eye, kron, zeros
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork


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
def inv3(X):
    detX = det3(X)
    ret = zeros([3,3])
    ret[0, 0] = (X[1, 1] * X[2, 2] - X[2, 1] * X[1, 2])
    ret[0, 1] = (X[0, 2] * X[2, 1] - X[0, 1] * X[2, 2])
    ret[0, 2] = (X[0, 1] * X[1, 2] - X[0, 2] * X[1, 1])
    ret[1, 0] = (X[1, 2] * X[2, 0] - X[1, 0] * X[2, 2])
    ret[1, 1] = (X[0, 0] * X[2, 2] - X[0, 2] * X[2, 0])
    ret[1, 2] = (X[1, 0] * X[0, 2] - X[0, 0] * X[1, 2])
    ret[2, 0] = (X[1, 0] * X[2, 1] - X[2, 0] * X[1, 1])
    ret[2, 1] = (X[2, 0] * X[0, 1] - X[0, 0] * X[2, 1])
    ret[2, 2] = (X[0, 0] * X[1, 1] - X[1, 0] * X[0, 1])
    return ret / detX

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
def gram_rev(A):
    """ Returns the Gram matrix for A^T
    """
    return dot(A, A.T)

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

def reorder(X, order='typical'):
    """ Reorders the columns of X
    """
    if order=='typical':
        perm = array([0,1,11,12,13,2,5,8,3,6,9,4,7,10,14,15,16])
    elif order=='atypical':
        perm = array([0,1,5,8,11,6,9,12,7,10,13,2,3,4,14,15,16])

    return X[perm]

@jit
def eigvalsh3(M, overwriteM=0):
    """ Returns the eigenvalues for symmetric M
    """
    p1 = M[0,1]**2 + M[0,2]**2 + M[1,2]**2
    q = tr(M)/3
    p2 = (M[0,0]-q)**2 + (M[1,1]-q)**2 + (M[2,2]-q)**2 + 2*p1
    p = sqrt(p2/6)

    if overwriteM:
        M[0,0] -= q
        M[1,1] -= q
        M[2,2] -= q
        r = det3(M) / (2*p**3)
    else:
        B = (M - q*eye(3))
        r = det3(B) / (2*p**3)

    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    if r <= -1:
       φ = pi / 3
    elif r >= 1:
       φ = 0
    else:
       φ = arccos(r) / 3

    # the eigenvalues satisfy λ3 <= λ2 <= λ1
    λ1 = q + 2 * p * cos(φ)
    λ3 = q + 2 * p * cos(φ + (2*pi/3))
    λ2 = 3 * q - λ1 - λ3           # since trace(M) = λ1 + λ2 + λ3
    return λ1, λ2, λ3

def eigh3(M, V=None, overwriteM=0):
    λ1, λ2, λ3 = eigvalsh3(M, overwriteM=overwriteM)
    if V is None:
        V = zeros([3,3])
    M00 = M[0,0]
    M01 = M[0,1]
    M02 = M[0,2]
    M11 = M[1,1]
    M22 = M[2,2]

    M01_M12 = M01*M[1,2]
    M02_M10 = M02*M[0,1]
    M01_M10 = M01**2

    a = M00-λ1
    b = M11-λ1
    V[0,0] = M01_M12 - M02*b
    V[0,1] = M02_M10 - a*M22
    V[0,2] = a*b - M01_M10
    a = M00-λ2
    b = M11-λ2
    V[1,0] = M01_M12 - M02*b
    V[1,1] = M02_M10 - a*M22
    V[1,2] = a*b - M01_M10
    a = M00-λ3
    b = M11-λ3
    V[2,0] = M01_M12 - M02*b
    V[2,1] = M02_M10 - a*M22
    V[2,2] = a*b - M01_M10

def eigh3_1(M, x, y, z, overwriteM=0):
    """ Returns an upper bound on the maximum eigenvalue of the following matrix:
        [M11+x  M12  M13  y]
        [M21    M22  M23  0]
        [M31    M32  M33  0]
        [y      0    0    z]
        where X=(x y z).
    """
    return None

def eigvalsn(a, n):
    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a,))
    lwork = _compute_lwork(geev_lwork, n, compute_vl=0, compute_vr=0)
    w, _, _, _, _ = geev(a, lwork=lwork, compute_vl=0, compute_vr=0, overwrite_a=1)
    return w