#include "../../etc/types.h"
#include "../objects/gpr_objects.h"


double tr(Matr X)
{
    return X.trace();
}

double det(Matr X)
{
    return X.determinant();
}

Mat2_2 inv2(Mat2_2r X)
{
    double detX = det(X);
    Mat2_2 ret;
    ret(0, 0) = X(1, 1);
    ret(0, 1) = - X(0, 1);
    ret(1, 0) = - X(1, 0);
    ret(1, 1) = X(0, 0);
    return ret / detX;
}

Mat3_3 inv3(Mat3_3r X)
{
    double detX = det(X);
    Mat3_3 ret;
    ret(0, 0) = (X(1, 1) * X(2, 2) - X(2, 1) * X(1, 2));
    ret(0, 1) = (X(0, 2) * X(2, 1) - X(0, 1) * X(2, 2));
    ret(0, 2) = (X(0, 1) * X(1, 2) - X(0, 2) * X(1, 1));
    ret(1, 0) = (X(1, 2) * X(2, 0) - X(1, 0) * X(2, 2));
    ret(1, 1) = (X(0, 0) * X(2, 2) - X(0, 2) * X(2, 0));
    ret(1, 2) = (X(1, 0) * X(0, 2) - X(0, 0) * X(1, 2));
    ret(2, 0) = (X(1, 0) * X(2, 1) - X(2, 0) * X(1, 1));
    ret(2, 1) = (X(2, 0) * X(0, 1) - X(0, 0) * X(2, 1));
    ret(2, 2) = (X(0, 0) * X(1, 1) - X(1, 0) * X(0, 1));
    return ret / detX;
}

double L2_1D(Vec3r x)
{
    return x.squaredNorm();
}

double L2_2D(Mat3_3r X)
{
    return X.squaredNorm();
}

double dot(Vec3r u, Vec3r v)
{
    return u.dot(v);
}

Mat3_3 gram(Mat3_3r A)
{   // Returns the Gram matrix for A
    return A.transpose() * A;
}

Mat3_3 devG(Mat3_3r A)
{   // Returns the deviator of G (the Gramian of A)

    Mat3_3 ret = gram(A);
    double x = tr(ret)/3;
    ret(0,0) -= x;
    ret(1,1) -= x;
    ret(2,2) -= x;
    return ret;
}

Mat3_3 AdevG(Mat3_3r A)
{
    Mat3_3 G = gram(A);
    double x = tr(G)/3;
    G(0,0) -= x;
    G(1,1) -= x;
    G(2,2) -= x;
    return A*G;
}

Mat3_3 GdevG(Mat3_3r G)
{
    return G*G - tr(G)/3 * G;
}

Mat3_3 outer(Vec3r x, Vec3r y)
{
    Mat3_3 ret;
    ret.noalias() = x * y.transpose();
    return ret;
}


/*


def gram_rev(A):
    """ Returns the Gram matrix for A^T
    """
    return dot(A, A.T)

def outer3self(x):
    """ Returns the outer product of x with itself
    """
    ret = empty([3,3])
    for i in range(3):
        for j in range(3):
            ret[i,j] = x[i]*x[j]
    return ret

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

*/
