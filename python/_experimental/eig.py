from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork


def eigvalsh3(M, overwriteM=0):
    """ Returns the eigenvalues for symmetric M
    """
    p1 = M[0, 1]**2 + M[0, 2]**2 + M[1, 2]**2
    q = tr(M) / 3
    p2 = (M[0, 0] - q)**2 + (M[1, 1] - q)**2 + (M[2, 2] - q)**2 + 2 * p1
    p = sqrt(p2 / 6)

    if overwriteM:
        M[0, 0] -= q
        M[1, 1] -= q
        M[2, 2] -= q
        r = det3(M) / (2 * p**3)
    else:
        B = (M - q * eye(3))
        r = det3(B) / (2 * p**3)

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
    λ3 = q + 2 * p * cos(φ + (2 * pi / 3))
    λ2 = 3 * q - λ1 - λ3           # since tr(M) = λ1 + λ2 + λ3
    return λ1, λ2, λ3


def eigh3(M, V=None, overwriteM=0):
    λ1, λ2, λ3 = eigvalsh3(M, overwriteM=overwriteM)
    if V is None:
        V = zeros([3, 3])
    M00 = M[0, 0]
    M01 = M[0, 1]
    M02 = M[0, 2]
    M11 = M[1, 1]
    M22 = M[2, 2]

    M01_M12 = M01 * M[1, 2]
    M02_M10 = M02 * M[0, 1]
    M01_M10 = M01**2

    a = M00 - λ1
    b = M11 - λ1
    V[0, 0] = M01_M12 - M02 * b
    V[0, 1] = M02_M10 - a * M22
    V[0, 2] = a * b - M01_M10
    a = M00 - λ2
    b = M11 - λ2
    V[1, 0] = M01_M12 - M02 * b
    V[1, 1] = M02_M10 - a * M22
    V[1, 2] = a * b - M01_M10
    a = M00 - λ3
    b = M11 - λ3
    V[2, 0] = M01_M12 - M02 * b
    V[2, 1] = M02_M10 - a * M22
    V[2, 2] = a * b - M01_M10


def eigh3_1(M, x, y, z, overwriteM=0):
    """ Returns an upper bound on the maximum eigenvalue of the following matrix:
        [M11+x  M12  M13  y]
        [M21    M22  M23  0]
        [M31    M32  M33  0]
        [y      0    0    z]
        where X=(x y z).
    """
    return None