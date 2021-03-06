from numpy import array, dot, eye, sqrt


def lim(x):
    """ Enforces abs(x) > TOL
    """
    TOL = 1e-11
    if abs(x) < TOL:
        if x >= 0:
            return TOL
        else:
            return -TOL
    else:
        return x


""" MATRIX FUNCTIONS """


def tr(X):
    return X[0, 0] + X[1, 1] + X[2, 2]


def det3(X):
    return (X[0][0] * (X[1][1] * X[2][2] - X[2][1] * X[1][2])
            - X[1][0] * (X[0][1] * X[2][2] - X[2][1] * X[0][2])
            + X[2][0] * (X[0][1] * X[1][2] - X[1][1] * X[0][2]))


""" NORMS """


def L2_1D(x):
    return x[0]**2 + x[1]**2 + x[2]**2


def L2_2D(X):
    """ Returns sum(Xij^2)
    """
    return (X[0, 0]**2 + X[0, 1]**2 + X[0, 2]**2
            + X[1, 0]**2 + X[1, 1]**2 + X[1, 2]**2
            + X[2, 0]**2 + X[2, 1]**2 + X[2, 2]**2)


def sigma_norm(σ):
    """ Returns the norm defined in Boscheri et al
        Equal to sqrt(3/2) * |dev(σ)|
    """
    tmp1 = (σ[0, 0] - σ[1, 1])**2 + (σ[1, 1] -
                                     σ[2, 2])**2 + (σ[2, 2] - σ[0, 0])**2
    tmp2 = σ[0, 1]**2 + σ[1, 2]**2 + σ[2, 0]**2
    return sqrt(0.5 * tmp1 + 3 * tmp2)


""" MATRIX INVARIANTS """


def I_1(G):
    """ Returns the first invariant of G
    """
    return tr(G)


def I_2(G):
    """ Returns the second invariant of G
    """
    return 1 / 2 * (tr(G)**2 - L2_2D(G))


def I_3(G):
    """ Returns the third invariant of G
    """
    return det3(G)


""" DEVIATORS """


def dev(G):
    """ Returns the deviator of G
    """
    return G - tr(G) / 3 * eye(3)


def GdevG(G):
    return dot(G, dev(G))


def AdevG(A, G):
    return dot(A, dev(G))


""" MISC """


def gram(A):
    """ Returns the Gram matrix for A
    """
    return dot(A.T, A)


def gram_rev(A):
    """ Returns the Gram matrix for A^T
    """
    return dot(A, A.T)


def reorder(X):
    """ Reorders the columns of X from atypical ordering to typical ordering
    """
    if len(X) == 17:
        perm = array([0, 1, 11, 12, 13, 2, 5, 8, 3, 6, 9, 4, 7, 10, 14, 15, 16])
    else:
        perm = array([0, 1, 11, 12, 13, 2, 5, 8, 3, 6, 9, 4, 7, 10])

    return X[perm]
