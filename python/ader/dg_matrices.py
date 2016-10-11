from numpy import array, concatenate, diag, eye, ones, zeros
from scipy.sparse import csc_matrix

from ader.basis import quad, basis_polys, derivative_values
from auxiliary.funcs import kron_prod
from options import ndim, N1, NT


def inner_products(N1, nodes, weights, ψ, ψDer):
    """ Returns the elements of the matrices used in the Galerkin predictor
    """
    I11 = zeros([N1, N1])
    I1 = zeros([N1, N1])
    I2 = zeros([N1, N1])
    for a in range(N1):
        for b in range(N1):
            I11[a,b] = ψ[a](1) * ψ[b](1)
            if a==b:
                I1[a,b] = weights[a]
                I2[a,b] = (ψ[a](1)**2 - ψ[a](0)**2) / 2
            else:
                I2[a,b] = weights[a] * ψDer[1][b](nodes[a])
    I = eye(N1)
    return I11, I1, I2, I

def system_matrices():
    """ Returns the matrices used in the Galerkin predictor
    """
    nodes, _, weights = quad()
    ψ, ψDer, _        = basis_polys()
    derivs            = derivative_values()

    I11, I1, I2, I = inner_products(N1, nodes, weights, ψ, ψDer)

    T = zeros([ndim, NT, NT])
    for i in range(ndim):
        T[i] = kron_prod([I]*(i+1) + [derivs] + [I]*(ndim-1-i))

    W = concatenate([ψ[a](0) * kron_prod([I1]*ndim) for a in range(N1)])
    U = kron_prod([I11-I2.T] + [I1]*ndim)
    V = zeros([ndim, NT, NT])
    for i in range(ndim):
        V[i] = kron_prod([I1]*(i+1) + [I2] + [I1]*(ndim-1-i))
    Z = kron_prod([I1]*(ndim+1))

    U = csc_matrix(U)
    Z = (diag(Z) * ones([18, NT])).T

    return W, U, V, Z, T
