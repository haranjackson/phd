from itertools import product

from numpy import concatenate, diag, eye, ones, zeros

from solvers.basis import quad, basis_polys, derivative_values
from system.gpr.misc.functions import kron_prod
from options import ndim, N1, NT, nV


def inner_products(N1, nodes, weights, ψ, ψDer):
    """ Returns the elements of the matrices used in the Galerkin predictor
    """
    I11 = zeros([N1, N1])       # I11[a,b] = ψ_a(1) * ψ_b(1)
    I1 = zeros([N1, N1])        # I1[a,b] = ψ_a • ψ_b
    I2 = zeros([N1, N1])        # I2[a,b] = ψ_a • ψ_b'

    for a, b in product(range(N1), range(N1)):

        I11[a,b] = ψ[a](1) * ψ[b](1)

        if a==b:
            I1[a,b] = weights[a]
            I2[a,b] = (ψ[a](1)**2 - ψ[a](0)**2) / 2
        else:
            I2[a,b] = weights[a] * ψDer[1][b](nodes[a])

    return I11, I1, I2, eye(N1)

def system_matrices():
    """ Returns the matrices used in the Galerkin predictor
    """
    nodes, _, weights = quad()
    ψ, ψDer, _        = basis_polys()
    DERIVS            = derivative_values()

    I11, I1, I2, I = inner_products(N1, nodes, weights, ψ, ψDer)

    W = concatenate([ψ[a](0) * kron_prod([I1]*ndim) for a in range(N1)])

    U = kron_prod([I11-I2.T] + [I1]*ndim)

    V = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        V[i-1] = kron_prod([I1]*i + [I2] + [I1]*(ndim-i))

    Z = kron_prod([I1]*(ndim+1))
    Z = (diag(Z) * ones([nV, NT])).T

    T = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        T[i-1] = kron_prod([I]*i + [DERIVS] + [I]*(ndim-i))

    return W, U, V, Z, T
