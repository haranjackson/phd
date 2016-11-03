from itertools import product

from numba import jit
from numpy import array, concatenate, diag, eye, ones, sqrt, zeros
from scipy.sparse import csc_matrix

from ader.basis import quad, basis_polys, derivative_values
from auxiliary.funcs import kron_prod
from options import ndim, N1, NT


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
    derivs            = derivative_values()

    I11, I1, I2, I = inner_products(N1, nodes, weights, ψ, ψDer)

    W = concatenate([ψ[a](0) * kron_prod([I1]*ndim) for a in range(N1)])

    U = kron_prod([I11-I2.T] + [I1]*ndim)
    U = csc_matrix(U)

    V = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        V[i-1] = kron_prod([I1]*i + [I2] + [I1]*(ndim-i))

    Z = kron_prod([I1]*(ndim+1))
    Z = (diag(Z) * ones([18, NT])).T

    T = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        T[i-1] = kron_prod([I]*i + [derivs] + [I]*(ndim-i))

    return W, U, V, Z, T

def Uinv1():
    """ Returns the U^-1 for N=1
    """
    X = 1/3 * array([[2,1-sqrt(3)],[1+sqrt(3),2]])
    return kron_prod([X] + [2*eye(2)]*ndim)

c1 = 2*(1-sqrt(3))/3
c2 = 2*(1+sqrt(3))/3

@jit
def UinvDot1(x):
    """ Returns U^-1.x for N=1 and ndim=1
    """
    ret = 4*x/3
    ret[0] += c1 * x[2]
    ret[1] += c1 * x[3]
    ret[2] += c2 * x[0]
    ret[3] += c2 * x[1]
    return ret
