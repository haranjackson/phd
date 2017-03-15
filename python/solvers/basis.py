from numpy import array, concatenate, eye, zeros
from numpy import polyder, polyint
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange

from options import N1


def quad():
    """ Returns Legendre-Gauss nodes and weights, scaled to [0,1]
    """
    nodes, weights = leggauss(N1)
    nodes += 1
    nodes /= 2
    weights /= 2
    gaps = nodes - concatenate(([0], nodes[:-1]))
    return nodes, gaps, weights

def basis_polys():
    """ Returns basis polynomials and their derivatives and antiderivatives
    """
    nodes, _, _ = quad()
    psi = [lagrange(nodes,eye(N1)[i]) for i in range(N1)]
    psiDer = [[polyder(psip, m=a) for psip in psi] for a in range(N1+1)]
    psiInt = [polyint(psip) for psip in psi]
    return psi, psiDer, psiInt

def end_values():
    """ Returns the values of th basis functions at 0,1
    """
    psi, _, _ = basis_polys()
    ret = zeros([N1, 2])
    for i in range(N1):
        ret[i,0] = psi[i](0)
        ret[i,1] = psi[i](1)
    return ret

def derivative_values():
    """ Returns the value of the derivative of the jth basis function at the ith node
    """
    nodes, _, _ = quad()
    _, psiDer, _ = basis_polys()
    ret = zeros([N1, N1])
    for i in range(N1):
        for j in range(N1):
            ret[i,j] = psiDer[1][j](nodes[i])
    return ret

def derivative_end_values():
    """ Returns the value of the derivative of the ith basis function at 0 and 1
    """
    _, psiDer, _ = basis_polys()
    ret = zeros([N1, 2])
    for i in range(N1):
        ret[i,0] = psiDer[1][i](0)
        ret[i,1] = psiDer[1][i](1)
    return ret

def mid_values():
    psi, _, _ = basis_polys()
    return array([psii(0.5) for psii in psi])
