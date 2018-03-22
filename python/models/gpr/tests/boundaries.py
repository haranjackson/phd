from numpy import dot
from scipy.linalg import det, svd


def destress(Q, MP):
    """ Removes the stress and associated energy from state Q
    """
    A = Q[5:14].reshape([3,3])
    detA = det(A)
    U, _, Vh = svd(A)
    Q[5:14] = detA**(1 / 3) * dot(U, Vh).ravel()
