from numpy import dot, prod
from scipy.linalg import det, svd

from ader.etc.boundaries import extend_grid

from gpr.misc.structures import Cvec, State


def destress(A):
    """ Removes the stress and associated energy from state Q
    """
    detA = det(A)
    U, _, Vh = svd(A)
    return detA**(1 / 3) * dot(U, Vh)


def free_traction_ghost_state(Q, MP):
    P = State(Q, MP)
    Ai = P.A
    pi = P.p()

    ρg = P.ρ
    pg = pi
    # pg = -pi
    vg = -P.v
    # A_ = destress(Ai)
    # Ag = 2 * A_ - Ai
    Ag = Ai

    return Cvec(ρg, pg, vg, MP, Ag)


def wall_BC(u, N, NDIM, wall, MP):
    """ Extends the grid u in all dimensions. If wall[d]=1 then the
        boundaries in dimension d are no-slip, else they are transmissive.
    """
    ret = u.copy()

    for d in range(NDIM):

        ret = extend_grid(ret, N, d, 0)

        if wall[d]:
            shape = ret.shape
            n1 = int(prod(shape[:d]))
            n2 = shape[d]
            n3 = int(prod(shape[d + 1: NDIM]))
            retr = ret.reshape(n1, n2, n3, -1)
            for i in range(n1):
                for j in range(n3):
                    for k in range(N):
                        QL = retr[i, N+k, j]
                        QR = retr[i, -(N+k+1), j]
                        QLg = free_traction_ghost_state(QL, MP)
                        QRg = free_traction_ghost_state(QR, MP)
                        retr[i, N-1-k, j] = QLg
                        retr[i, -(N-k), j] = QRg
    return ret


def renormalize_A(u, MP):

    ncell = prod(u.shape[:-1])
    for i in range(ncell):
        Q = u.reshape(ncell, -1)[i]
        ρ = Q[0]
        A = Q[5:14].reshape([3,3])
        factor = ρ / (MP.ρ0 * det(A))
        A *= factor**(1/3)
