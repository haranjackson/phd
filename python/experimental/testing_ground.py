from numpy import dot, sign
from numpy.linalg import solve
from numpy.random import rand

from gpr.misc.functions import det3
from gpr.misc.structures import State, Cvec

from gpr.sys.conserved import M_cons
from gpr.sys.primitive import M_prim
from gpr.sys.jacobians import dQdP


def systems(Q, d, MP):
    P = State(Q, MP)
    DQDP = dQdP(P, MP)

    M1 = M_prim(Q, d, MP)
    M2 = M_cons(Q, d, MP)
    M3 = solve(DQDP, dot(M2, DQDP))
    return M1, M2, M3


def generate_vecs(MP):

    A = rand(3, 3)
    A /= sign(det3(A))
    ρ = det3(A) * MP.ρ0
    p = rand()
    v = rand(3)
    J = rand(3)

    Q = Cvec(ρ, p, v, MP, A, J)
    P = State(Q, MP)
    return Q, P


def cons_to_class(QL, QR, QL_, QR_, MPs):
    PL = State(QL, MPs[0])
    PR = State(QR, MPs[1])
    PL_ = State(QL_, MPs[0])
    PR_ = State(QR_, MPs[1])
    return PL, PR, PL_, PR_
