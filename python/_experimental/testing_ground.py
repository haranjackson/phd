from numpy import dot, sign
from numpy.linalg import solve
from numpy.random import rand

from gpr.misc.functions import det3
from gpr.misc.structures import State, Cvec

from gpr.systems.conserved import M_cons
from gpr.systems.primitive import M_prim
from gpr.systems.jacobians import dQdP


def systems(Q, d, MP):
    P = State(Q, MP)
    DQDP = dQdP(P)

    M1 = M_prim(Q, d, MP, pForm=1)
    M2 = M_prim(Q, d, MP, pForm=0)
    M3 = M_cons(Q, d, MP)
    M4 = solve(DQDP, dot(M3, DQDP))
    return M1, M2, M3, M4


def generate_vecs(MP):

    A = rand(3, 3)
    A /= sign(det3(A))
    ρ = det3(A) * MP.ρ0
    p = rand()
    v = rand(3)
    J = rand(3)

    Q = Cvec(ρ, p, v, A, J, MP)
    P = State(Q, MP)
    return Q, P


def cons_to_class(QL, QR, QL_, QR_, MPs):
    PL = State(QL, MPs[0])
    PR = State(QR, MPs[1])
    PL_ = State(QL_, MPs[0])
    PR_ = State(QR_, MPs[1])
    return PL, PR, PL_, PR_
