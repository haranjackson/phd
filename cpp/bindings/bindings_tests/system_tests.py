import GPRpy

from numpy import dot, zeros
from numpy.random import rand

from bindings_tests.test_functions import check, generate_vector

from gpr.sys.conserved import F_cons, B_cons, S_cons, M_cons


""" EQUATIONS """


def F_test(d, MP):

    Q = generate_vector(MP)
    NV = len(Q)

    F_cp = zeros(NV)
    GPRpy.system.flux(F_cp, Q, d, MP)
    F_py = F_cons(Q, d, MP)

    print("F     ", check(F_cp, F_py))
    return F_cp, F_py


def S_test(d, MP):

    Q = generate_vector(MP)
    NV = len(Q)

    S_cp = zeros(NV)
    GPRpy.system.source(S_cp, Q, MP)
    S_py = S_cons(Q, MP)

    print("S     ", check(S_cp, S_py))
    return S_cp, S_py


def B_test(d, MP):

    Q = generate_vector(MP)
    NV = len(Q)
    x = rand(NV)

    Bx_cp = zeros(NV)
    Bx_py = zeros(NV)
    GPRpy.system.Bdot(Bx_cp, Q, x, d, MP)
    B_py = B_cons(Q, d, MP)
    Bx_py = dot(B_py, x)

    print("Bdot  ", check(Bx_cp, Bx_py))
    return Bx_cp, Bx_py


def M_test(d, MP):

    Q = generate_vector(MP)
    M_cp = GPRpy.system.system_matrix(Q, d, MP)
    M_py = M_cons(Q, d, MP)

    print("M     ", check(M_cp, M_py))
    return M_cp, M_py
