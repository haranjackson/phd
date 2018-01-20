from numpy import zeros

from gpr.systems.conserved import flux_cons_ref, Bdot_cons, system_cons
from gpr.systems.conserved import block_cons_ref, source_cons_ref
from gpr.systems.eigenvalues import max_abs_eigs
from options import nV


def flux_ref(ret, Q, d, MP):
    flux_cons_ref(ret, Q, d, MP)


def block_ref(ret, Q, d, MP):
    block_cons_ref(ret, Q, d, MP)


def source_ref(ret, Q, MP):
    source_cons_ref(ret, Q, MP)


def flux(Q, d, MP):
    ret = zeros(nV)
    flux_ref(ret, Q, d, MP)
    return ret


def block(Q, d, MP):
    ret = zeros([nV, nV])
    block_ref(ret, Q, d, MP)
    return ret


def source(Q, MP):
    ret = zeros(nV)
    source_ref(ret, Q, MP)
    return ret


def Bdot(ret, x, Q, d, MP):
    Bdot_cons(ret, x, Q, d, MP)


def system(Q, d, MP):
    return system_cons(Q, d, MP)


def max_eig(Q, d, MP):
    return max_abs_eigs(Q, d, MP)
