from numpy import zeros

from system.gpr.systems.conserved import flux_cons_ref, Bdot_cons, system_cons
from system.gpr.systems.conserved import block_cons_ref, source_cons_ref
from options import nV


def flux_ref(ret, Q, d, PAR):
    flux_cons_ref(ret, Q, d, PAR)

def block_ref(ret, Q, d, PAR):
    block_cons_ref(ret, Q, d, PAR)

def source_ref(ret, Q, PAR):
    source_cons_ref(ret, Q, PAR)

def flux(Q, d, PAR):
    ret = zeros(nV)
    flux_ref(ret, Q, d, PAR)
    return ret

def block(Q, d, PAR):
    ret = zeros([nV, nV])
    block_ref(ret, Q, d, PAR)
    return ret

def source(Q, PAR):
    ret = zeros(nV)
    source_ref(ret, Q, PAR)
    return ret

def Bdot(ret, x, Q, d, PAR):
    Bdot_cons(ret, x, Q, d, PAR)

def system(Q, d, PAR):
    return system_cons(Q, d, PAR)
