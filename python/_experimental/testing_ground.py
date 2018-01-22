from numpy import dot, sign
from numpy.linalg import solve
from numpy.random import rand

from gpr.misc.functions import det3
from gpr.misc.objects import material_parameters
from gpr.misc.structures import Cvec_to_Pclass, Cvec, Cvec_to_Pvec

from gpr.systems.conserved import system_cons
from gpr.systems.primitive import system_prim
from gpr.systems.jacobians import dQdP


def systems(Q, d, MP):
    P = Cvec_to_Pclass(Q, MP)
    DQDP = dQdP(P)

    M1 = system_prim(Q, d, MP, pForm=1)
    M2 = system_prim(Q, d, MP, pForm=0)
    M3 = system_cons(Q, d, MP)
    M4 = solve(DQDP, dot(M3, DQDP))
    return M1, M2, M3, M4

def generate_pars():
    γ = 1.4
    cv = 2.5
    pinf = 0
    ρ0 = 1
    p0 = 1
    cs = 2
    μ = 1e-3
    cα = 1.5
    κ = 1e-4

    return material_parameters(EOS='sg', γ=γ, pINF=pinf, cv=cv, ρ0=ρ0, p0=p0,
                               b0=cs, cα=cα, μ=μ, κ=κ)

def generate_vecs(MP=None):

    if MP is None:
        MP = generate_pars()

    A = rand(3,3)
    A /= sign(det3(A))
    ρ = det3(A)
    p = rand()
    v = rand(3)
    J = rand(3)

    Q = Cvec(ρ, p, v, A, J, MP)
    P = Cvec_to_Pvec(Q, MP)
    Pc = Cvec_to_Pclass(Q, MP)
    return Q, P, Pc

def cons_to_class(QL,QR,QL_,QR_,MPs):
    PL = Cvec_to_Pclass(QL,MPs[0])
    PR = Cvec_to_Pclass(QR,MPs[1])
    PL_ = Cvec_to_Pclass(QL_,MPs[0])
    PR_ = Cvec_to_Pclass(QR_,MPs[1])
    return PL, PR, PL_, PR_
