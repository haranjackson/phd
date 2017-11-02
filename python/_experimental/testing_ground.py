from system.gpr.misc.objects import material_parameters
from system.gpr.systems.conserved import system_cons
from system.gpr.systems.primitive import system_prim
from system.gpr.systems.jacobians import dQdP
from system.gpr.misc.structures import Cvec_to_Pclass, Cvec, Cvec_to_Pvec


def systems(Q, d, PAR):
    P = Cvec_to_Pclass(Q, PAR)
    DQDP = dQdP(P)

    M1 = system_prim(Q, d, PAR, pForm=1)
    M2 = system_prim(Q, d, PAR, pForm=0)
    M3 = system_cons(Q, d, PAR)
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
    α = 1.5
    κ = 1e-4

    return material_parameters(γ=γ, pINF=pinf, cv=cv, ρ0=ρ0, p0=p0, cs=cs, α=α,
                               μ=μ, κ=κ)

def generate_vecs(PAR=None):

    if PAR is None:
        PAR = generate_pars()

    A = rand(3,3)
    A /= sign(det(A))
    ρ = det(A)
    p = rand()
    v = rand(3)
    J = rand(3)

    Q = Cvec(ρ, p, v, A, J, 0, PAR)
    P = Cvec_to_Pvec(Q, PAR)
    Pc = Cvec_to_Pclass(Q, PAR)
    return Q, P, Pc

def cons_to_class(QL,QR,QL_,QR_,PARs):
    PL = Cvec_to_Pclass(QL,PARs[0])
    PR = Cvec_to_Pclass(QR,PARs[1])
    PL_ = Cvec_to_Pclass(QL_,PARs[0])
    PR_ = Cvec_to_Pclass(QR_,PARs[1])
    return PL, PR, PL_, PR_
