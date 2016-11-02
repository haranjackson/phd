from numpy import array, ones, sqrt, zeros

from gpr.variables.vectors import conserved, primitive
from gpr.variables.wavespeeds import c_0
from options import dx


A_ = zeros([3,3])
J_ = zeros(3)
c_ = 0


def A0(ρ, γ):
    return 2 / (ρ * (γ+1))

def B(p, γ, pINF):
    return p * (γ-1) / (γ+1) + (2*γ*pINF) / (γ+1)

def f0(z, P, PAR):

    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    c0 = c_0(ρ, p, γ, pINF)

    if (z > p):
        temp = sqrt(A0(ρ,γ) / (z+B(p,γ,pINF)))
        return (z-p) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), (γ-1)/(2*γ))
        return 2 * c0 / (γ-1) * (temp-1)

def f1(z, P, PAR):

    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    c0 = c_0(ρ, p, γ, pINF)

    if (z > p):
        temp = sqrt( A0(ρ,γ) / (z+B(p,γ,pINF)) )
        return (1 - (z-p) / (2*(z+B(p,γ,pINF))) ) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), -(γ+1)/(2*γ))
        return temp * c0 / (γ*(p+pINF))

def f(z, PL, PR, PARL, PARR):
    uR = PR.v[0]
    uL = PL.v[0]
    return f0(z, PL, PARL) + f0(z, PR, PARR) + uR - uL

def f_deriv(z, PL, PR, PARL, PARR):
    return f1(z, PL, PARL) + f1(z, PR, PARR)

def CHA(pk, pk_1):
    return 2 * abs(pk-pk_1) / abs(pk+pk_1)

def p_star(PL, PR, PARL, PARR):

    TOL = 1e-6
    p0 = (PL.p + PR.p) / 2
    p1 = 3*p0                       # CHA(p1,p0)=1 on first iteration

    while (CHA(p1,p0) > TOL):
        p0 = p1
        p1 = p0 - f(p0, PL, PR, PARL, PARR) / f_deriv(p0, PL, PR, PARL, PARR)
        if (p1<0):
            p1 = TOL
    return p1

def u_star(p_, PL, PR, PARL, PARR):
    uL = PL.v[0]
    uR = PR.v[0]
    return (uL + uR + f0(p_, PR, PARR) - f0(p_, PL, PARL)) / 2

def Q(p_, P, PAR):
    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    return sqrt( (p_ + B(p,γ,pINF)) / A0(ρ,γ) )

def Wfan(S, P, a, PAR):
    ρ = P.ρ
    u = P.v[0]
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    temp = 2/(γ+1) + (γ-1)*(u-S)/((γ+1)*a)

    rf = ρ * pow(temp, 2/(γ-1))
    vf = array([2 * (a + (γ-1)*u/2 + S) / (γ+1), 0, 0])
    pf = (p+pINF) * pow(temp, 2*γ/(γ-1)) - pINF

    return conserved(rf, pf, vf, A_, J_, c_, PAR, 0, 0, 0)

def ρ_star_shock(p_, P, PAR):

    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    temp1 = (p_+pINF)/(p+pINF) + (γ-1)/(γ+1)
    temp2 = (γ-1)/(γ+1)*(p_+pINF)/(p+pINF) + 1
    return ρ * temp1 / temp2

def ρ_star_fan(p_, P, PAR):

    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    return ρ * pow((p_+pINF)/(p+pINF), 1/γ)

def c0_star(p_, P, PAR):

    ρ = P.ρ
    p = P.p
    γ = PAR.γ
    pINF = PAR.pINF
    c0 = c_0(ρ, p, γ, pINF)
    return c0 * pow((p_+pINF)/(p+pINF), (γ-1)/(2*γ))

def exact_euler(n, t, x0, QL, QR, PARL, PARR):
    """ Returns the exact solution to the Euler equations at (x,t), given initial states PL for x<x0
        and PR for x>x0
    """
    ret = zeros([n, 1, 1, 18])
    PL = primitive(QL, PARL, 0, 0, 0)
    PR = primitive(QR, PARR, 0, 0, 0)

    ρL = PL.ρ
    ρR = PR.ρ
    uL = PL.v[0]
    uR = PR.v[0]
    pL = PL.p
    pR = PR.p
    c0L = c_0(ρL, pL, PARL.γ, PARL.pINF)
    c0R = c_0(ρR, pR, PARR.γ, PARR.pINF)

    p_ = p_star(PL, PR, PARL, PARR)
    u_ = u_star(p_, PL, PR, PARL, PARR)

    print('Interface:', u_ * t + x0)

    for i in range(n):
        x = (i+0.5)/n
        S = (x-x0)/t

        if (S < u_):

            if (p_ < pL):		# Left fan
                if (S < uL-c0L):
                    ret[i, 0, 0] = QL
                else:
                    STL = u_ - c0_star(p_, PL, PARL)
                    if (S < STL):
                        ret[i, 0, 0] = Wfan(S, PL, c0L, PARL)
                    else:
                        ρ_ = ρ_star_fan(p_, PL, PARL)
                        v_ = array([u_, 0, 0])
                        ret[i, 0, 0] = conserved(ρ_, p_, v_, A_, J_, c_, PARL, 0, 0, 0)

            else:				# Left shock
                SL = uL - Q(p_, PL, PARL) / ρL
                if (S < SL):
                    ret[i, 0, 0] = QL
                else:
                    ρ_ = ρ_star_shock(p_, PL, PARL)
                    v_ = array([u_, 0, 0])
                    ret[i, 0, 0] = conserved(ρ_, p_, v_, A_, J_, c_, PARL, 0, 0, 0)

        else:

            if (p_ < pR):		# Right fan
                if (uR+c0R < S):
                    ret[i, 0, 0] = QR
                else:
                    STR = u_ + c0_star(p_, PR, PARR)
                    if (STR < S):
                        ret[i, 0, 0] = Wfan(S, PR, -c0R, PARR)
                    else:
                        ρ_ = ρ_star_fan(p_, PR, PARR)
                        v_ = array([u_, 0, 0])
                        ret[i, 0, 0] = conserved(ρ_, p_, v_, A_, J_, c_, PARR, 0, 0, 0)

            else:				# Right shock
                SR = uR + Q(p_, PR, PARR) / ρR
                if (SR < S):
                    ret[i, 0, 0] = QR
                else:
                    ρ_ = ρ_star_shock(p_, PR, PARR)
                    v_ = array([u_, 0, 0])
                    ret[i, 0, 0] = conserved(ρ_, p_, v_, A_, J_, c_, PARR, 0, 0, 0)

    return ret

def mask(u, Q1, Q2):
    n = len(u)
    ret = ones(n)
    for i in range(n):
        if (u[i,0,0]==Q1).all() or (u[i,0,0]==Q2).all():
            ret[i] = 0
    return ret

def multi_euler(n, t, interfaceLocations, initialStates, materialParameters):
    m = len(interfaceLocations)
    ret = zeros([n, 1, 1, 18])
    solutions = [exact_euler(n, t, interfaceLocations[i], initialStates[i], initialStates[i+1],
                             materialParameters[i], materialParameters[i+1]) for i in range(m)]

    intLocs = [0] + interfaceLocations + [1]
    for i in range(m):
        sol = solutions[i]
        solMask = mask(sol, initialStates[i], initialStates[i+1])
        for j in range(n):
            if solMask[j]:
                ret[j] = sol[j]
            elif intLocs[i] <= j*dx <= intLocs[i+1]:
                ret[j] = initialStates[i]
            elif intLocs[i+1] <= j*dx <= intLocs[i+2]:
                ret[j] = initialStates[i+1]

    return ret
