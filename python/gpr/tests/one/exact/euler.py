from numpy import array, ones, sqrt, zeros

from gpr.misc.structures import Cvec, State
from gpr.vars.wavespeeds import c_0


A_ = zeros([3, 3])
J_ = zeros(3)


def A0(r, y):
    return 2 / (r * (y+1))


def B(p, y, pINF):
    return p * (y-1) / (y+1) + (2*y*pINF) / (y+1)


def f0(z, P, MP):

    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    c0 = c_0(r, p, eye(3), MP)

    if (z > p):
        temp = sqrt(A0(r, y) / (z+B(p, y, pINF)))
        return (z-p) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), (y-1)/(2*y))
        return 2 * c0 / (y-1) * (temp-1)


def f1(z, P, MP):

    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    c0 = c_0(r, p, eye(3), MP)

    if (z > p):
        temp = sqrt(A0(r, y) / (z+B(p, y, pINF)))
        return (1 - (z-p) / (2*(z+B(p, y, pINF)))) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), -(y+1)/(2*y))
        return temp * c0 / (y*(p+pINF))


def f(z, PL, PR, MPL, MPR):
    uR = PR.v[0]
    uL = PL.v[0]
    return f0(z, PL, MPL) + f0(z, PR, MPR) + uR - uL


def f_deriv(z, PL, PR, MPL, MPR):
    return f1(z, PL, MPL) + f1(z, PR, MPR)


def CHA(pk, pk_1):
    return 2 * abs(pk-pk_1) / abs(pk+pk_1)


def p_star(PL, PR, MPL, MPR):

    TOL = 1e-6
    p0 = (PL.p() + PR.p()) / 2
    p1 = 3*p0                       # CHA(p1,p0)=1 on first iteration

    while (CHA(p1, p0) > TOL):
        p0 = p1
        p1 = p0 - f(p0, PL, PR, MPL, MPR) / f_deriv(p0, PL, PR, MPL, MPR)
        if (p1 < 0):
            p1 = TOL
    return p1


def u_star(p_, PL, PR, MPL, MPR):
    uL = PL.v[0]
    uR = PR.v[0]
    return (uL + uR + f0(p_, PR, MPR) - f0(p_, PL, MPL)) / 2


def Q(p_, P, MP):
    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    return sqrt((p_ + B(p, y, pINF)) / A0(r, y))


def Wfan(S, P, MP, a):
    r = P.ρ
    u = P.v[0]
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    temp = 2/(y+1) + (y-1)*(u-S)/((y+1)*a)

    rf = r * pow(temp, 2/(y-1))
    vf = array([2 * (a + (y-1)*u/2 + S) / (y+1), 0, 0])
    pf = (p+pINF) * pow(temp, 2*y/(y-1)) - pINF

    return Cvec(rf, pf, vf, A_, J_, MP)


def r_star_shock(p_, P, MP):

    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    temp1 = (p_+pINF)/(p+pINF) + (y-1)/(y+1)
    temp2 = (y-1)/(y+1)*(p_+pINF)/(p+pINF) + 1
    return r * temp1 / temp2


def r_star_fan(p_, P, MP):

    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    return r * pow((p_+pINF)/(p+pINF), 1/y)


def c0_star(p_, P, MP):

    r = P.ρ
    p = P.p()
    y = MP.γ
    pINF = MP.pINF
    c0 = c_0(r, p, eye(3), MP)
    return c0 * pow((p_+pINF)/(p+pINF), (y-1)/(2*y))


def exact_euler(n, t, x0, QL, QR, MPL, MPR):
    """ Returns the exact solution to the Euler equations at (x,t),
        given initial states PL for x<x0 and PR for x>x0
    """
    ret = zeros([n, len(QL)])
    PL = State(QL, MPL)
    PR = State(QR, MPR)

    ρL = PL.ρ
    ρR = PR.ρ
    uL = PL.v[0]
    uR = PR.v[0]
    pL = PL.p()
    pR = PR.p()
    c0L = c_0(ρL, pL, eye(3), MPL)
    c0R = c_0(ρR, pR, eye(3), MPR)

    p_ = p_star(PL, PR, MPL, MPR)
    u_ = u_star(p_, PL, PR, MPL, MPR)

    print('Interface:', u_ * t + x0)

    for i in range(n):
        x = (i+0.5)/n
        S = (x-x0)/t

        if (S < u_):

            if (p_ < pL):		# Left fan
                if (S < uL-c0L):
                    ret[i] = QL
                else:
                    STL = u_ - c0_star(p_, PL, MPL)
                    if (S < STL):
                        ret[i] = Wfan(S, PL, MPL, c0L)
                    else:
                        r_ = r_star_fan(p_, PL, MPL)
                        v_ = array([u_, 0, 0])
                        ret[i] = Cvec(r_, p_, v_, A_, J_, MPL)

            else:				# Left shock
                SL = uL - Q(p_, PL, MPL) / ρL
                if (S < SL):
                    ret[i] = QL
                else:
                    r_ = r_star_shock(p_, PL, MPL)
                    v_ = array([u_, 0, 0])
                    ret[i] = Cvec(r_, p_, v_, A_, J_, MPL)

        else:

            if (p_ < pR):		# Right fan
                if (uR+c0R < S):
                    ret[i] = QR
                else:
                    STR = u_ + c0_star(p_, PR, MPR)
                    if (STR < S):
                        ret[i] = Wfan(S, PR, MPR, -c0R)
                    else:
                        r_ = r_star_fan(p_, PR, MPR)
                        v_ = array([u_, 0, 0])
                        ret[i] = Cvec(ρ1, p, v, A, J, MPR)

            else:				# Right shock
                SR = uR + Q(p_, PR, MPR) / ρR
                if (SR < S):
                    ret[i] = QR
                else:
                    r_ = r_star_shock(p_, PR, MPR)
                    v_ = array([u_, 0, 0])
                    ret[i] = Cvec(r_, p_, v_, A_, J_, MPR)

    return ret


def mask(u, Q1, Q2):
    n = len(u)
    ret = ones(n)
    for i in range(n):
        if (u[i] == Q1).all() or (u[i] == Q2).all():
            ret[i] = 0
    return ret


def multi_euler(n, t, interfaceLocations, initialStates, MPs):
    m = len(interfaceLocations)
    ret = zeros([n, 18])
    solutions = [exact_euler(n, t, interfaceLocations[i],
                             initialStates[i], initialStates[i+1],
                             MPs[i], MPs[i+1]) for i in range(m)]

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
