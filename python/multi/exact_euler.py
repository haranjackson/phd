from numpy import array, ones, sqrt, zeros

from gpr.functions import conserved, primitive
from gpr.variables import c_0
from options import dx


A_ = zeros([3,3])
J_ = zeros(3)
c_ = 0


def A0(r, y):
    return 2 / (r * (y+1))

def B(p, y, pINF):
    return p * (y-1) / (y+1) + (2*y*pINF) / (y+1)

def f0(z, P, params):

    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    c0 = c_0(r, p, y, pINF)

    if (z > p):
        temp = sqrt(A0(r,y) / (z+B(p,y,pINF)))
        return (z-p) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), (y-1)/(2*y))
        return 2 * c0 / (y-1) * (temp-1)

def f1(z, P, params):

    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    c0 = c_0(r, p, y, pINF)

    if (z > p):
        temp = sqrt( A0(r,y) / (z+B(p,y,pINF)) )
        return (1 - (z-p) / (2*(z+B(p,y,pINF))) ) * temp
    else:
        temp = pow((z+pINF)/(p+pINF), -(y+1)/(2*y))
        return temp * c0 / (y*(p+pINF))

def f(z, PL, PR, paramsL, paramsR):
    uR = PR.v[0]
    uL = PL.v[0]
    return f0(z, PL, paramsL) + f0(z, PR, paramsR) + uR - uL

def f_deriv(z, PL, PR, paramsL, paramsR):
    return f1(z, PL, paramsL) + f1(z, PR, paramsR)

def CHA(pk, pk_1):
    return 2 * abs(pk-pk_1) / abs(pk+pk_1)

def p_star(PL, PR, paramsL, paramsR):

    TOL = 1e-6
    p0 = (PL.p + PR.p) / 2
    p1 = 3*p0                       # CHA(p1,p0)=1 on first iteration

    while (CHA(p1,p0) > TOL):
        p0 = p1
        p1 = p0 - f(p0, PL, PR, paramsL, paramsR) / f_deriv(p0, PL, PR, paramsL, paramsR)
        if (p1<0):
            p1 = TOL
    return p1

def u_star(p_, PL, PR, paramsL, paramsR):
    uL = PL.v[0]
    uR = PR.v[0]
    return (uL + uR + f0(p_, PR, paramsR) - f0(p_, PL, paramsL)) / 2

def Q(p_, P, params):
    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    return sqrt( (p_ + B(p,y,pINF)) / A0(r,y) )

def Wfan(S, P, params, a):
    r = P.r
    u = P.v[0]
    p = P.p
    y = params.y
    pINF = params.pINF
    temp = 2/(y+1) + (y-1)*(u-S)/((y+1)*a)

    rf = r * pow(temp, 2/(y-1))
    vf = array([2 * (a + (y-1)*u/2 + S) / (y+1), 0, 0])
    pf = (p+pINF) * pow(temp, 2*y/(y-1)) - pINF

    return conserved(rf, pf, vf, A_, J_, c_, params, 0, 0, 0)

def r_star_shock(p_, P, params):

    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    temp1 = (p_+pINF)/(p+pINF) + (y-1)/(y+1)
    temp2 = (y-1)/(y+1)*(p_+pINF)/(p+pINF) + 1
    return r * temp1 / temp2

def r_star_fan(p_, P, params):

    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    return r * pow((p_+pINF)/(p+pINF), 1/y)

def c0_star(p_, P, params):

    r = P.r
    p = P.p
    y = params.y
    pINF = params.pINF
    c0 = c_0(r, p, y, pINF)
    return c0 * pow((p_+pINF)/(p+pINF), (y-1)/(2*y))

def exact_euler(n, t, x0, QL, QR, paramsL, paramsR):
    """ Returns the exact solution to the Euler equations at (x,t), given initial states PL for x<x0
        and PR for x>x0
    """
    ret = zeros([n, 1, 1, 18])
    PL = primitive(QL, paramsL, 0, 0, 0)
    PR = primitive(QR, paramsR, 0, 0, 0)

    rL = PL.r
    rR = PR.r
    uL = PL.v[0]
    uR = PR.v[0]
    pL = PL.p
    pR = PR.p
    c0L = c_0(rL, pL, paramsL.y, paramsL.pINF)
    c0R = c_0(rR, pR, paramsR.y, paramsR.pINF)

    p_ = p_star(PL, PR, paramsL, paramsR)
    u_ = u_star(p_, PL, PR, paramsL, paramsR)

    print('Interface:', u_ * t + x0)

    for i in range(n):
        x = (i+0.5)/n
        S = (x-x0)/t

        if (S < u_):

            if (p_ < pL):		# Left fan
                if (S < uL-c0L):
                    ret[i, 0, 0] = QL
                else:
                    STL = u_ - c0_star(p_, PL, paramsL)
                    if (S < STL):
                        ret[i, 0, 0] = Wfan(S, PL, paramsL, c0L)
                    else:
                        r_ = r_star_fan(p_, PL, paramsL)
                        v_ = array([u_, 0, 0])
                        ret[i, 0, 0] = conserved(r_, p_, v_, A_, J_, c_, paramsL, 0, 0, 0)

            else:				# Left shock
                SL = uL - Q(p_, PL, paramsL)/rL
                if (S < SL):
                    ret[i, 0, 0] = QL
                else:
                    r_ = r_star_shock(p_, PL, paramsL)
                    v_ = array([u_, 0, 0])
                    ret[i, 0, 0] = conserved(r_, p_, v_, A_, J_, c_, paramsL, 0, 0, 0)

        else:

            if (p_ < pR):		# Right fan
                if (uR+c0R < S):
                    ret[i, 0, 0] = QR
                else:
                    STR = u_ + c0_star(p_, PR, paramsR)
                    if (STR < S):
                        ret[i, 0, 0] = Wfan(S, PR, paramsR, -c0R)
                    else:
                        r_ = r_star_fan(p_, PR, paramsR)
                        v_ = array([u_, 0, 0])
                        ret[i, 0, 0] = conserved(r_, p_, v_, A_, J_, c_, paramsR, 0, 0, 0)

            else:				# Right shock
                SR = uR + Q(p_, PR, paramsR) / rR
                if (SR < S):
                    ret[i, 0, 0] = QR
                else:
                    r_ = r_star_shock(p_, PR, paramsR)
                    v_ = array([u_, 0, 0])
                    ret[i, 0, 0] = conserved(r_, p_, v_, A_, J_, c_, paramsR, 0, 0, 0)

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
