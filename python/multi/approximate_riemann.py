from numpy import amax, concatenate, dot, zeros
from scipy.linalg import solve, inv

from gpr.eig import primitive_eigs
from gpr.functions import primitive, primitive_vector, primitive_to_conserved
#from gpr.primitive import source_primitive_reordered
from gpr.variables import sigma, heat_flux, sigma_A
from options import dx


starTOL = 1e-8


def check_star_convergence(QLstar, QRstar, paramsL, paramsR, viscous, thermal, reactive):
    PLstar = primitive(QLstar, paramsL, viscous, thermal, reactive)
    PRstar = primitive(QRstar, paramsR, viscous, thermal, reactive)
    sigLstar = sigma(PLstar.r, PLstar.A, paramsL.cs2)[0]
    sigRstar = sigma(PRstar.r, PRstar.A, paramsR.cs2)[0]
    qLstar = heat_flux(PLstar.T, PLstar.J, paramsL.alpha2)[0]
    qRstar = heat_flux(PRstar.T, PRstar.J, paramsR.alpha2)[0]
    return amax(abs(sigLstar-sigRstar)) < starTOL and abs(qLstar-qRstar) < starTOL

def riemann_constraints(q, params, sgn, viscous, thermal, reactive):
    _, Lhat, _ = primitive_eigs(q, params, viscous, thermal, reactive)
    P = primitive(q, params, viscous, thermal, reactive)
    r = P.r; p = P.p; A = P.A
    pINF = params.pINF; cs2 = params.cs2; alpha2 = params.alpha2

    q0 = heat_flux(P.T, P.J, alpha2)[0]
    sig0 = sigma(r, A, cs2)[0]
    dsdA = sigma_A(r, A, cs2)[0]
    Pi1 = dsdA[:,:,0]
    Pi2 = dsdA[:,:,1]
    Pi3 = dsdA[:,:,2]

    Lhat[:4] = 0
    Lhat[0, 0] = -q0 / r
    Lhat[0, 1] = q0 / (p + pINF)
    Lhat[0, 14] = alpha2 * P.T
    Lhat[1:4, 0] = sig0 / r
    Lhat[1:4, 2:5] = Pi1
    Lhat[1:4, 5:8] = Pi2
    Lhat[1:4, 8:11] = Pi3
    Lhat[4:8, 11:15] *= sgn

    return Lhat, inv(Lhat)

def star_stepper(QL, QR, paramsL, paramsR, dt, viscous, thermal, reactive,
                 SL=zeros(18), SR=zeros(18)):

    FIX_Q = 0

    PL = primitive(QL, paramsL, viscous, thermal, reactive)
    PR = primitive(QR, paramsR, viscous, thermal, reactive)
    LL, RL = riemann_constraints(QL, paramsL, -1, viscous, thermal, reactive)
    LR, RR = riemann_constraints(QR, paramsR, 1, viscous, thermal, reactive)

    TL = zeros([4, 4])
    TR = zeros([4, 4])
    TL[0] = RL[1,:4]
    TL[1:4] = RL[11:14, :4]
    TR[0] = RR[1,:4]
    TR[1:4] = RR[11:14, :4]

    sigL = sigma(PL.r, PL.A, paramsL.cs2)[0]
    sigR = sigma(PR.r, PR.A, paramsR.cs2)[0]
    qL = heat_flux(PL.T, PL.J, paramsL.alpha2)[0]
    qR = heat_flux(PR.T, PR.J, paramsR.alpha2)[0]
    xL = concatenate([[qL], sigL])
    xR = concatenate([[qR], sigR])

    if FIX_Q:
        q_ = 2 * (PL.T - PR.T) / (10 * dx * (1/paramsL.kappa + 1/paramsR.kappa))
        b = PR.v - PL.v + TR[1:4,0] * (q_ - qR) - TL[1:4,0] * (q_ - qL)
        b -= (dot(TR[1:4,1:], sigR) - dot(TL[1:4,1:], sigL))
        M = (TL - TR)[1:,1:]
        xStar = concatenate([[q_], solve(M, b)])
    else:
        yL = concatenate([[PL.p], PL.v])
        yR = concatenate([[PR.p], PR.v])
        xStar = solve(TL-TR, yR-yL + dot(TL, xL) - dot(TR, xR))

    cL = zeros(18)
    cR = zeros(18)
    cL[:4] = xStar - xL
    cR[:4] = xStar - xR

    PLvec = primitive_vector(PL)
    PRvec = primitive_vector(PR)
    PLstarvec = solve(LL, cL) + PLvec + dt*SL
    PRstarvec = solve(LR, cR) + PRvec + dt*SR
    QLstar = primitive_to_conserved(PLstarvec, paramsL, viscous, thermal, reactive)
    QRstar = primitive_to_conserved(PRstarvec, paramsR, viscous, thermal, reactive)
    return QLstar, QRstar

def star_states(QL, QR, paramsL, paramsR, dt, viscous, thermal, reactive):
#    LL, RL = riemann_constraints(QL, paramsL, -1, viscous, thermal, reactive)
#    LR, RR = riemann_constraints(QR, paramsR, 1, viscous, thermal, reactive)
#    SL = source_primitive_reordered(QL, paramsL)
#    SR = source_primitive_reordered(QR, paramsR)
    QLstar, QRstar = star_stepper(QL, QR, paramsL, paramsR, dt, viscous, thermal, reactive)
    while not check_star_convergence(QLstar, QRstar, paramsL, paramsR, viscous, thermal, reactive):
        QLstar, QRstar = star_stepper(QLstar, QRstar, paramsL, paramsR, dt, viscous, thermal, reactive)
    return QLstar, QRstar
