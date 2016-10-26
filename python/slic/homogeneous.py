from numpy import float, vectorize, zeros

from ader.fv_fluxes import Dos, Drus
from gpr.matrices_conserved import B0dot, flux
from options import dx, method


TOL = 1e-5
if method == 'osher':
    D = Dos
elif method == 'rusanov':
    D = Drus


def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1

def bound_below(x):
    if abs(x) < TOL:
        return TOL * sgn(x)
    else:
        return x

def limiter(d):
    if d <= 0:
        return 0
    elif d <= 0.5:
        return 2*d
    elif d <= 1:
        return 1
    else:
        return min([d, 2/(1+d), 2])

vbound_below = vectorize(bound_below, otypes=[float])
vlimiter = vectorize(limiter, otypes=[float])


def interface_values(un, params, subsystems, dt):

    unR = un[2:]
    unL = un[:-2]
    unM = un[1:-1]

    diff = unR - unL
    δ = (unM - unL) / vbound_below(unR - unM)
    ζ = vlimiter(δ)
    Δ = 0.25 * ζ * diff

    uL = unM - Δ
    uR = unM + Δ
    newDiff = uL - uR

    vL = uL[:,:,:,2:5] / uL[:,:,:,0]
    vR = uR[:,:,:,2:5] / uR[:,:,:,0]

    FL = zeros(uL.shape)
    FR = zeros(uR.shape)
    BL = newDiff.copy()
    BR = newDiff.copy()
    viscous = subsystems.viscous
    for i in range(len(FL)):
        FL[i,0,0] = flux(uL[i,0,0], 0, params, subsystems)
        FR[i,0,0] = flux(uR[i,0,0], 0, params, subsystems)
        B0dot(BL[i,0,0], newDiff[i,0,0], vL[i,0,0], viscous)
        B0dot(BR[i,0,0], newDiff[i,0,0], vR[i,0,0], viscous)

    fluxDiff = FL - FR

    d = dt / (2 * dx)
    u_L = uL + d * (fluxDiff + BL)
    u_R = uR + d * (fluxDiff + BR)
    return u_L, u_R

def flux_stepper(u, un, params, subsystems, dt):
    u_L, u_R = interface_values(un, params, subsystems, dt)
    d = dt / (2 * dx)
    for i in range(len(u)):
        qL1 = u_R[i,0,0]
        qM0 = u_L[i+1,0,0]
        qM1 = u_R[i+1,0,0]
        qR0 = u_L[i+2,0,0]
        u[i,0,0] -= d * (  D(qM1, qR0, 0, 1, params, subsystems)
                         + D(qM0, qL1, 0, 0, params, subsystems))
