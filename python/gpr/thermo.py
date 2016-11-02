from numpy import exp, zeros

from options import Rc, minE, dx


def temperature(U, PAR):
    E = U[:,1] / U[:,0]
    λ = U[:,3] / U[:,0]
    E1 = E - PAR.Qc * (λ - minE)
    return  E1 / PAR.cv

def flux(U, n, PAR):
    T = temperature(U, PAR)
    q = PAR.α2 * T * U[:,2] / U[:,0]
    ret = zeros([n, 4])
    ret[:, 1] = q
    ret[:, 2] = T
    return ret

def thermal_stepper(u, dt, PAR):
    n = len(u)
    U = u[:, 0, 0, [0,1,14,17]]
    F = flux(U, n, PAR)

    UL = U[:-1]
    UR = U[1:]
    FL = F[:-1]
    FR = F[1:]

    Flf = 0.5 * (FL + FR + (dx/dt) * (UL - UR))
    Uri = 0.5 * (UL + UR + (dt/dx) * (FL-FR))
    Fri = flux(Uri, n-1, PAR)
    Ffo = 0.5 * (Flf + Fri)

    ret = u[1:-1]
    ret[:, 0, 0, 1] += (dt/dx) * (Ffo[:-1, 1] - Ffo[1:, 1])
    ret[:, 0, 0, 14] += (dt/dx) * (Ffo[:-1, 2] - Ffo[1:, 2])

    Unew = ret[:, 0, 0, [0,1,14,17]]
    Tnew = temperature(Unew, PAR)

    k1 = Tnew / Unew[:,0] * (PAR.r0 / (PAR.T0 * PAR.τ2))
    k2 = PAR.Bc * exp(-PAR.Ea / (Rc * Tnew))
    ret[:, 0, 0, 14] *= exp(-k1 * dt)
    ret[:, 0, 0, 17] *= exp(-k2 * dt)

    return ret
