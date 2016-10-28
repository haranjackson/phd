from numpy import exp, zeros

from options import Rc, minE, dx


def temperature(U, params):
    E = U[:,1] / U[:,0]
    λ = U[:,3] / U[:,0]
    E1 = E - params.Qc * (λ - minE)
    return  E1 / params.cv

def flux(U, n, params):
    T = temperature(U, params)
    q = params.α2 * T * U[:,2] / U[:,0]
    ret = zeros([n, 4])
    ret[:, 1] = q
    ret[:, 2] = T
    return ret

def thermal_stepper(u, params, dt):
    n = len(u)
    U = u[:, 0, 0, [0,1,14,17]]
    F = flux(U, n, params)

    UL = U[:-1]
    UR = U[1:]
    FL = F[:-1]
    FR = F[1:]

    Flf = 0.5 * (FL + FR + (dx/dt) * (UL - UR))
    Uri = 0.5 * (UL + UR + (dt/dx) * (FL-FR))
    Fri = flux(Uri, n-1, params)
    Ffo = 0.5 * (Flf + Fri)

    ret = u[1:-1]
    ret[:, 0, 0, 1] += (dt/dx) * (Ffo[:-1, 1] - Ffo[1:, 1])
    ret[:, 0, 0, 14] += (dt/dx) * (Ffo[:-1, 2] - Ffo[1:, 2])

    Unew = ret[:, 0, 0, [0,1,14,17]]
    Tnew = temperature(Unew, params)

    k1 = Tnew / Unew[:,0] * (params.r0 / (params.T0 * params.τ2))
    k2 = params.Bc * exp(-params.Ea / (Rc * Tnew))
    ret[:, 0, 0, 14] *= exp(-k1 * dt)
    ret[:, 0, 0, 17] *= exp(-k2 * dt)

    return ret
