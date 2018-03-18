from numpy import array, inf, sqrt, zeros

from gpr.misc.objects import material_parameters, hyperelastic_params
from options import NV, RGFM


def cell_sizes(Lx, nx, Ly=1, ny=1, Lz=1, nz=1):
    return array([Lx / nx, Ly / ny, Lz / nz])


def riemann_IC(tf, nx, dX, QL, QR, MPL, MPR, x0):

    u = zeros([nx, NV])

    if RGFM:

        for i in range(nx):

            if i * dX[0] < x0:
                u[i] = QL
            else:
                u[i] = QR

            u[i, -1] = (i + 0.5) * dX[0] - x0

        return u, [MPL, MPR], tf, dX

    else:

        for i in range(nx):

            if i * dX[0] < x0:
                u[i] = QL
            else:
                u[i] = QR

        return u, [MPL], tf, dX


MP_AIR = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0,
                             b0=1, cα=1, μ=5e-4, Pr=2 / 3)

MP_WAT2 = material_parameters(EOS='sg', ρ0=1000, cv=950, p0=1e5, γ=4.4, pINF=6e8,
                              b0=1e-4, cα=1e-4, μ=1e-3, Pr=7)

MP_AIR2 = material_parameters(EOS='sg', ρ0=1.18, cv=721, p0=10100, γ=1.4, pINF=0,
                              b0=1, cα=1, μ=1.85e-5, Pr=0.714)

MP_HEL2 = material_parameters(EOS='sg', ρ0=0.163, cv=3127, p0=10100, γ=1.66, pINF=0,
                              b0=1, cα=1, μ=1.99e-5, Pr=0.688)


HYP_ALU = hyperelastic_params(ρ0=2.71, α=1, β=3.577, γ=2.088, cv=9e-4, T0=300,
                              b0=3.16, c0=6.22)

HYP_COP = hyperelastic_params(ρ0=8.9, α=1, β=3, γ=2, cv=4e-4, T0=300,
                              b0=2.1, c0=4.6)

MP_ALU_SG = material_parameters(EOS='sg', ρ0=2700, cv=897, p0=1e5, γ=3.4, pINF=21.5e9,
                                b0=3160, τ1=inf)

MP_COP_GR = material_parameters(EOS='gr', ρ0=8.9, cv=4e-4, p0=0,
                                c0=sqrt(4.6**2 - 4 / 3 * 2.1**2),
                                α=1, β=3, γ=2,
                                b0=2.1, τ1=inf,
                                Tref=300)  # , κ=4e-8, cα=8.9 * sqrt(4.6**2 - 4 / 3 * 2.1**2) * sqrt(4e-4 / 300))

MP_COP_SMG = material_parameters(EOS='smg', ρ0=8.9, cv=4e-4, p0=0,
                                 c0=3.909, Γ0=1.99, s=1.5, e0=0,
                                 b0=2.1, τ1=inf, β=3)

MP_COP_SMG_P = material_parameters(EOS='smg', ρ0=8.93, cv=1, p0=0,
                                   c0=0.394, Γ0=2, s=1.48, e0=0,
                                   b0=0.219, σY=9e-4, τ1=0.1, n=10, PLASTIC=True)

MP_COP_CC = material_parameters(EOS='cc', ρ0=8.9, cv=4e-4, p0=0,
                                Γ0=2, A=1.4567, B=0.1287, R1=2.99, R2=4.1,
                                b0=2.1, τ1=inf, β=3)
