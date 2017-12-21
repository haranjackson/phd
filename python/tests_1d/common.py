from itertools import product

from numpy import eye, inf, zeros

from system.gpr.misc.objects import material_parameters, hyperelastic_params
from system.gpr.misc.structures import Cvec
from options import nx, ny, nz, nV, dx, RGFM


def riemann_IC(ρL, pL, vL, ρR, pR, vR, PARL, PARR=None, x0=0.5):
    """ constructs the riemann problem corresponding to the parameters given
    """
    AL = ρL**(1/3) * eye(3)
    JL = zeros(3)
    AR = ρR**(1/3) * eye(3)
    JR = zeros(3)

    if PARR is None:
        PARR = PARL

    QL = Cvec(ρL, pL, vL, AL, JL, PARL)
    QR = Cvec(ρR, pR, vR, AR, JR, PARR)

    u = zeros([nx, ny, nz, nV])

    if RGFM:

        for i, j, k in product(range(nx), range(ny), range(nz)):

            if i*dx < x0:
                u[i, j, k] = QL
            else:
                u[i, j, k] = QR

            u[i, j, k, -1] = i*dx - x0

        return u, [PARL, PARR]

    else:

        for i, j, k in product(range(nx), range(ny), range(nz)):

            if i*dx < x0:
                u[i, j, k] = QL
            else:
                u[i, j, k] = QR

        return u, [PARL]


PAR_AIR = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0,
                              cs=1, α=1, μ=5e-4, Pr=2/3)

PAR_WAT2 = material_parameters(EOS='sg', ρ0=1000, cv=950, p0=1e5, γ=4.4, pINF=6e8,
                               cs=1e-4, α=1e-4, μ=1e-3, Pr=7)

PAR_AIR2 = material_parameters(EOS='sg', ρ0=1.18, cv=721, p0=10100, γ=1.4, pINF=0,
                               cs=1, α=1, μ=1.85e-5, Pr=0.714)

PAR_HEL2 = material_parameters(EOS='sg', ρ0=0.163, cv=3127, p0=10100, γ=1.66, pINF=0,
                               cs=1, α=1, μ=1.99e-5, Pr=0.688)


HYP_ALU = hyperelastic_params(ρ0=2.71, α=1, β=3.577, γ=2.088, cv=9e-4, T0=300,
                              b0=3.16, c0=6.22)

HYP_COP = hyperelastic_params(ρ0=8.9, α=1, β=3, γ=2, cv=4e-4, T0=300,
                              b0=2.1, c0=4.6)


PAR_COP_SMG = material_parameters(EOS='smg', ρ0=8.9, cv=4e-4, p0=0,
                                  c0=3.909, Γ0=1.99, s=1.5,
                                  cs=2.1, τ1=inf, β=3)

PAR_COP_SMG_P = material_parameters(EOS='smg', ρ0=8.93, cv=1, p0=0,
                                    c0=0.394, Γ0=2, s=1.48,
                                    cs=0.219, σY=9e-4, τ1=0.1, n=10, PLASTIC=True)

PAR_COP_CC = material_parameters(EOS='cc', ρ0=8.9, cv=4e-4, p0=0,
                                 Γ0=2, A=1.4567, B=0.1287, ε1=2.99, ε2=4.1,
                                 cs=2.1, τ1=inf, β=3)