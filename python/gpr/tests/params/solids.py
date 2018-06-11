from numpy import inf

from gpr.misc.objects import material_params


""" SI Units """

Al_GRP_SI = material_params(EOS='gr', ρ0=2710, p0=0, cv=900, Tref=300,
                            c0=5037, α=1, β=3.577, γ=2.088, b0=3160,
                            σY=0.4e9, τ1=1, n=100)

Cu_SMGP_SI = material_params(EOS='smg', ρ0=8930, p0=0, cv=390,
                             c0=3940, Γ0=2, s=1.49,
                             b0=2244, σY=9e7, τ1=1, n=1000)

Cu_GR_SI = material_params(EOS='gr', ρ0=8930, p0=0, cv=390, Tref=300,
                           c0=3939, α=1, β=3, γ=2, b0=2141, τ1=inf)

# TODO: find b0
W_SMG_SI = material_params(EOS='smg', ρ0=17600, p0=0, cv=134,
                           c0=4030, Γ0=1.43, s=1.24, b0=2100, τ1=inf)


""" CGS Units """

Al_GR_CGS = material_params(EOS='gr', ρ0=2.71, p0=0, cv=9e-4, Tref=300,
                            c0=5.037, α=1, β=3.577, γ=2.088, b0=3.16, τ1=inf,
                            cα=2, κ=204)

Cu_GR_CGS = material_params(EOS='gr', ρ0=8.93, p0=0, cv=3.94e-4, Tref=300,
                            c0=3.909, α=1, β=3, γ=2, b0=2.1, τ1=inf)
# κ=4e-8, cα=8.9*3.909*sqrt(4e-4/300))
