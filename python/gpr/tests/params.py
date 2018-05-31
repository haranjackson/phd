from numpy import inf

from gpr.misc.objects import material_params, hyperelastic_params


""" SI Units """

Air_SG_SI = material_params(EOS='sg', ρ0=1.18, p0=10100, cv=721,
                            γ=1.4, b0=1, cα=1, μ=1.85e-5, Pr=0.714)

He_SG_SI = material_params(EOS='sg', ρ0=0.163, p0=10100, cv=3127,
                           γ=1.67, b0=1, cα=1, μ=1.99e-5, Pr=0.688)

H20_SG_SI = material_params(EOS='sg', ρ0=1000, p0=1e5, cv=950,
                            γ=4.4, pINF=6e8, b0=1e-4, cα=1e-4, μ=1e-3, Pr=7)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, p0=1e5, γ=2.85, b0=1e-4, μ=1e-5)


Al_GR_SI = material_params(EOS='gr', ρ0=2710, p0=0, cv=900, Tref=300,
                           c0=5037, α=1, β=3.577, γ=2.088, b0=3160, τ1=inf)

Al_GRP_SI = material_params(EOS='gr', ρ0=2710, p0=0, cv=900, Tref=300,
                            c0=5037, α=1, β=3.577, γ=2.088, b0=3160,
                            σY=0.4e9, τ1=1, n=100, PLASTIC=True)

Cu_SMG_SI = material_params(EOS='smg', ρ0=8930, p0=0, cv=390,
                            c0=3909, Γ0=1.99, s=1.48, b0=2100, τ1=inf, β=3)

Cu_SMGP_SI = material_params(EOS='smg', ρ0=8930, p0=0, cv=390,
                             c0=3909, Γ0=1.99, s=1.48,
                             b0=2100, σY=9e7, τ1=1, n=500, PLASTIC=True)

Cu_GR_SI = material_params(EOS='gr', ρ0=8930, p0=0, cv=390, Tref=300,
                           c0=3909, α=1, β=3, γ=2, b0=2100, τ1=inf)

W_SMG_SI = material_params(EOS='smg', ρ0=17600, p0=0, cv=134,
                           c0=4030, Γ0=1.43, s=1.24, b0=2100, τ1=inf, β=3)


""" CGS Units """

Al_GR_CGS = material_params(EOS='gr', ρ0=2.71, p0=0, cv=9e-4, Tref=300,
                            c0=5.037, α=1, β=3.577, γ=2.088, b0=3.16, τ1=inf)

Cu_GR_CGS = material_params(EOS='gr', ρ0=8.93, p0=0, cv=3.94e-4, Tref=300,
                            c0=3.909, α=1, β=3, γ=2, b0=2.1, τ1=inf)
# κ=4e-8, cα=8.9*3.909*sqrt(4e-4/300))

Cu_SMG_CGS = material_params(EOS='smg', ρ0=8.93, p0=0, cv=3.94e-4,
                             c0=3.909, Γ0=1.99, s=1.48, b0=2.1, τ1=inf, β=3)

Cu_CC_CGS = material_params(EOS='cc', ρ0=8.93, p0=0, cv=3.94e-4,
                            Γ0=2, A=1.4567, B=0.1287, R1=2.99, R2=4.1,
                            b0=2.1, τ1=inf, β=3)


""" Non-Dimensionalized """

Air_SG_ND = material_params(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0, b0=1,
                            cα=1, μ=5e-4, Pr=2/3)


""" Alternative EOSs """

Al_HYP_CGS = hyperelastic_params(ρ0=2.71, α=1, β=3.577, γ=2.088, cv=9e-4,
                                 T0=300, b0=3.16, c0=6.22)

Cu_HYP_SI = hyperelastic_params(ρ0=8900, α=1, β=3, γ=2, cv=400,
                                T0=300, b0=2100, c0=4600)

Cu_HYP_CGS = hyperelastic_params(ρ0=8.9, α=1, β=3, γ=2, cv=4e-4,
                                 T0=300, b0=2.1, c0=4.6)

VAC = material_params(EOS='vac', ρ0=0, cv=0, p0=0)
