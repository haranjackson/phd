from numpy import inf

from gpr.misc.objects import material_params


""" SI Units """

Al_GRP_SI = material_params(EOS='gr', ρ0=2710, p0=0, cv=900, Tref=300,
                            c0=5037, α=1, β=3.577, γ=2.088, b0=3160,
                            σY=0.4e9, τ1=1, n=100)

Cu_SMGP_SI = material_params(EOS='smg', ρ0=8930, p0=0, cv=390,
                             c0=3940, Γ0=2, s=1.49, b0=2244,
                             σY=9e7, τ1=1, n=100)

Cu_GR_SI = material_params(EOS='gr', ρ0=8930, p0=0, cv=390, Tref=300,
                           c0=3939, α=1, β=3, γ=2, b0=2141, τ1=inf)

Cu_GRP_SI = material_params(EOS='gr', ρ0=8900, p0=0, cv=390, Tref=300,
                            c0=3939, α=1, β=3, γ=2, b0=2141,
                            σY=4.5e7, τ1=0.92, n=10.1)

Cu_CC_SI = material_params(EOS='cc', ρ0=8900, p0=0, cv=390,
                           Γ0=2, A=145.67, B=12.87, R1=2.99, R2=4.1, b0=2100,
                           τ1=inf)

W_SMGP_SI = material_params(EOS='smg', ρ0=19300, p0=0, cv=134,
                            c0=4030, Γ0=1.43, s=1.24, b0=2888,
                            σY=15e7, τ1=1, n=100)

TNT_JWL_SI = material_params(EOS='jwl', ρ0=1840, p0=0, cv=815,
                             Γ0=0.25, A=854.5, B=20.5, R1=4.6, R2=1.35,
                             b0=2100, τ1=inf)

Steel_SMGP_SI = material_params(EOS='smg', ρ0=19300, p0=0, cv=134,
                                c0=4030, Γ0=1.43, s=1.24, b0=2888,
                                σY=15e7, τ1=1, n=100)


""" CGS Units """

Al_GR_CGS = material_params(EOS='gr', ρ0=2.71, p0=0, cv=9e-4, Tref=300,
                            c0=5.037, α=1, β=3.577, γ=2.088, b0=3.16, τ1=inf,
                            cα=2, κ=204)

Cu_GR_CGS = material_params(EOS='gr', ρ0=8.9, p0=0, cv=3.94e-4, Tref=300,
                            c0=3.909, α=1, β=3, γ=2, b0=2.1, τ1=inf)
# κ=4e-8, cα=8.9*3.909*sqrt(4e-4/300))



""" Notes
bulk modulus (Pa) = K0 * ρ0
shear modulus (Pa) = b0^2 * ρ0
"""
