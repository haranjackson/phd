from gpr.misc.objects import material_params


""" SI Units """

Al_GRP_SI = material_params(EOS='gr', ρ0=2710, cv=900, Tref=300,
                            c0=5037, α=1, β=3.577, γ=2.088, b0=3160,
                            σY=0.4e9, τ0=1, n=100)

Cu_SMGP_SI = material_params(EOS='smg', ρ0=8930, cv=390,
                             c0=3939, Γ0=2, s=1.5, b0=2244,
                             σY=9e7, τ0=1, n=100)

Cu_GR_SI = material_params(EOS='gr', ρ0=8930, cv=390, Tref=300,
                           c0=3939, α=1, β=3, γ=2, b0=2141)

Cu_GRP_SI = material_params(EOS='gr', ρ0=8900, cv=390, Tref=300,
                            c0=3939, α=1, β=3, γ=2, b0=2141,
                            σY=4.5e7, τ0=0.92, n=10.1)

Cu_GRP_SI2 = material_params(EOS='gr', ρ0=8930, cv=390, Tref=300,
                             c0=3909, α=1, β=3, γ=2, b0=2100,
                             σY=7e7, τ0=1, n=10)

Cu_CC_SI = material_params(EOS='cc', ρ0=8900, cv=390,
                           Γ0=2, A=145.67, B=12.87, R1=2.99, R2=4.1, b0=2100)

W_SMGP_SI = material_params(EOS='smg', ρ0=17600, cv=134,
                            c0=4030, Γ0=1.43, s=1.24, b0=2888,
                            σY=1.5e9, τ0=1, n=100)

Steel_SMGP_SI = material_params(EOS='smg', ρ0=7870, cv=134,
                                c0=4569, Γ0=2.17, s=1.49, b0=3235,
                                σY=0.53e9, τ0=1, n=100)


""" CGS Units """

Al_SMGP_CGS = material_params(EOS='smg', ρ0=2.785, cv=9e-4,
                              c0=0.533, Γ0=2, s=1.338, b0=0.305,
                              σY=0.003, τ0=1, n=20)

Al_GR_CGS = material_params(EOS='gr', ρ0=2.71, cv=9e-4, Tref=300, c0=5.037,
                            α=1, β=3.577, γ=2.088, b0=3.16, cα=2, κ=204)

Cu_GR_CGS = material_params(EOS='gr', ρ0=8.9, cv=3.94e-4, Tref=300,
                            c0=3.909, α=1, β=3, γ=2, b0=2.1)
                           #κ=4e-8, cα=8.9*3.909*sqrt(4e-4/300))


""" Notes
bulk modulus (Pa) = K0 * ρ0
shear modulus (Pa) = b0^2 * ρ0
"""
