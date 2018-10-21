from gpr.misc.objects import material_params


""" SI Units """

TNT_JWL_SI = material_params(EOS='jwl', ρ0=1840, p0=0, cv=815,
                             Γ0=0.25, A=854.5, B=20.5, R1=4.6, R2=1.35,
                             b0=2100, τ1=inf)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, p0=1e5, γ=2.85, b0=1, μ=1e-2)


""" Scaled Units """

NM_PROD_JWL_SCALED = material_params(EOS='jwl', ρ0=1, p0=0, cv=0.0096,
                                     Γ0=1.237, A=67.69, B=-0.0406, R1=10, R2=1,
                                     b0=1300/6243.2, μ=6.2e-4/4.43e5)
