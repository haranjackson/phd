from gpr.misc.objects import material_params


""" SI Units """

Air_SG_SI = material_params(EOS='sg', ρ0=1.18, p0=101325, cv=721,
                            γ=1.4, b0=1, cα=1, μ=1.85e-5, Pr=0.714)

He_SG_SI = material_params(EOS='sg', ρ0=0.163, p0=101325, cv=3127,
                           γ=5/3, b0=1, cα=1, μ=1.99e-5, Pr=0.688)

H20_SG_SI = material_params(EOS='sg', ρ0=997, p0=3.5e8, cv=950,
                            γ=4.4, pINF=6e8, b0=1, cα=1, μ=1e-3, Pr=7)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, p0=1e5, γ=2.85, b0=1e-4, μ=1e-5)
