from gpr.misc.objects import material_params


""" SI Units """

TNT_JWL_SI = material_params(EOS='jwl', ρ0=1840, cv=815,
                             Γ0=0.25, A=854.5, B=20.5, R1=4.6, R2=1.35,
                             b0=2100)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, γ=2.85, b0=1, μ=1e-2)


""" Scaled Units """

NM_JWL_Scaled = material_params(EOS='jwl', ρ0=1, cv=0.0067,
                                Γ0=0.3, A=4.578, B=0.128, R1=4.4, R2=1.2,
                                b0=1300/6243.2, μ=6.2e-4/4.43e5,

                                REACTION='i', Qc=0.115,
                                G1=1100, a=0.667, b=0.667, d=4, λ0=0.99,

                                MULTI=True, EOS_2='jwl', cv_2=0.0096,
                                Γ0_2=1.237, A_2=67.69, B_2=-0.0406, R1_2=10, R2_2=1,
                                b0_2=1300/6243.2, μ_2=6.2e-4/4.43e5)
