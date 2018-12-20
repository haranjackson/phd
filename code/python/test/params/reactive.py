from gpr.misc.objects import material_params


""" SI Units """

TNT_JWL_SI = material_params(EOS='jwl', ρ0=1840, cv=815,
                             Γ0=0.25, A=854.5, B=20.5, R1=4.6, R2=1.35,
                             b0=2100)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, γ=2.85, b0=1, μ=1e-2)


""" Scaled Units """

# TODO: figure out NM_JWL EOSs

NM_CC_SI = material_params(EOS='cc', ρ0=1134, cv=1714,
                           Γ0=1.19, A=0.819e9, B=1.51e9, R1=4.53, R2=1.42,
                           b0=1, μ=6.2e-4,
                           REACTION='a', Qc=4.48e3, Bc=6.9e10, Ea=11350, Rc=1)

NM_JWL_SI = material_params(EOS='jwl', ρ0=1137, cv=1.4272e-3,
                            Γ0=1.237, A=3000, B=1.8003, R1=10, R2=1,
                            b0=1300, μ=6.2e-4,

                            REACTION='i', Qc=12402,
                            G1=121875e8, a=0.667, b=0.667, d=4, λ0=0.99,

                            MULTI=True, EOS_2='jwl', cv_2=1e-3,
                            Γ0_2=0.3, A_2=209.2, B_2=-5.689, R1_2=4.4, R2_2=1.2,
                            b0_2=1300, μ_2=6.2e-4)

NM_JWL_Scaled = material_params(
        EOS='jwl', ρ0=1, cv=0.0096,
        Γ0=1.237, A=67.69, B=-0.0406, R1=10, R2=1,
        b0=1300/6243.2, μ=6.2e-4/4.43e5,

        REACTION='i', Qc=0.115,
        G1=1100, a=0.667, b=0.667, d=4, λ0=0.99,

        MULTI=True, EOS_2='jwl', cv_2=0.0067,
        Γ0_2=0.3, A_2=4.578, B_2=0.128, R1_2=4.4, R2_2=1.2,
        b0_2=1300/6243.2, μ_2=6.2e-4/4.43e5)