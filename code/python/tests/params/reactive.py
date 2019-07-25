from gpr.misc.objects import material_params


""" SI Units """

TNT_JWL_SI = material_params(EOS='jwl', ρ0=1840, cv=815,
                             Γ0=0.25, A=854.5e9, B=20.5e9, R1=4.6, R2=1.35,
                             b0=2100)

PBX_SG_SI = material_params(EOS='sg', ρ0=1840, γ=2.85, b0=1, μ=1e-2)

C4_JWL_SI = material_params(EOS='jwl', ρ0=1601, cv=2.487e6/1601,
                            Γ0=0.8938, A=7.781e13, B=-5.031e9,
                            R1=11.3, R2=1.13, b0=1487,
                            REACTION='i', Qc=9e9/1601, I=4e6, G1=1.4e-20, G2=0,
                            a=0.0367, b=2/3, c=2/3, d=1/3, e=2/3, g=2/3,
                            x=7, y=2, z=3, φI=0.022, φG1=1, φG2=0)

C4_JWL_SI2 = material_params(EOS='jwl', ρ0=1601, cv=1e6/1601,
                            Γ0=0.25, A=6.0977e11, B=0.1295e11,
                            R1=4.5, R2=1.4, b0=1487,
                            REACTION='i', Qc=9e9/1601, I=4e6, G1=1.4e-20, G2=0,
                            a=0.0367, b=2/3, c=2/3, d=1/3, e=2/3, g=2/3,
                            x=7, y=2, z=3, φI=0.022, φG1=1, φG2=0)

C4_JWL_ARR_SI = material_params(EOS='jwl', ρ0=1601, cv=2.487e6/1601,
                                Γ0=0.8938, A=7.781e13, B=-5.031e9,
                                R1=11.3, R2=1.13, b0=1487,
                                REACTION='a', Qc=5.18e6, Bc=1.16e13, Ea=1.438e5,
                                Rc=8.314)

NM_CC_SI = material_params(EOS='cc', ρ0=1134, cv=1714,
                           Γ0=1.19, A=0.819e9, B=1.51e9, R1=4.53, R2=1.42,
                           b0=1, μ=6.2e-4,
                           REACTION='a', Qc=4.48e6, Bc=6.9e10, Ea=11350, Rc=1)

NM_JWL_SI = material_params(EOS='jwl', ρ0=1137, cv=1.4272e-3,
                            Γ0=1.237, A=3000, B=1.8003, R1=10, R2=1,
                            b0=1300, μ=6.2e-4,

                            REACTION='a', Qc=4.48e6, Bc=6.9e10, Ea=11350, Rc=1,

                            MULTI=True, EOS_2='jwl', cv_2=1e-3,
                            Γ0_2=0.3, A_2=209.2, B_2=-5.689, R1_2=4.4, R2_2=1.2,
                            b0_2=1300, μ_2=6.2e-4)


""" Scaled Units

NM_JWL_Scaled = material_params(
        EOS='jwl', ρ0=1, cv=0.0096,
        Γ0=1.237, A=67.69, B=-0.0406, R1=10, R2=1,
        b0=1300/6243.2, μ=6.2e-4/4.43e5,

        REACTION='i', Qc=0.115,
        G1=1100, a=0.667, b=0.667, d=4, λ0=0.99,

        MULTI=True, EOS_2='jwl', cv_2=0.0067,
        Γ0_2=0.3, A_2=4.578, B_2=0.128, R1_2=4.4, R2_2=1.2,
        b0_2=1300/6243.2, μ_2=6.2e-4/4.43e5)

"""
