from gpr.misc.objects import material_params, hyperelastic_params


# TODO: convert all tests from CGS to SI


""" Hyperelastic """

Cu_HYP_SI = hyperelastic_params(ρ0=8930, α=1, β=3, γ=2, cv=390,
                                T0=300, b0=2141, c0=4651)

Al_HYP_CGS = hyperelastic_params(ρ0=2.71, α=1, β=3.577, γ=2.088, cv=9e-4,
                                 T0=300, b0=3.16, c0=6.22)

Cu_HYP_CGS = hyperelastic_params(ρ0=8.9, α=1, β=3, γ=2, cv=4e-4,
                                 T0=300, b0=2.1, c0=4.6)


""" Other """

VAC = material_params(EOS='vac', ρ0=0, cv=0, p0=0)
