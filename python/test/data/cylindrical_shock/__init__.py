from test.data.tools import from_csv


def get_data(var, dim):

    return from_csv('./test/data/cylindrical_shock/' + var + str(dim) + '.csv')


ρx1, ρ1 = get_data('ρ', 1)
ρx2, ρ2 = get_data('ρ', 2)
ux1, u1 = get_data('u', 1)
ux2, u2 = get_data('u', 2)
Σx1, Σ1 = get_data('Σ', 1)
Σx2, Σ2 = get_data('Σ', 2)
Tx1, T1 = get_data('T', 1)
Tx2, T2 = get_data('T', 2)

ρ1 *= 1000
ρ2 *= 1000
u1 *= 1000
u2 *= 1000
Σ1 *= -1e9
Σ2 *= -1e9


figure(figsize=(6.4 / 0.72, 4.8 / 0.72))
plot(ρx2, ρ2, 'x', label='2D Barton et al.')
plot(ρx1, ρ1, label='1D Radially Symmetric')
legend()
