from gpr.opts import NV


def get_material_index(Q, m):
    LSET = m - 1
    return sum(Q[NV-LSET:] >= 0)
