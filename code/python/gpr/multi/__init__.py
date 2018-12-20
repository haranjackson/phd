import GPRpy


def get_material_index(Q, m):
    LSET = m - 1
    return sum(Q[GPRpy.options.NV()-LSET:] >= 0)
