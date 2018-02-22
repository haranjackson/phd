from numpy import concatenate, int64, ones

from options import NDIM


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension + 1], ones(n - 2),
                        [extension + 1])).astype(int64)
    return inarray.repeat(reps, axis=d)


def standard_BC(u, reflect=0):
    ret = extend(u, 1, 0)
    if reflect:
        ret[0, :, :, 2:5] *= -1
        ret[0, :, :, 14:17] *= -1
        ret[-1, :, :, 2:5] *= -1
        ret[-1, :, :, 14:17] *= -1
    if NDIM > 1:
        ret = extend(ret, 1, 1)
        if reflect:
            ret[:, 0, :, 2:5] *= -1
            ret[:, 0, :, 14:17] *= -1
            ret[:, -1, :, 2:5] *= -1
            ret[:, -1, :, 14:17] *= -1
    if NDIM > 2:
        ret = extend(ret, 1, 2)
        if reflect:
            ret[:, :, 0, 2:5] *= -1
            ret[:, :, 0, 14:17] *= -1
            ret[:, :, -1, 2:5] *= -1
            ret[:, :, -1, 14:17] *= -1
    return ret


def periodic_BC(u):
    ret = extend(u, 1, 0)
    ret[0] = ret[-2]
    ret[-1] = ret[1]
    if NDIM > 1:
        ret = extend(ret, 1, 1)
        ret[:, 0] = ret[:, -2]
        ret[:, -1] = ret[:, 1]
    if NDIM > 2:
        ret = extend(ret, 1, 2)
        ret[:, :, 0] = ret[:, :, -2]
        ret[:, :, -1] = ret[:, :, 1]
    return ret
