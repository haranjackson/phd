from numpy import arange, concatenate, flip, int64, ones, zeros

from options import N


def flat_index(coords):

    if len(coords) == 0:
        return 0
    elif len(coords) == 1:
        return coords[0]
    else:
        return N * flat_index(coords[:-1]) + coords[-1]


def extend_grid(arr, ext, d, kind):
    """ Extends the arr by ext cells on each surface in direction d.
        kind=0: transmissive, kind=1: no-slip, kind=2: periodic
    """
    n = arr.shape[d]

    if kind == 0:
        reps = concatenate((zeros(ext),
                            arange(n),
                            (n - 1) * ones(ext))).astype(int64)
    elif kind == 1:
        reps = concatenate((flip(arange(ext), 0),
                            arange(n),
                            flip(arange(n - ext, n), 0))).astype(int64)

    elif kind == 2:
        reps = concatenate((arange(n - ext, n),
                            arange(n),
                            arange(ext))).astype(int64)

    return arr.take(reps, axis=d)
