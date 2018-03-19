from itertools import product

from numpy import arange, array, zeros

from solvers.basis import WGHTS
from options import N, SPLIT, NDIM


TWGHTS = array([1]) if SPLIT else WGHTS
TN = len(TWGHTS)


def weight_list():

    ret = zeros([TN] + [N]*NDIM)
    coordList = [arange(N)] * NDIM

    for t in range(TN):

        wght = TWGHTS[t]

        for coords in product(*coordList):
            for d in range(NDIM):
                wght *= WGHTS[coords[d]]

            ret[(t,) + coords] = wght

    return ret, ret.sum(axis=-1).ravel()


WGHT, WGHT_END = weight_list()
