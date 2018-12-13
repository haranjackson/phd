from os import mkdir
from time import time

from numpy import array, int32, isnan, save


def save_results(uList, T, N, cfl, SPLIT, FLUX):
    try:
        mkdir('results')
    except:
        pass
    pars = [str(x) for x in [T.__name__, N, cfl, SPLIT, FLUX, int(time())]]
    name = '_'.join(pars)
    save('results/' + name + '.npy', uList)
