from codecs import open
from os import makedirs, path
from time import time

from numpy import array, linspace, int64, save

from options import RGFM, ISO_FIX, STAR_TOL, STIFF_RGFM
from options import CPP_LVL, SPLIT
from options import N, CFL, FLUX
from options import NUM_ODE, HALF_STEP, STRANG
from options import STIFF, STIFF_IG, N_K_IG, DG_TOL, DG_IT
from options import rc, λc, λs, eps
from options import PARA_DG, PARA_FV, NCORE


def print_stats(count, t, dt):
    print(count + 1)
    print('t  =', t)
    print('dt =', dt)


def save_config(path):
    with open(path, 'w+', encoding='utf-8') as f:

        f.write('RGFM     = %i\n' % RGFM)
        f.write('ISO_FIX  = %i\n' % ISO_FIX)
        f.write('STAR_TOL = %f\n' % STAR_TOL)
        f.write('STIFF_RGFM = %i\n\n' % STIFF_RGFM)

        f.write('CPP_LVL = %i\n' % CPP_LVL)
        f.write('SPLIT   = %i\n\n' % SPLIT)

        f.write('N    = %i\n' % N)
        f.write('CFL  = %f\n' % CFL)
        f.write('FLUX = %i\n\n' % FLUX)

        f.write('NUM_ODE   = %i\n' % NUM_ODE)
        f.write('HALF_STEP = %i\n' % HALF_STEP)
        f.write('STRANG    = %i\n\n' % STRANG)

        f.write('STIFF    = %i\n' % STIFF)
        f.write('STIFF_IG = %i\n' % STIFF_IG)
        f.write('N_K_IG   = %i\n' % N_K_IG)

        f.write('DG_TOL = %e\n' % DG_TOL)
        f.write('DG_IT  = %i\n\n' % DG_IT)

        f.write('rc  = %f\n' % rc)
        f.write('λc  = %e\n' % λc)
        f.write('λs  = %e\n' % λs)
        f.write('eps = %e\n\n' % eps)

        f.write('PARA_DG = %i\n' % PARA_DG)
        f.write('PARA_FV = %i\n' % PARA_FV)
        f.write('NCORE = %i\n\n' % NCORE)


def save_all(data):

    if not path.exists('_dump'):
        makedirs('_dump')

    gridArray = array([datum.grid for datum in data])
    timeArray = array([datum.time for datum in data])

    save('_dump/grid%d.npy' % time(), gridArray)
    save('_dump/time%d.npy' % time(), timeArray)
    save_config('_dump/opts%d.txt' % time())


def compress_data(data, N):
    n = len(data.time)
    inds = linspace(0, n - 1, N, dtype=int64)
    return [data.grid[inds], data.time[inds]]


class Data():
    """ An object to hold the arrays in which simulation data are saved
    """

    def __init__(self, u, t):
        self.grid = u.copy()
        self.time = t
