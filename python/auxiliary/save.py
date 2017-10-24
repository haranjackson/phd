from codecs import open
from os import makedirs, path
from time import time

from numpy import array, linspace, int64, save, zeros

from options import tf, Lx, Ly, Lz, nx, ny, nz
from options import VISCOUS, THERMAL, REACTIVE, REACTION_TYPE
from options import RGFM, ISO_FIX
from options import USE_CPP, SPLIT
from options import NUM_ODE, HALF_STEP, STRANG
from options import RECONSTRUCT_PRIM, WENO_AVERAGE
from options import  N, CFL, OSHER, PERRON_FROB
from options import HIDALGO, STIFF, SUPER_STIFF, FAIL_LIM, DG_TOL
from options import rc, λc, λs, eps
from options import MAX_ITER, PARA_DG, PARA_FV, NCORE


def print_stats(count, t, dt, interfaceLocations):
    print(count+1)
    print('t  =', t)
    print('dt =', dt)
    if RGFM:
        print('Interfaces =', interfaceLocations)

def make_u(fluids, inds):
    """ Builds u across the domain, from the different fluids grids
    """
    u = zeros(fluids[0].shape)
    for i in range(len(fluids)):
        l = inds[i]
        r = inds[i+1]
        u[l:r] = fluids[i][l:r]
    return u

def save_config(path):
    with open(path, 'w+', encoding='utf-8') as f:
        f.write('tf = %e\n' % tf)
        f.write('Lx = %e\n' % Lx)
        f.write('Ly = %e\n' % Ly)
        f.write('Lz = %e\n' % Lz)
        f.write('nx = %i\n' % nx)
        f.write('ny = %i\n' % ny)
        f.write('nz = %i\n\n' % nz)

        f.write('VISCOUS  = %i\n' % VISCOUS)
        f.write('THERMAL  = %i\n' % THERMAL)
        f.write('REACTIVE = %i\n' % REACTIVE)
        f.write('REACTION_TYPE = %s\n\n' % REACTION_TYPE)

        f.write('RGFM    = %i\n' % RGFM)
        f.write('ISO_FIX = %i\n\n' % ISO_FIX)

        f.write('USE_CPP = %i\n' % USE_CPP)
        f.write('SPLIT   = %i\n\n' % SPLIT)

        f.write('NUM_ODE   = %i\n' % NUM_ODE)
        f.write('HALF_STEP = %i\n' % HALF_STEP)
        f.write('STRANG    = %i\n\n' % STRANG)

        f.write('RECONSTRUCT_PRIM = %i\n' % RECONSTRUCT_PRIM)
        f.write('WENO_AVERAGE     = %i\n\n' % WENO_AVERAGE)

        f.write('N     = %i\n' % N)
        f.write('CFL   = %f\n' % CFL)
        f.write('OSHER = %i\n' % OSHER)
        f.write('PERRON_FROB = %i\n\n' % PERRON_FROB)

        f.write('HIDALGO     = %i\n' % HIDALGO)
        f.write('STIFF       = %i\n' % STIFF)
        f.write('SUPER_STIFF = %i\n' % SUPER_STIFF)
        f.write('FAIL_LIM    = %i\n' % FAIL_LIM)
        f.write('DG_TOL      = %e\n' % DG_TOL)
        f.write('MAX_ITER    = %i\n\n' % MAX_ITER)

        f.write('rc  = %f\n' % rc)
        f.write('λc  = %e\n' % λc)
        f.write('λs  = %e\n' % λs)
        f.write('eps = %e\n\n' % eps)

        f.write('PARA_DG = %i\n' % PARA_DG)
        f.write('PARA_FV = %i\n' % PARA_FV)
        f.write('NCORE   = %i\n\n' % NCORE)

def save_all(data):

    if not path.exists('_dump'):
        makedirs('_dump')

    gridArray = array([datum.grid for datum in data])
    timeArray = array([datum.time for datum in data])
    intArray  = array([datum.int  for datum in data])

    save('_dump/gridArray%d.npy' % time(), gridArray)
    save('_dump/timeArray%d.npy' % time(), timeArray)
    save('_dump/intArray%d.npy'  % time(), intArray)
    save_config('_dump/options%d.txt' % time())

def compress_arrays(saveArrays, N):

    n = len(saveArrays.time)
    inds = linspace(0,n-1,N,dtype=int64)

    return [saveArrays.data[inds],
            saveArrays.time[inds],
            saveArrays.interfaces[inds]]
