from os import makedirs, path
from time import time

from numpy import array, concatenate, expand_dims, linspace, int64, save, zeros

from options import tf, L, nx, ny, nz
from options import mechanical, viscous, thermal, reactive
from options import Ms, W, doubleTime, fullBurn, burnProp
from options import reactionType, reactiveEOS, minE
from options import GFM, RGFM, isoFix, SFix, TFix, UPDATE_STEP
from options import useDG, N, CFL, method, renormaliseRho, convertTemp, NOISE_LIM
from options import hidalgo, stiff, superStiff, failLim, TOL
from options import rc, λc, λs, eps
from options import MAX_ITER, minParaDGLen, minParaFVLen, ncore, reducedDomain


def print_stats(count, t, dt, interfaceLocations, subsystems):
    print(count+1)
    print('t  =', t)
    print('dt =', dt)
    print('Interfaces =', interfaceLocations)
    print('M,V,T,R =', subsystems.mechanical, subsystems.viscous, subsystems.thermal,
          subsystems.reactive)

def record_data(fluids, inds, t, interfaceLocations, saveArrays):
    """ Appends the latest data and timestep to the recording arrays
    """
    u = zeros(fluids[0].shape)
    for i in range(len(fluids)):
        l = inds[i]
        r = inds[i+1]
        u[l:r] = fluids[i][l:r]
    saveArrays.data = concatenate([saveArrays.data, expand_dims(u, axis=0)])
    saveArrays.time = concatenate([saveArrays.time, array([t])])
    saveArrays.interfaces = concatenate([saveArrays.interfaces,
                                         expand_dims(array(interfaceLocations), axis=0)])

def save_config(path):
    with open(path, 'w+') as f:
        f.write('tf = %e\n' % tf)
        f.write('L  = %e\n' % L)
        f.write('nx = %i\n' % nx)
        f.write('ny = %i\n' % ny)
        f.write('nz = %i\n\n' % nz)

        f.write('mechanical = %i\n' % mechanical)
        f.write('viscous    = %i\n' % viscous)
        f.write('thermal    = %i\n' % thermal)
        f.write('reactive   = %i\n\n' % reactive)

        f.write('Ms = %f\n' % Ms)
        f.write('W  = %e\n' % W)
        f.write('doubleTime = %e\n' % doubleTime)
        f.write('fullBurn = %i\n' % fullBurn)
        f.write('burnProp = %f\n\n' % burnProp)

        f.write('reactionType = %s\n' % reactionType)
        f.write('reactiveEOS  = %i\n' % reactiveEOS)
        f.write('minE = %i\n\n' % minE)

        f.write('GFM    = %i\n' % GFM)
        f.write('RGFM   = %i\n' % RGFM)
        f.write('isoFix = %i\n' % isoFix)
        f.write('SFix   = %i\n' % SFix)
        f.write('TFix   = %i\n' % TFix)
        f.write('UPDATE_STEP = %i\n\n' % UPDATE_STEP)

        f.write('useDG  = %i\n' % useDG)
        f.write('N      = %i\n' % N)
        f.write('CFL    = %f\n' % CFL)
        f.write('method = %s\n' % method)
        f.write('renormaliseRho = %i\n' % renormaliseRho)
        f.write('convertTemp    = %i\n' % convertTemp)
        f.write('NOISE_LIM = %e\n\n' % NOISE_LIM)

        f.write('hidalgo    = %i\n' % hidalgo)
        f.write('stiff      = %i\n' % stiff)
        f.write('superStiff = %i\n' % superStiff)
        f.write('failLim    = %i\n' % failLim)
        f.write('TOL        = %e\n' % TOL)
        f.write('MAX_ITER   = %i\n\n' % MAX_ITER)

        f.write('rc   = %f\n' % rc)
        f.write('λc   = %e\n' % λc)
        f.write('λs   = %e\n' % λs)
        f.write('eps  = %e\n\n' % eps)

        f.write('minParaDGLen = %i\n' % minParaDGLen)
        f.write('minParaFVLen = %i\n' % minParaFVLen)
        f.write('ncore = %i\n' % ncore)
        f.write('reducedDomain = %i\n\n' % reducedDomain)

def save_all(saveArrays):
    if not path.exists('_dump'):
        makedirs('_dump')
    save('_dump/dataArray%d.npy' % time(), saveArrays.data)
    save('_dump/timeArray%d.npy' % time(), saveArrays.time)
    save('_dump/interArray%d.npy' % time(), saveArrays.interfaces)
    save_config('_dump/options%d.txt' % time())

def compress_arrays(saveArrays, N):
    n = len(saveArrays.time)
    inds = linspace(0,n-1,N,dtype=int64)
    return [saveArrays.data[inds], saveArrays.time[inds], saveArrays.interfaces[inds]]
