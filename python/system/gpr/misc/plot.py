from matplotlib.pyplot import figure, plot, scatter, axvline, get_cmap, imshow, colorbar, streamplot
from matplotlib.pyplot import ticklabel_format, xlabel, ylabel, xlim
from numpy import arange, zeros, linspace, mgrid, flipud

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from solvers.basis import quad, basis_polys
from system.gpr.misc.structures import Cvec_to_Pclass
from multi.gfm import interface_inds
from options import Lx, nx, ny, ndim, N1, nV


def plot1d(y, style, x, label, color, ylab, xlab='x', sci=1):

    if x is None:
        x = arange(len(y))+0.5

    if style == 'line':
        plot(x, y, label=label, color=color, linewidth=1)
    elif style == 'scatter':
        scatter(x, y, label=label, color=color)
    elif style == 'cross':
        scatter(x, y, marker='x', s=10, label=label, color=color, linewidth=1)

    xlim(x[0], x[-1])

    if sci:
        ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    xlabel(xlab)
    ylabel(ylab)
    plt.show()

def plot2d(x, style, y=None):
    if style=='colormap':
        im = imshow(y, get_cmap('viridis'))
        colorbar(im)
    elif style=='streams':
        Y, X = mgrid[0:nx, 0:ny]
        streamplot(X, Y, flipud(y), flipud(x))

def plot_density(u, style='line', x=None, label=None, color=None, sci=0):
    figure(0)
    if ndim==1:
        y = u[:, 0, 0, 0]
        plot1d(y, style, x, label, color, 'Density', sci=sci)
    elif ndim==2:
        y = u[:, :, 0, 0]
        plot2d(y, 'colormap')

def plot_energy(u, style='line', x=None, label=None, color=None, sci=0):
    figure(1)
    if ndim==1:
        y = u[:, 0, 0, 1] / u[:, 0, 0, 0]
        plot1d(y, style, x, label, color, 'Total Energy', sci=sci)
    elif ndim==2:
        y = u[:, 0, :, 1] / u[:, 0, :, 0]
        plot2d(y, 'colormap')

def plot_velocity(u, i=0, style='line', x=None, label=None, color=None, sci=0,
                  offset=0, dims=None):
    figure(2+i)
    if dims==None:
        dims = ndim
    if dims==1:
        y = u[:, 0, 0, 2+i] / u[:, 0, 0, 0] + offset
        plot1d(y, style, x, label, color, 'Velocity Component %d' % (i+1),
               sci=sci)
    elif dims==2:
        x = u[:, :, 0, 2] / u[:, :, 0, 0] + offset
        y = u[:, :, 0, 3] / u[:, :, 0, 0] + offset
        plot2d(x, 'streams', y)

def plot_distortion(u, i, j, style='line', x=None, label=None, color=None,
                    sci=0):
    figure(5+i*3+j)
    y = u[:, 0, 0, 5+3*j+i]
    plot1d(y, style, x, label, color, 'Distortion Component %d,%d' % (i+1, j+1),
           sci=sci)

def plot_thermal_impulse(u, i, style='line', x=None, label=None, color=None,
                         sci=0):
    figure(14+i)
    y = u[:, 0, 0, 14+i] / u[:, 0, 0, 0]
    plot1d(y, style, x, label, color, 'Thermal Impulse Component %d' % (i+1),
           sci=sci)

def plot_concentration(u, style='line', x=None, label=None, color=None, sci=0):
    figure(18)
    y = u[:, 0, 0, 17] / u[:, 0, 0, 0]
    plot1d(y, style, x, label, color, 'Concentration', sci=sci)

def plot_pressure(u, PARs, intfLocs=[], style='line', x=None, label=None,
                  color=None, sci=0):
    figure(19)
    n = len(u)
    inds = interface_inds(intfLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = Cvec_to_Pclass(u[l, 0, 0], PARs[k]).p
    plot1d(y, style, x, label, color, 'Pressure', sci=sci)

def plot_temperature(u, PARs, intfLocs=[], style='line', x=None, label=None,
                     color=None, sci=0):
    figure(20)
    n = len(u)
    inds = interface_inds(intfLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = Cvec_to_Pclass(u[l, 0, 0], PARs[k]).T
    plot1d(y, style, x, label, color, 'Temperature', sci=sci)

def plot_sigma(u, i, j, PARs, intfLocs=[], style='line', x=None, label=None,
               color=None, sci=0):

    figure(21+i*3+j)
    n = len(u)
    inds = interface_inds(intfLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = Cvec_to_Pclass(u[l, 0, 0], PARs[k])
            y[l] = P.σ[i, j]
    plot1d(y, style, x, label, color,
           ' Viscous Stress Component %d,%d' % (i+1, j+1), sci=sci)

def plot_heat_flux(u, i, PARs, intfLocs=[], style='line', x=None, label=None,
                   color=None, sci=0):
    figure(30+i)
    n = len(u)
    inds = interface_inds(intfLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = Cvec_to_Pclass(u[l, 0, 0], PARs[k])
            y[l] = P.q[i]
    plot1d(y, style, x, label, color, 'Heat Flux Component %d' % (i+1), sci=sci)

def plot_entropy(u, PARs, intfLocs=[], style='line', x=None, label=None,
                 color=None, sci=0):
    figure(33)
    n = len(u)
    inds = interface_inds(intfLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = Cvec_to_Pclass(u[l, 0, 0], PARs[k])
            y[l] = P.s()
    plot1d(y, style, x, label, color, 'Entropy', sci=sci)

def plot_variable(u, var, style='line', x=None, label=None, color=None, sci=0):

    figure(34)
    y = u[:, 0, 0, var]
    plot1d(y, style, x, label, color, 'Variable %d' % var, sci=sci)

def plot_primitives(u, PARs, intfLocs=[], style='line', x=None):

    plot_density(u, style=style, x=x)
    plot_velocity(u, 0, style=style, x=x)
    plot_pressure(u, PARs, intfLocs=intfLocs, style=style, x=x)

def plot_interfaces(intfLocs, figNum=None, loc=None, color=None):
    if figNum is not None:
        figure(figNum)
    for i in intfLocs:
        if loc=='true':
            axvline(x=i, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        elif loc=='cell':
            ind = int(i*nx/Lx) + 0.5
            axvline(x=ind, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        else:
            axvline(i/Lx, ymin=-1e16, ymax=1e16, linestyle='--', color=color)

def colors(n):
    cmap = get_cmap('viridis')
    return [cmap.colors[i] for i in linspace(0, 255, n, dtype=int)]

def plot_weno(wh, var, PARs=None):
    n = len(wh)
    x = zeros(N1*n)
    u = zeros([N1*n,1,1,nV])
    NODES, _, _ = quad()
    for i in range(n):
        ind = N1*i
        for j in range(N1):
            x[ind+j] = i+NODES[j]
            u[ind+j] = wh[i,0,0,j]

    if var=='density':
        plot_density(u, x=x)
    if var=='energy':
        plot_energy(u, x=x)
    if var=='velocity':
        plot_velocity(u, x=x)
    if var=='pressure':
        plot_pressure(u, PARs, x=x)

def plot_dg(qh, var, t, PARs=None):
    psi, _, _ = basis_polys()
    n = len(qh)
    wh = zeros([n,1,1,N1,nV])
    for i in range(n):
        for j in range(N1):
            for k in range(N1):
                wh[i,0,0,j] += psi[k](t) * qh[i,0,0,k,j]

    plot_weno(wh, var, PARs)

def plot_res_ref(res, ref, x=None, reflab='Reference', reslab='Results'):
    cm = colors(3)
    if x is not None:
        plot(x, res, color=cm[1], label=reslab, marker='x', linestyle='none',
             markersize=5)
        plot(x, ref, color=cm[0], label=reflab, linewidth=1)
    else:
        plot(res, color=cm[1], label=reslab, marker='x', linestyle='none',
             markersize=5)
        plot(ref, color=cm[0], label=reflab, linewidth=1)

def anim(data, var):
    fig = plt.figure()

    im = plt.imshow(data[0][:,:,0,var])

    def animate(i):
        im.set_array(data[i][:,:,0,var])
        return im,

    return animation.FuncAnimation(fig, animate, interval=200, repeat=True)
