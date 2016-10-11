from matplotlib.pyplot import figure, plot, scatter, axvline, get_cmap, imshow, colorbar, streamplot
from matplotlib.pyplot import ticklabel_format, xlabel, ylabel, xlim, gca
from numpy import arange, zeros, linspace, mgrid, flipud

from gpr.functions import primitive
from gpr.variables import sigma, entropy, heat_flux
from multi.gfm import interface_indices
from options import L, nx, ny, ndim


def plot1d(y, style, x, label, color, xlab, ylab, sci=1):
    if x is None:
        if style == 'line':
            plot(arange(len(y))+0.5, y, label=label, color=color)
        elif style == 'scatter':
            scatter(arange(len(y))+0.5, y, label=label, color=color)
        elif style == 'cross':
            scatter(arange(len(y))+0.5, y, marker='x', label=label, color=color)
    else:
        if style == 'line':
            plot(x, y, label=label, color=color)
        elif style == 'scatter':
            scatter(x, y, label=label, color=color)
        elif style == 'cross':
            scatter(x, y, marker='x', label=label, color=color)
        xlim(x[0], x[-1])

    if sci:
        ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    xlabel(xlab)
    ylabel(ylab)

def plot2d(x, style, y=None):
    if style=='colormap':
        im = imshow(y, get_cmap('viridis'))
        colorbar(im)
    elif style=='streams':
        Y, X = mgrid[0:nx, 0:ny]
        streamplot(X, Y, flipud(y), flipud(x))
     #   gca().invert_yaxis()

def plot_density(u, style='line', x=None, label=None, color=None, xlab='x', sci=0):
    figure(0)
    if ndim==1:
        y = u[:, 0, 0, 0]
        plot1d(y, style, x, label, color, xlab, 'Density', sci)
    elif ndim==2:
        y = u[:, :, 0, 0]
        plot2d(y, 'colormap')

def plot_energy(u, style='line', x=None, label=None, color=None, xlab='x', sci=0):
    figure(1)
    if ndim==1:
        y = u[:, 0, 0, 1] / u[:, 0, 0, 0]
        plot1d(y, style, x, label, color, xlab, 'Total Energy', sci)
    elif ndim==2:
        y = u[:, :, 0, 1] / u[:, :, 0, 0]
        plot2d(y, 'colormap')

def plot_velocity(u, i=0, style='line', x=None, label=None, color=None, xlab='x', sci=0, offset=0):
    figure(2+i)
    if ndim==1:
        y = u[:, 0, 0, 2+i] / u[:, 0, 0, 0] + offset
        plot1d(y, style, x, label, color, xlab, 'Velocity Component %d' % (i+1), sci)
    elif ndim==2:
        x = u[:, :, 0, 2] / u[:, :, 0, 0] + offset
        y = u[:, :, 0, 3] / u[:, :, 0, 0] + offset
        plot2d(x, 'streams', y)

def plot_distortion(u, i, j, style='line', x=None, label=None, color=None, xlab='x', sci=0):
    figure(5+i*3+j)
    y = u[:, 0, 0, 5+3*j+i]
    plot1d(y, style, x, label, color, xlab, 'Distortion Component %d,%d' % (i+1, j+1), sci)

def plot_thermal_impulse(u, i, style='line', x=None, label=None, color=None, xlab='x', sci=0):
    figure(14+i)
    y = u[:, 0, 0, 14+i] / u[:, 0, 0, 0]
    plot1d(y, style, x, label, color, xlab, 'Thermal Impulse Component %d' % (i+1), sci)

def plot_concentration(u, style='line', x=None, label=None, color=None, xlab='x', sci=0):
    figure(18)
    y = u[:, 0, 0, 17] / u[:, 0, 0, 0]
    plot1d(y, style, x, label, color, xlab, 'Concentration', sci)

def plot_pressure(u, materialParams, subsystems, intLocs=[], style='line', x=None, label=None,
                  color=None, xlab='x', sci=0):
    figure(19)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = primitive(u[l, 0, 0], materialParams[k], subsystems).p
    plot1d(y, style, x, label, color, xlab, 'Pressure', sci)

def plot_temperature(u, materialParams, subsystems, intLocs=[], style='line', x=None, label=None,
                     color=None, xlab='x', sci=0):
    figure(20)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = primitive(u[l, 0, 0], materialParams[k], subsystems).T
    plot1d(y, style, x, label, color, xlab, 'Temperature', sci)

def plot_sigma(u, i, j, materialParams, subsystems, intLocs=[], style='line', x=None, label=None,
               color=None, xlab='x', sci=0):

    figure(21+i*3+j)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = primitive(u[l, 0, 0], materialParams[k], subsystems)
            y[l] = sigma(P.r, P.A, materialParams[k].cs2)[i, j]
    plot1d(y, style, x, label, color, xlab, ' Viscous Stress Component %d,%d' % (i+1, j+1), sci)

def plot_heat_flux(u, i, materialParams, subsystems, intLocs=[], style='line', x=None, label=None,
                   color=None, xlab='x', sci=0):
    figure(30+i)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = primitive(u[l, 0, 0], materialParams[k], subsystems)
            y[l] = heat_flux(P.T, P.J, materialParams[k].α2)[i]
    plot1d(y, style, x, label, color, xlab, 'Heat Flux Component %d' % (i+1), sci)

def plot_entropy(u, materialParams, subsystems, intLocs=[], style='line', x=None, label=None,
                 color=None, xlab='x', sci=0):
    figure(33)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = entropy(u[l, 0, 0], materialParams[k], subsystems)
    plot1d(y, style, x, label, color, xlab, 'Entropy', sci)

def plot_variable(u, var, style='line', x=None, label=None, color=None, xlab='x', sci=0):

    figure(34)
    y = u[:, 0, 0, var]
    plot1d(y, style, x, label, color, xlab, 'Variable %d' % var, sci)

def plot_primitives(u, materialParams, subsystems, intLocs=[], style='line', x=None):

    plot_density(u, style=style, x=x)
    plot_velocity(u, 0, style=style, x=x)
    plot_pressure(u, materialParams, subsystems, intLocs=intLocs, style=style, x=x)

def plot_interfaces(intLocs, figNum=None, loc=None, color=None):
    if figNum is not None:
        figure(figNum)
    for i in intLocs:
        if loc=='true':
            axvline(x=i, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        elif loc=='cell':
            ind = int(i*nx/L) + 0.5
            axvline(x=ind, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        else:
            axvline(i/L*loc, ymin=-1e16, ymax=1e16, linestyle='--', color=color)

def colors(n):
    cmap = get_cmap('viridis')
    return [cmap.colors[i] for i in linspace(0, 255, n, dtype=int)]
