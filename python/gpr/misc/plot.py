from types import MethodType

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy import nan, nanmax, nanmin, arange, isnan, linspace, mgrid, prod, zeros
from numpy.linalg import norm
from matplotlib.pyplot import colorbar, contour, contourf, figure, get_cmap, \
    imshow, plot, xlim, streamplot, ticklabel_format, xlabel, ylabel

from ader.etc.basis import Basis

from gpr.misc.structures import State
from gpr.multi import get_material_index


def plot1d(y, style, x, lab, col, ylab, xlab='x', sci=1):

    if x is None:
        x = arange(len(y)) + 0.5

    plot(x, y, style, label=lab, color=col, linewidth=1)

    xlim(x[0], x[-1])

    if sci:
        ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    xlabel(xlab)
    ylabel(ylab)
    plt.show()


def plot2d(x, plotType, y=None, vmin=None, vmax=None, lsets=None):

    if plotType == 'colormap':
        im = imshow(x, vmin=vmin, vmax=vmax)
        colorbar(im)

    elif plotType == 'contour':

        if vmin is None:
            vmin = nanmin(x)
        if vmax is None:
            vmax = nanmax(x)

        levels = linspace(vmin, vmax, 25)
        im = contour(x, levels=levels)

        if lsets is not None:
            for lset in lsets:
                contour(lset, levels=[0], colors='black')

        colorbar(im)

    elif plotType == 'contourf':
        im = contourf(x, 25, vmin=vmin, vmax=vmax)
        colorbar(im)

    elif plotType == 'streams':
        nx, ny = x.shape[:2]
        Y, X = mgrid[0:ny, 0:nx]
        streamplot(X, Y, x.T, y.T)


def plot_compound(u, MPs, style, x, lab, col, title, sci, attr, plotType,
                  vmin, vmax, i=None, j=None, takeNorm=False):

    shape = u.shape[:-1]
    n = prod(shape)
    y = zeros(n)
    nmat = len(MPs)

    for ii in range(n):
        Q = u.reshape([n, -1])[ii]
        ind = get_material_index(Q, nmat)
        MP = MPs[ind]

        if MP.EOS > -1:  # not a vacuum
            P = State(Q, MP)

            if isinstance(getattr(P, attr), MethodType):
                var = getattr(P, attr)()
            else:
                var = getattr(P, attr)

            if i is not None:
                if j is None:
                    var = var[i]
                else:
                    var = var[i,j]

            if takeNorm:
                var = norm(var)

            if isnan(var):
                print('Warning: nan in cell', ii)
            else:
                y[ii] = var

        else:
            y[ii] = nan

    if u.ndim - 1 == 1:
        plot1d(y, style, x, lab, col, title, sci=sci)
    else:
        plot2d(y.reshape(shape), plotType, vmin=vmin, vmax=vmax,
               lsets=[u[:, :, -(i+1)] for i in range(nmat-1)])


def plot_density(u, MPs, style='-', x=None, lab=None, col=None, sci=0,
                 square=0, plotType='colormap', vmin=None, vmax=None):

    figure(0, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Density', sci, 'ρ', plotType,
                  vmin, vmax)


def plot_energy(u, MPs, style='-', x=None, lab=None, col=None, sci=0, square=0,
                plotType='colormap', vmin=None, vmax=None):

    figure(1, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Total Energy',
                  sci, 'v', plotType, vmin, vmax)


def plot_velocity(u, MPs, i=0, style='-', x=None, lab=None, col=None, sci=0,
                  square=0, plotType='streams', vmin=None, vmax=None):
    figure(2 + i, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Velocity Component %d' % (i + 1),
                  sci, 'v', plotType, vmin, vmax, i=i)


def plot_distortion(u, MPs, i, j, style='-', x=None, lab=None, col=None, sci=0,
                    fig=None, square=0, plotType='colormap', vmin=None, vmax=None):
    ind = 5 + i * 3 + j
    if fig is None:
        fig = ind
    figure(fig, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Distortion Component %d,%d' % (i + 1, j + 1), sci, 'A',
                  plotType, vmin, vmax, i=i, j=j)


def plot_thermal_impulse(u, MPs, i, style='-', x=None, lab=None, col=None, sci=0,
                         square=0, plotType='colormap', vmin=None, vmax=None):
    figure(14 + i, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Thermal Impulse Component %d' % (i + 1), sci, 'J',
                  plotType, vmin, vmax)


def plot_concentration(u, style='-', x=None, lab=None, col=None, sci=0,
                       square=0):
    figure(18, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Concentration', sci, 'λ',
                  plotType, vmin, vmax)


def plot_pressure(u, MPs, style='-', x=None, lab=None, col=None, sci=0,
                  square=0, plotType='colormap', vmin=None, vmax=None):
    figure(19, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Pressure', sci, 'p', plotType,
                  vmin, vmax)


def plot_temperature(u, MPs, style='-', x=None, lab=None, col=None, sci=0,
                     square=0, plotType='colormap', vmin=None, vmax=None):
    figure(20, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Temperature', sci, 'T',
                  plotType, vmin, vmax)


def plot_sigma(u, MPs, i, j, style='-', x=None, lab=None, col=None, sci=0,
               fig=None, square=0, plotType='colormap', vmin=None, vmax=None, takeNorm=False):

    if takeNorm:
        if fig is None:
            fig = 21
        figure(fig, figsize=fig_size(square))
        plot_compound(u, MPs, style, x, lab, col,
                      'Viscous Stress Norm',
                      sci, 'σ', plotType, vmin, vmax, takeNorm=True)
    else:
        if fig is None:
            fig = 21 + i * 3 + j
        figure(fig, figsize=fig_size(square))
        plot_compound(u, MPs, style, x, lab, col,
                      'Viscous Stress Component %d,%d' % (i + 1, j + 1),
                      sci, 'σ', plotType, vmin, vmax, i=i, j=j)


def plot_Sigma(u, MPs, i, j, style='-', x=None, lab=None, col=None, sci=0,
               fig=None, square=0, plotType='colormap', vmin=None, vmax=None):
    if fig is None:
        fig = 21 + i * 3 + j
    figure(fig, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Total Stress Component %d,%d' % (i + 1, j + 1), sci, 'Σ',
                  plotType, vmin, vmax, i=i, j=j)


def plot_heat_flux(u, MPs, i, style='-', x=None, lab=None, col=None, sci=0,
                   square=0, plotType='colormap', vmin=None, vmax=None):
    figure(30 + i, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Heat Flux Component %d' % (i + 1), sci, 'q', plotType,
                  vmin, vmax, i=i)


def plot_variable(u, var, style='-', x=None, lab=None, col=None, sci=0,
                  square=0):
    figure(34, figsize=fig_size(square))

    NDIM = u.ndim - 1
    if NDIM == 1:
        plot1d(u[:, var], style, x, lab, col, 'Variable %d' %
               (var), xlab='x', sci=sci)

    elif NDIM == 2:
        plot2d(u[:, :, var])


def plot_primitives(u, MPs, style='-', x=None):
    plot_density(u, style=style, x=x)
    plot_velocity(u, 0, style=style, x=x)
    plot_pressure(u, MPs, style=style, x=x)


def plot_interfaces(u, figNum=None, loc=None, col=None):
    # use axvline
    return "Not implemented"


def colors(n):
    cmap = get_cmap('viridis')
    return [cmap.colors[i] for i in linspace(0, 255, n, dtype=int)]


def fig_size(square):
    if square:
        return (10, 10)
    else:
        return (6.4 / 0.72, 4.8 / 0.72)


def plot_weno(wh, var, MPs=None):

    NDIM = int((wh.ndim - 1) / 2)
    shape = wh.shape[:NDIM]
    N = wh.shape[-2]
    NV = wh.shape[-1]

    basis = Basis(N)
    inds = [N * s for s in shape]
    x = zeros(inds)
    u = zeros(inds + [NV])

    for i in range(shape[0]):
        ind = i * N
        for j in range(N):
            x[ind + j] = i + basis.NODES[j]
            u[ind + j] = wh[i, j]

    if var == 'density':
        plot_density(u, x=x)
    if var == 'energy':
        plot_energy(u, x=x)
    if var == 'velocity':
        plot_velocity(u, x=x)
    if var == 'pressure':
        plot_pressure(u, MPs, x=x)


def plot_dg(qh, var, t, MPs=None):
    n, _, _, N, _, NV = qh.shape
    basis = Basis(N)
    wh = zeros([n, 1, 1, N, NV])
    for i in range(n):
        for j in range(N):
            for k in range(N):
                wh[i, 0, 0, j] += basis.ψ[k](t) * qh[i, 0, 0, k, j]

    plot_weno(wh, var, MPs)


def plot_res_ref(res, ref, x=None, reflab='Reference', reslab='Results'):
    cm = colors(3)
    if x is not None:
        plot(x, res, col=cm[1], label=reslab, marker='x', linestyle='none',
             markersize=5)
        plot(x, ref, col=cm[0], label=reflab, linewidth=1)
    else:
        plot(res, col=cm[1], label=reslab, marker='x', linestyle='none',
             markersize=5)
        plot(ref, col=cm[0], label=reflab, linewidth=1)


def anim(data, var):
    fig = figure()

    im = imshow(data[0].grid[:, :, 0, var])

    def animate(i):
        im.set_array(data[i].grid[:, :, 0, var])
        return im,

    return animation.FuncAnimation(fig, animate, interval=200, repeat=True)
