import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy import arange, linspace, mgrid, prod, zeros
from matplotlib.pyplot import colorbar, figure, get_cmap, imshow, plot, xlim
from matplotlib.pyplot import streamplot, ticklabel_format, xlabel, ylabel

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


def plot2d(x, style, y=None):

    if style == 'colormap':
        im = imshow(x, get_cmap('viridis'))
        colorbar(im)

    elif style == 'streams':
        nx, ny = x.shape[:2]
        Y, X = mgrid[0:ny, 0:nx]
        streamplot(X, Y, x.T, y.T)


def plot_simple(u, style, x, lab, col, title, sci, ind, divρ):

    NDIM = len(u.shape) - 1

    if NDIM == 1:
        y = u[:, ind]
        if divρ:
            y /= u[:, 0]
        plot1d(y, style, x, lab, col, title, sci=sci)
    else:
        y = u[:, :, ind]
        if divρ:
            y /= u[:, :, 0]
        plot2d(y, 'colormap')


def plot_compound(u, MPs, style, x, lab, col, title, sci, attr, i=None,
                  j=None):
    NDIM = u.ndim - 1
    n = prod(u.shape[:-1])
    NV = u.shape[-1]
    y = zeros(n)

    for ii in range(n):
        Q = u.reshape([n, NV])[ii]
        ind = get_material_index(Q, len(MPs))
        MP = MPs[ind]
        P = State(Q, MP)
        var = getattr(P, attr)()
        if j is None:
            if i is None:
                y[ii] = var
            else:
                y[ii] = var[i]
        else:
            y[ii] = var[i, j]

    if NDIM == 1:
        plot1d(y, style, x, lab, col, title, sci=sci)
    else:
        plot2d(y, 'colormap')


def plot_density(u, style='-', x=None, lab=None, col=None, sci=0, square=0):

    figure(0, figsize=fig_size(square))
    plot_simple(u, style, x, lab, col, 'Density', sci, 0, False)


def plot_energy(u, style='-', x=None, lab=None, col=None, sci=0, square=0):

    figure(1, figsize=fig_size(square))
    plot_simple(u, style, x, lab, col, 'Total Energy', sci, 1, True)


def plot_velocity(u, i=0, style='-', x=None, lab=None, col=None, sci=0,
                  square=0):
    figure(2 + i, figsize=fig_size(square))
    NDIM = len(u.shape) - 1

    if NDIM == 1:
        y = u[:, 2 + i] / u[:, 0]
        plot1d(y, style, x, lab, col, 'Velocity Component %d' % (i + 1),
               sci=sci)
    else:
        x = u[:, :, 2] / u[:, :, 0]
        y = u[:, :, 3] / u[:, :, 0]
        plot2d(x, 'streams', y)


def plot_distortion(u, i, j, style='-', x=None, lab=None, col=None, sci=0,
                    fig=None, square=0):
    ind = 5 + i * 3 + j
    if fig is None:
        fig = ind
    figure(fig, figsize=fig_size(square))

    plot_simple(u, style, x, lab, col,
                'Distortion Component %d,%d' % (i + 1, j + 1), sci, ind, False)


def plot_thermal_impulse(u, i, style='-', x=None, lab=None, col=None, sci=0,
                         square=0):
    figure(14 + i, figsize=fig_size(square))
    plot_simple(u, style, x, lab, col,
                'Thermal Impulse Component %d' % (i + 1), sci, 14 + i, True)


def plot_concentration(u, style='-', x=None, lab=None, col=None, sci=0,
                       square=0):
    figure(18, figsize=fig_size(square))
    plot_simple(u, style, x, lab, col, 'Concentration', sci, 17, True)


def plot_pressure(u, MPs, style='-', x=None, lab=None, col=None, sci=0,
                  square=0):
    figure(19, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Pressure', sci, 'p')


def plot_temperature(u, MPs, style='-', x=None, lab=None, col=None, sci=0,
                     square=0):
    figure(20, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col, 'Temperature', sci, 'T')


def plot_sigma(u, i, j, MPs, style='-', x=None, lab=None, col=None, sci=0,
               fig=None, square=0):
    if fig is None:
        fig = 21 + i * 3 + j
    figure(fig, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Viscous Stress Component %d,%d' % (i + 1, j + 1), sci, 'σ',
                  i=i, j=j)


def plot_heat_flux(u, i, MPs, style='-', x=None, lab=None, col=None, sci=0,
                   square=0):
    figure(30 + i, figsize=fig_size(square))
    plot_compound(u, MPs, style, x, lab, col,
                  'Heat Flux Component %d' % (i + 1), sci, 'q', i=i)


def plot_variable(u, var, style='-', x=None, lab=None, col=None, sci=0,
                  square=0):
    figure(34, figsize=fig_size(square))
    plot_simple(u, style, x, lab, col, 'Variable %d' % (var), sci, var, False)


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
