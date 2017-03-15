from matplotlib.pyplot import figure, plot, scatter, axvline, get_cmap, imshow, colorbar, streamplot
from matplotlib.pyplot import ticklabel_format, xlabel, ylabel, xlim
from numpy import arange, zeros, linspace, mgrid, flipud

from solvers.basis import quad, basis_polys
from gpr.variables.state import sigma, entropy, heat_flux
from gpr.variables.vectors import primitive
from multi.gfm import interface_indices
from options import Lx, nx, ny, ndim


def plot1d(y, style, x, label, color, xlab, ylab, sci=1):
    if x is None:
        if style == 'line':
            plot(arange(len(y))+0.5, y, label=label, color=color, linewidth=1)
        elif style == 'scatter':
            scatter(arange(len(y))+0.5, y, label=label, color=color)
        elif style == 'cross':
            scatter(arange(len(y))+0.5, y, marker='x', s=10, label=label, color=color, linewidth=1)
    else:
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

def plot_velocity(u, i=0, style='line', x=None, label=None, color=None, xlab='x', sci=0, offset=0,
                  dims=None):
    figure(2+i)
    if dims==None:
        dims = ndim
    if dims==1:
        y = u[:, 0, 0, 2+i] / u[:, 0, 0, 0] + offset
        plot1d(y, style, x, label, color, xlab, 'Velocity Component %d' % (i+1), sci)
    elif dims==2:
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

def plot_pressure(u, materialParams, SYS, intLocs=[], style='line', x=None, label=None, color=None,
                  xlab='x', sci=0):
    figure(19)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = primitive(u[l, 0, 0], materialParams[k], SYS).p
    plot1d(y, style, x, label, color, xlab, 'Pressure', sci)

def plot_temperature(u, materialParams, SYS, intLocs=[], style='line', x=None, label=None,
                     color=None, xlab='x', sci=0):
    figure(20)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = primitive(u[l, 0, 0], materialParams[k], SYS).T
    plot1d(y, style, x, label, color, xlab, 'Temperature', sci)

def plot_sigma(u, i, j, materialParams, SYS, intLocs=[], style='line', x=None, label=None,
               color=None, xlab='x', sci=0):

    figure(21+i*3+j)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = primitive(u[l, 0, 0], materialParams[k], SYS)
            y[l] = sigma(P.ρ, P.A, materialParams[k].cs2)[i, j]
    plot1d(y, style, x, label, color, xlab, ' Viscous Stress Component %d,%d' % (i+1, j+1), sci)

def plot_heat_flux(u, i, materialParams, SYS, intLocs=[], style='line', x=None, label=None,
                   color=None, xlab='x', sci=0):
    figure(30+i)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            P = primitive(u[l, 0, 0], materialParams[k], SYS)
            y[l] = heat_flux(P.T, P.J, materialParams[k].α2)[i]
    plot1d(y, style, x, label, color, xlab, 'Heat Flux Component %d' % (i+1), sci)

def plot_entropy(u, materialParams, SYS, intLocs=[], style='line', x=None, label=None, color=None,
                 xlab='x', sci=0):
    figure(33)
    n = len(u)
    inds = interface_indices(intLocs, n)
    y = zeros(n)

    for k in range(len(inds)-1):
        for l in range(inds[k], inds[k+1]):
            y[l] = entropy(u[l, 0, 0], materialParams[k], SYS)
    plot1d(y, style, x, label, color, xlab, 'Entropy', sci)

def plot_variable(u, var, style='line', x=None, label=None, color=None, xlab='x', sci=0):

    figure(34)
    y = u[:, 0, 0, var]
    plot1d(y, style, x, label, color, xlab, 'Variable %d' % var, sci)

def plot_primitives(u, materialParams, SYS, intLocs=[], style='line', x=None):

    plot_density(u, style=style, x=x)
    plot_velocity(u, 0, style=style, x=x)
    plot_pressure(u, materialParams, SYS, intLocs=intLocs, style=style, x=x)

def plot_interfaces(intLocs, figNum=None, loc=None, color=None):
    if figNum is not None:
        figure(figNum)
    for i in intLocs:
        if loc=='true':
            axvline(x=i, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        elif loc=='cell':
            ind = int(i*nx/Lx) + 0.5
            axvline(x=ind, ymin=-1e16, ymax=1e16, linestyle='--', color=color)
        else:
            axvline(i/Lx*loc, ymin=-1e16, ymax=1e16, linestyle='--', color=color)

def colors(n):
    cmap = get_cmap('viridis')
    return [cmap.colors[i] for i in linspace(0, 255, n, dtype=int)]

def plot_weno(wh, var, gauss_basis=1):
    n = len(wh)
    x = zeros(2*n)
    y = zeros(2*n)
    nodes, _, _ = quad()
    x1,x2 = nodes
    for i in range(n):
        ind = 2*i
        x[ind] = i
        x[ind+1] = i+1
        if gauss_basis:
            y1 = wh[i,0,0,0,var]
            y2 = wh[i,0,0,1,var]
            m = (y2-y1)/(x2-x1)
            y[ind] = y1 - m*x1
            y[ind+1] = y1 + m*(1-x1)
        else:
            y[ind] = wh[i,0,0,0,var]
            y[ind+1] = wh[i,0,0,0,var] + wh[i,0,0,1,var]

    plot(x,y)
    plot(x,y,marker='x')

def plot_dg(qh, var, t=0):
    psi, _, _ = basis_polys()
    n = len(qh)
    x = zeros(2*n)
    y = zeros(2*n)
    for i in range(n):
        ind = 2*i
        x[ind] = i
        x[ind+1] = i+1
        for j in range(2):
            for m in range(2):
                for n in range(2):
                    y[ind+j] += qh[i,0,0,2*m+n,var] * psi[m](t) * psi[n](j)

    plot(x,y)
    plot(x,y,marker='x')

def plot_res_ref(res, ref, x=None, reflab='Reference', reslab='Results'):
    cm = colors(3)
    if x is not None:
        plot(x,res,color=cm[1],label=reslab,marker='x',linestyle='none',markersize=5)
        plot(x,ref,color=cm[0],label=reflab,linewidth=1)
    else:
        plot(res,color=cm[1],label=reslab,marker='x',linestyle='none',markersize=5)
        plot(ref,color=cm[0],label=reflab,linewidth=1)
