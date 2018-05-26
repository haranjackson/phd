import GPRpy

from concurrent.futures import ProcessPoolExecutor
from time import time

from numpy import array, int32, ones, prod, zeros

from ader.etc.boundaries import standard_BC, periodic_BC

from gpr.multi import get_material_index
from solver import SolverPlus
from solver.cpp import solve_full_cpp
from solver.gfm.fill import fill_ghost_cells


flux_types = {'rusanov': 0,
              'roe': 1,
              'osher': 2}


class MultiSolver():

    def __init__(self, nvar, ndim, F, B=None, S=None, model_params=[],
                 M=None, max_eig=None, order=2, ncore=4,
                 riemann_solver='rusanov', stiff_dg=False,
                 stiff_dg_guess=False, newton_dg_guess=False, split=False,
                 half_step=True, ode_solver=None):

        self.nvar = nvar
        self.NDIM = ndim
        self.N = order
        self.nmat = len(model_params)
        self.MPs = model_params
        self.ncore = ncore

        self.split = split
        self.half_step = half_step
        self.stiff_dg = stiff_dg
        self.flux_type = flux_types[riemann_solver]
        self.pars = model_params

        self.solvers = [SolverPlus(nvar, ndim, F=F, B=B, S=S, model_params=MP,
                                   M=M, max_eig=max_eig, order=order,
                                   ncore=ncore, split=split,
                                   ode_solver=ode_solver,
                                   riemann_solver=riemann_solver)
                        for MP in self.MPs]

    def make_u(self, grids, masks):
        """ Builds u across the domain, from the different material grids
        """
        ur = self.u.reshape([self.ncell, self.nvar])
        gridsr = [grid.reshape([self.ncell, self.nvar]) for grid in grids]
        masksr = [mask.ravel() for mask in masks]
        for i in range(self.ncell):
            matSum = zeros(self.nlset)
            matCnt = 0
            for mat in range(self.nmat):
                if masksr[mat][i]:
                    matSum += gridsr[mat][i, -self.nlset:]
                    matCnt += 1
            if matCnt > 0:
                ur[i, -self.nlset:] = matSum / matCnt

            mi = get_material_index(ur[i], self.nmat)
            if self.solvers[mi].pars.EOS > -1:
                ur[i][:-self.nlset] = gridsr[mi][i][:-self.nlset]
            else:
                ur[i][:-self.nlset] = 0

    def resume(self):

        with ProcessPoolExecutor(max_workers=self.ncore) as executor:

            dt = 0
            grids = [self.u.copy() for mat in range(self.nmat)]
            masks = [ones(self.u.shape[:-1], dtype=bool)
                     for mat in range(self.nmat)]

            while self.t < self.tf:

                t0 = time()

                if self.nmat > 1:
                    fill_ghost_cells(grids, masks, self.u, self.nmat, self.N,
                                     self.dX, self.MPs, dt)
                    """
                    nX = array(self.u.shape[:-1], dtype=int32)
                    grids = GPRpy.VectorVec([grid.ravel() for grid in grids])
                    masks = GPRpy.VectorbVec([mask.ravel() for mask in masks])
                    GPRpy.multi.fill_ghost_cells(grids, masks, self.u.ravel(),
                                                 nX, self.dX, dt, self.MPs)
                    grids = [grid.reshape(self.u.shape) for grid in grids]
                    masks = [mask.reshape(self.u.shape[:-1]) for mask in masks]
                    """

                for solver, grid in zip(self.solvers, grids):
                    solver.u = grid

                dt = min([solver.timestep(mask)
                          for solver, mask in zip(self.solvers, masks)
                          if solver.pars.EOS > -1])

                for solver, mask in zip(self.solvers, masks):
                    if solver.pars.EOS > -1:
                        solver.stepper(executor, dt, mask)

                if self.nmat > 1:
                    self.make_u(grids, masks)
                else:
                    self.u = grids[0].copy()

                self.t += dt
                self.count += 1

                for i in range(self.nmat):
                    self.solvers[i].count = self.count

                if self.callback is not None:
                    self.callback(self.u, self.t, self.count)

                if self.verbose:
                    print('\nIteration:', self.count)
                    print('t  = {:.3e}'.format(self.t))
                    print('dt = {:.3e}'.format(dt))
                    print('Iteration Time = {:.3f}s'.format(time() - t0))

        return self.u

    def initialize_sub_solver(self, solver):

        solver.u = self.u
        solver.t = self.t
        solver.count = self.count

        solver.tf = self.tf
        solver.dX = self.dX
        solver.cfl = self.cfl
        solver.cpp_level = self.cpp_level

        solver.verbose = False
        solver.callback = None

        solver.BC = self.BC

    def initialize(self, u0, tf, dX, cfl=0.9, bcs='transitive',
                   verbose=False, callback=None, cpp_level=0):

        self.u = u0.copy()
        self.t = 0
        self.count = 0

        self.tf = tf
        self.dX = dX
        self.cfl = cfl
        self.cpp_level = cpp_level

        self.verbose = verbose
        self.callback = callback

        if bcs == 'transitive':
            self.BC = standard_BC
        elif bcs == 'periodic':
            self.BC = periodic_BC
        elif callable(bcs):
            self.BC = bcs

        for solver in self.solvers:
            self.initialize_sub_solver(solver)

        self.ncell = prod(self.u.shape[:-1])
        self.nlset = self.nmat - 1

    def solve(self, u0, tf, dX, cfl=0.9, bcs='transitive', verbose=False,
              callback=None, cpp_level=0, nOut=50):

        if cpp_level == 2:
            self.u = solve_full_cpp(self, u0, tf, dX, cfl, nOut, callback)
            return self.u

        else:
            self.initialize(u0, tf, dX, cfl, bcs, verbose, callback, cpp_level)
            return self.resume()
