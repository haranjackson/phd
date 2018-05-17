from concurrent.futures import ProcessPoolExecutor
from itertools import product
from time import time

from numpy import sum

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

        self.NDIM = ndim
        self.N = order
        self.m = len(model_params)
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

    def make_u(self):
        """ Builds u across the domain, from the different material grids
        """
        realGrids = [solver.u for solver in self.solvers
                     if solver.pars.EOS > -1]
        av = sum(realGrids, axis=0) / len(realGrids)

        for coords in product(*[range(s) for s in av.shape[:self.NDIM]]):

            materialIndex = get_material_index(av[coords], self.m)

            if self.solvers[materialIndex].pars.EOS > -1:
                self.u[coords] = self.solvers[materialIndex].u[coords]
            else:
                self.u[coords][-(self.m - 1):] = av[coords][-(self.m - 1):]

    def resume(self):

        with ProcessPoolExecutor(max_workers=self.ncore) as executor:

            dt = 0

            while self.t < self.final_time:

                t0 = time()

                grids, masks = fill_ghost_cells(self.u, self.m, self.N,
                                                self.dX, self.MPs, dt)

                for solver, grid in zip(self.solvers, grids):
                    solver.u = grid

                dt = min([solver.timestep(mask)
                          for solver, mask in zip(self.solvers, masks)
                          if solver.pars.EOS > -1])

                for solver, mask in zip(self.solvers, masks):
                    if solver.pars.EOS > -1:
                        solver.stepper(executor, dt, mask)

                self.make_u()

                self.t += dt
                self.count += 1

                for i in range(self.m):
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

        solver.final_time = self.final_time
        solver.dX = self.dX
        solver.cfl = self.cfl
        solver.cpp_level = self.cpp_level

        solver.verbose = False
        solver.callback = None

        solver.BC = self.BC

    def initialize(self, initial_grid, final_time, dX, cfl=0.9,
                   boundary_conditions='transitive', verbose=False,
                   callback=None, cpp_level=0):

        self.u = initial_grid
        self.t = 0
        self.count = 0

        self.final_time = final_time
        self.dX = dX
        self.cfl = cfl
        self.cpp_level = cpp_level

        self.verbose = verbose
        self.callback = callback

        if boundary_conditions == 'transitive':
            self.BC = standard_BC
        elif boundary_conditions == 'periodic':
            self.BC = periodic_BC
        elif callable(boundary_conditions):
            self.BC = boundary_conditions

        for solver in self.solvers:
            self.initialize_sub_solver(solver)

    def solve(self, initial_grid, final_time, dX, cfl=0.9,
              boundary_conditions='transitive', verbose=False, callback=None,
              cpp_level=0):

        if cpp_level == 2:
            self.u = solve_full_cpp(self, initial_grid, final_time, dX, cfl)
            return self.u

        else:
            self.initialize(initial_grid, final_time, dX, cfl, boundary_conditions,
                        verbose, callback, cpp_level)
            return self.resume()
