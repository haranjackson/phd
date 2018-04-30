from concurrent.futures import ProcessPoolExecutor
from itertools import product
from time import time

from numpy import sum

from ader.etc.boundaries import standard_BC, periodic_BC

from gpr.opts import NV
from solver import SolverPlus
from solver.gfm.fill import fill_ghost_cells


def get_material_index(Q, m):
    LSET = m - 1
    return sum(Q[NV-LSET:] >= 0)


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

        self.solvers = [SolverPlus(nvar, ndim, F=F, B=B, S=S, model_params=MP,
                                   M=M, max_eig=max_eig, order=order,
                                   ncore=ncore, split=split,
                                   ode_solver=ode_solver,
                                   riemann_solver=riemann_solver)
                        for MP in self.MPs]

    def make_u(self, grids):
        """ Builds u across the domain, from the different material grids
        """
        av = sum(grids, axis=0) / len(grids)

        for coords in product(*[range(s) for s in av.shape[:self.NDIM]]):

            materialIndex = get_material_index(av[coords], self.m)
            self.u[coords] = grids[materialIndex][coords]

    def resume(self):

        with ProcessPoolExecutor(max_workers=self.ncore) as executor:

            dt = 0

            while self.t < self.final_time:

                t0 = time()

                grids, masks = fill_ghost_cells(self.u, self.m, self.N,
                                                self.dX, self.MPs, dt)

                for i in range(self.m):
                    self.solvers[i].u = grids[i]

                dt = min([self.solvers[i].timestep(masks[i])
                          for i in range(self.m)])

                for i in range(self.m):
                    self.solvers[i].stepper(executor, dt, masks[i])

                self.make_u([solver.u for solver in self.solvers])

                self.t += dt
                self.count += 1

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

        self.initialize(initial_grid, final_time, dX, cfl, boundary_conditions,
                        verbose, callback, cpp_level)

        return self.resume()
