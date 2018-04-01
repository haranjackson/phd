from models.gpr.tests.one import solids
from solvers.solver_class import Solver
from system import flux, nonconservative_matrix, source, system_matrix, max_eig


u, MPs, tf, dX = solids.elastic1_IC()

S = Solver(17, 1, flux, nonconservative_matrix, source, system_matrix, max_eig,
           model_params=MPs[0], ncore=2)

ret = S.solve(u, tf, dX, cfl=0.6, verbose=True)
