from joblib import Parallel
from numpy import abs, amax
from numpy.linalg import eigvals
from tangent import autodiff


def make_system_matrix(flux, nonconservative_matrix, model_params):

    dFdQ = autodiff(flux)

    def system_matrix(Q, d, model_params=None):
        ret = dFdQ(Q, d, model_params)
        if nonconservative_matrix is not None:
            ret += nonconservative_matrix(Q, d, model_params)
        return ret

    return system_matrix


def make_max_eig(system_matrix):

    def max_eig(Q, d, model_params=None):
        M = system_matrix(Q, d, model_params=None)
        return amax(abs(eigvals(M)))

    return max_eig


class Solver():

    def __init__(self, initial_grid, flux,
                 nonconservative_matrix=None, source=None, system_matrix=None,
                 max_eig=None, model_params=None,
                 boundaries='transmissive',
                 cfl_number=0.9, order=2, flux_type='rusanov', parallel=False,
                 stiff_dg=False, stiff_dg_initial_guess=False,
                 newtonkrylov_dg_initial_guess=False,
                 DG_TOL=1e-6, DG_IT=50,
                 WENO_r=8, WENO_λc=1e5, WENO_λs = 1, WENO_ε = 1e-14):

        self.initial_grid = initial_grid
        self.current_grid = initial_grid
        self.NDIM = initial_grid.ndim - 1
        self.NV = initial_grid.shape[-1]
        self.pool = Parallel(n_jobs=-1)

        self.flux = flux
        self.nonconservative_matrix = nonconservative_matrix
        self.source = source
        self.model_params = model_params

        if system_matrix is None:
            self.system_matrix = make_system_matrix(self.flux,
                                                    self.nonconservative_matrix,
                                                    self.model_params)
        else:
            self.system_matrix = system_matrix

        if max_eig is None:
            self.max_eig = make_max_eig(self.system_matrix)
        else:
            self.max_eig = max_eig


