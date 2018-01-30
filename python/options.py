""" Domain Parameters """

Lx = 1                      # Length of domain in x direction
Ly = 1                      # Length of domain in x direction
Lz = 1                      # Length of domain in x direction
nx = 200                    # Number of cells in x direction
ny = 1                      # Number of cells in y direction
nz = 1                      # Number of cells in z direction
nV = 17                     # Number of variables - should be:
#     5+VISCOUS*9+THERMAL*3+MULTI*2+REACTIVE+LSET

""" GFM Options """

LSET = 0                    # Number of level sets
RGFM = 0                    # Use Riemann GFM
ISO_FIX = 0                 # Use isobaric fix

STAR_TOL = 1e-6             # Tolerance to which star states converge
STIFF_RGFM = 1              # Whether to use a stiff solver to find star states

""" Solver Options """

CPP_LVL = 2                 # Level of C++ use (0=none, 1=some, 2=full)
SPLIT = 0                   # Whether or not to use a split solver

NUM_ODE = 0                 # Use numerical ODE solver (SPLIT=1)
HALF_STEP = 1               # Step forwards WENO solver by dt/2 (SPLIT=1)
STRANG = 1                  # Use Strang splitting (SPLIT=1)

N = 3                       # Order of the method
CFL = 0.6                   # CFL number
OSHER = 1                   # Whether to use Osher flux (else Rusanov flux)
PERR_FROB = 0               # Use Perron-Frobenius approximation to max λ

""" DG Options """

HIDALGO = 0                 # Use Hidalgo initial guess
STIFF = 0                   # Use Newton-Krylov
SUPER_STIFF = 0             # Use Newton-Krylov for Hidalgo initial guess

DG_TOL = 1e-6               # Tolerance to which the predictor must converge
MAX_ITER = 50               # Max number of non-stiff iterations attempted

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # WENO coefficient of central stencils
λs = 1                      # WENO coefficient of side stencils
eps = 1e-14                 # Ensures oscillation indicators don't blow up

""" Speed-Up Parameters """

PARA_DG = 0                 # Parallelise DG step
PARA_FV = 0                 # Parallelise FV step
NCORE = 4                   # Number of cores used if running in parallel


""" Derived Values """

ndim = (nx > 1) + (ny > 1) + (nz > 1)
NT = N**(ndim + 1)
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
