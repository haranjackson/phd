""" System Parameters """

NDIM = 1                    # Number of dimensions
NV   = 14                   # Number of variables

""" Solver Options """

CPP_LVL = 2                 # Level of C++ use (0=none, 1=some, 2=full)
SPLIT = 0                   # Whether or not to use a split solver

N = 2                       # Order of the method
CFL = 0.6                   # CFL number
FLUX = 0                    # Flux type (0=Rusanov, 1=Roe, 2=Osher)

NUM_ODE   = 0               # Use numerical ODE solver (SPLIT=1)
HALF_STEP = 1               # Step forwards WENO solver by dt/2 (SPLIT=1)
STRANG    = 1               # Use Strang splitting (SPLIT=1)

""" DG Options """

STIFF    = 0                # Use Newton-Krylov to find DG coefficients
STIFF_IG = 0                # Use stiff initial guess to DG coefficients
N_K_IG   = 0                # Use Newton-Krylov for stiff initial guess

DG_TOL = 1e-6               # Tolerance to which the predictor must converge
DG_IT  = 50                 # Max number of non-stiff iterations attempted

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # WENO coefficient of central stencils
λs = 1                      # WENO coefficient of side stencils
eps = 1e-14                 # Ensures oscillation indicators don't blow up

""" Speed-Up Parameters """

PARA_DG = 0                 # Parallelise DG step
PARA_FV = 0                 # Parallelise FV step
NCORE = 4                   # Number of cores used if running in parallel

""" GFM Options """

LSET = 0                    # Number of level sets
RGFM = 0                    # Use Riemann GFM
ISO_FIX = 0                 # Use isobaric fix

STAR_TOL = 1e-6             # Tolerance to which star states converge
STIFF_RGFM = 0              # Whether to use a stiff solver to find star states

NV += LSET
