Rc   = 8.314459848          # Universal gas constant

""" Domain Parameters """

tf = 0.2                    # Final time of the simulation
L  = 1                      # Length of domain in x direction
nx = 200                    # Number of cells in x direction
ny = 1                      # Number of cells in y direction
nz = 1                      # Number of cells in z direction

""" Model Options """

mechanical = 1              # Whether to include evolution of density and velocity
viscous    = 0              # Whether to include viscosity
thermal    = 0              # Whether to include thermal conduction
reactive   = 0              # Whether to include reactivity

""" Problem Parameters """

Ms = 2                      # Mach number of viscous shock
W  = 1                      # Power input at wall in cook off
doubleTime = 1e-6           # Time taken for temperature at wall to double
fullBurn = 0                # Whether to run simulation until 50% of reactant has burned
burnProp = 0.5              # If fullBurn=1, simulation stops when proportion of cells with reactant
                            # remaining falls below burnProp

""" Reactive System Options """

reactionType = 'a'          # 'a' (Arrhenius) or 'd' (Discrete)
reactiveEOS  = 1            # Use a reactive equation of state (if not, energy term has sources)
minE = 1                    # Whether to set the chemical energy to -Qc when concentration=0

""" GFM Options """

GFM  = 0                    # Whether to use original GFM
RGFM = 1                    # Whether to use RGFM (requires GFM=1)
isoFix = 0                  # Whether to use isobaric fix
SFix   = 0                  # Whether to fix entropy in ghost cells
TFix   = 1                  # Whether to fix temperature in ghost cells (entropyFix must be 0)
UPDATE_STEP = 5             # Number of timesteps used to update interface locations

""" Solver Options """

solver = 'AW'               # Which solver to use ('AW', 'SLIC', or 'NEW')
useDG  = 1                  # Whether to use DG to get higher order time accuracy with ADER-WENO
N      = 1                  # Method is order N+1
CFL    = 0.9                # CFL number
method = 'rusanov'          # Method used for intercell fluxes ('osher' or 'rusanov')
convertTemp    = 1          # Whether to use constant-pressure approximation in cookoff
altThermSolve  = 1          # Whether to use operator splitting solver for the thermal subsystem
NOISE_LIM      = 1e-16      # Values below NOISE_LIM in finite volume solver are discarded

""" DG Options """

hidalgo    = 0              # Whether to use the Hidalgo initial guess
stiff      = 0              # Whether source terms are stiff (Newton-Krylov solver is used)
superStiff = 0              # Whether to use Newton-Krylov to compute the Hidalgo initial guess
failLim    = 180            # Maximum number of non-stiff solves that are allowed to fail
TOL        = 6e-6           # Tolerance to which the Galerkin Predictor must converge
MAX_ITER   = 50             # Maximum number of non-stiff iterations attempted in DG

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # Coefficient of oscillation indicator of central stencil(s)
λs = 1                      # Coefficient of oscillation indicator of side stencils
eps  = 1e-14                # Constant ensuring oscillation indicators don't blow up

""" Speed-Up Parameters """

paraDG = 0                  # Whether to parallelise the DG step
paraFV = 0                  # Whether to parallelise the FV step
ncore  = 4                  # Number of cores to use if running in parallel

""" Debug Options """

NO_WARNING = 1      # Turn off all SciPy/NumPy warnings. Potentially dangerous. Overridden by DEBUG.
DEBUG      = 0      # In debug mode, warnings are given if complex values are encountered, and
                    # exceptions are raised if non-numeric values are encountered


""" Derived Values """

from numpy import array
from auxiliary.classes import active_subsystems
ndim = sum(array([nx, ny, nz]) > 1)
N1 = N+1
NT = N1**(ndim+1)
dx = L / nx
subsystems = active_subsystems(mechanical, viscous, thermal, reactive)

""" Compatibility Settings """

if fullBurn:
    tf = 1e16
if GFM:
    reducedDomain = 0

from numpy import seterr
if DEBUG:
    seterr(all='raise')
elif NO_WARNING:
    seterr(all='ignore')
