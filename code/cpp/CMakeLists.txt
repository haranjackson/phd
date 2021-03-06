cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 17)
project(GPRcpp)

SET(PYTHON_DIR "~/miniconda3/include/python3.7m")
include_directories(include ${PYTHON_DIR})

SET(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
SET(OpenMP_C_LIB_NAMES "omp")
SET(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
SET(OpenMP_CXX_LIB_NAMES "omp")
SET(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
# -march=native seems slower sometimes
# -ffast-math causes nans in some tests

#SET(MKLROOT "/opt/intel/mkl")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -I${MKLROOT}/include")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")

SET(SRCS include/skfmm/base_marcher.cpp
         include/skfmm/distance_marcher.cpp
         include/skfmm/heap.cpp

         src/etc/debug.cpp
         src/etc/globals.cpp
         src/etc/grid.cpp

         src/multi/fill.cpp
         src/multi/functions.cpp
         src/multi/pfmm.cpp

         src/scipy/legendre.cpp
         src/scipy/lgmres.cpp
         src/scipy/newton_krylov.cpp
         src/scipy/poly.cpp

         src/solvers/iterator.cpp
         src/solvers/steppers.cpp
         src/solvers/utils.cpp

         src/solvers/fv/fluxes.cpp
         src/solvers/fv/fv.cpp

         src/solvers/split/homogeneous.cpp
         src/solvers/split/ode.cpp
         src/solvers/split/numeric.cpp
         src/solvers/split/rk.cpp

         src/solvers/dg/dg.cpp
         src/solvers/dg/dg1.cpp
         src/solvers/dg/dg2.cpp
         src/solvers/dg/dg_matrices.cpp
         src/solvers/dg/initial_guess.cpp

         src/solvers/poly/basis.cpp
         src/solvers/poly/evaluations.cpp

         src/solvers/weno/weno.cpp
         src/solvers/weno/weno_matrices.cpp

         src/system/eig.cpp
         src/system/equations.cpp
         src/system/jacobians.cpp

         src/system/energy/derivatives.cpp
         src/system/energy/eos.cpp
         src/system/energy/mg.cpp
         src/system/energy/multi.cpp

         src/system/functions/matrices.cpp
         src/system/functions/vectors.cpp

         src/system/multi/conditions.cpp
         src/system/multi/eigenvecs.cpp
         src/system/multi/riemann.cpp
         src/system/multi/rotations.cpp

         src/system/relaxation/analytic.cpp
         src/system/relaxation/numerical.cpp

         src/system/variables/sources.cpp
         src/system/variables/state.cpp

         src/system/waves/shear.cpp
         src/system/waves/speeds.cpp

         bindings/bindings.cpp
   )

find_package(OpenMP REQUIRED)
add_subdirectory(include/pybind11-master)
pybind11_add_module(GPRpy ${SRCS})
target_link_libraries(GPRpy PRIVATE OpenMP::OpenMP_CXX)
