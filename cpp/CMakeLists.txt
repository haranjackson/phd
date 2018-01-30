project(GPRcpp)
SET(BINDINGS_NAME "GPRpy")
cmake_minimum_required(VERSION 2.8)

SET(PYTHON_DIR "/home/hari/miniconda3/include/python3.6m")

include_directories(include/pybind11
                    include/eigen3
                    include
                    ${PYTHON_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -fPIC")

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -DNDEBUG")

#SET(MKLROOT /opt/intel/mkl)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -I${MKLROOT}/include")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")

SET(SRCS src/etc/globals.cpp
         src/etc/grid.cpp

         src/scipy/legendre.cpp
         src/scipy/lgmres.cpp
         src/scipy/newton_krylov.cpp
         src/scipy/poly.cpp

         src/solvers/basis.cpp
         src/solvers/evaluations.cpp
         src/solvers/iterator.cpp
         src/solvers/steppers.cpp

         src/solvers/weno/weno.cpp
         src/solvers/weno/weno_matrices.cpp

         src/solvers/fv/fluxes.cpp
         src/solvers/fv/fv.cpp

         src/solvers/split/homogeneous.cpp
         src/solvers/split/ode.cpp
         src/solvers/split/ode_analytic.cpp

         src/solvers/dg/dg.cpp
         src/solvers/dg/dg_matrices.cpp

         src/system/eig.cpp
         src/system/equations.cpp
         src/system/jacobians.cpp

         src/system/functions/matrices.cpp
         src/system/functions/vectors.cpp

         src/system/variables/derivatives.cpp
         src/system/variables/eos.cpp
         src/system/variables/mg.cpp
         src/system/variables/shear.cpp
         src/system/variables/sources.cpp
         src/system/variables/state.cpp
         src/system/variables/wavespeeds.cpp
   )

#set(SRCS_EXEC src/main.cpp ${SRCS})
#add_executable(${PROJECT_NAME} ${SRCS_EXEC})

set(SRCS_SO bindings/bindings.cpp src/etc/debug.cpp ${SRCS})
add_library(${BINDINGS_NAME} SHARED ${SRCS_SO})
set_target_properties(${BINDINGS_NAME} PROPERTIES PREFIX "")
