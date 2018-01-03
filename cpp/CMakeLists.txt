project(GPRcpp)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -g")

SET(SRC_LIST src/main.cpp

             src/etc/globals.cpp
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

             src/system/functions/matrices.cpp
             src/system/functions/vectors.cpp

             src/system/variables/eos.cpp
             src/system/variables/materials.cpp
             src/system/variables/mg.cpp
             src/system/variables/state.cpp
   )

add_executable(${PROJECT_NAME} ${SRC_LIST})
