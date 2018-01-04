clang++ bindings.cpp -std=c++11 -Wall -O3 -shared -fPIC \
-o ../../python/GPRpy.so \
-I ../include/pybind11   \
-I ../include/eigen3     \
-I ../include            \
-I /home/hari/miniconda3/include/python3.6m   \
-I etc              ../src/etc/*.cpp               \
-I scipy            ../src/scipy/*.cpp             \
-I solvers          ../src/solvers/*.cpp           \
-I solvers/weno     ../src/solvers/weno/*.cpp      \
-I solvers/dg       ../src/solvers/dg/*.cpp        \
-I solvers/split    ../src/solvers/split/*.cpp     \
-I solvers/fv       ../src/solvers/fv/*.cpp        \
-I system           ../src/system/*.cpp            \
-I system/functions ../src/system/functions/*.cpp  \
-I system/objects                                  \
-I system/variables ../src/system/variables/*.cpp  \
-march=native \
-DNDEBUG \
#-m64 -I/opt/intel/mkl/include \                 \
#-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--#end-group -lgomp -lpthread -lm -ldl \
