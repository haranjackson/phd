# GPR-cpp
A C++ implementation of the Godunov-Peshkov-Romenski model of continuum mechanics

### Notes
The following lines have been changed in pybind11/eigen.h:

```c++
#include <Eigen/Core>
#include <Eigen/SparseCore>
```

to

```c++
#include "eigen3/Core"
#include "eigen3/SparseCore"
```

### Future Improvements
* odeint for general source terms
* automatic differentiation for the Newton method
* PNPM
