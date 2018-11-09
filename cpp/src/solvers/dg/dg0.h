#ifndef DG0_H
#define DG0_H

#include <functional>

double stiff_ode_solve(double x0, double dt, std::function<double(double)> f);

#endif // DG0_H
