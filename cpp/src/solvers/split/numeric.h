#ifndef NUMERIC_H
#define NUMERIC_H

#include <functional>

double stiff_ode_solve(double x0, double dt, std::function<double(double)> f);

#endif // NUMERIC_H
