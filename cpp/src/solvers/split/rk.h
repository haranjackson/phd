#ifndef RK_H
#define RK_H

#include <functional>

double runge_kutta(std::function<double(double)> f, double tf, double y,
                   int N_STEP);

#endif // RK_H
