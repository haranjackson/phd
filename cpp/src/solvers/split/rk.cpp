#include <cmath>
#include <functional>

#include "../../etc/debug.h"

auto rk4(std::function<double(double)> f)
{
    return [f](double y, double dt) -> double {
        return [y, dt, f](double dy1) -> double {
            return [y, dt, f, dy1](double dy2) -> double {
                return [y, dt, f, dy1, dy2](double dy3) -> double {
                    return [f, dy1, dy2, dy3](double dy4) -> double {
                        return (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6;
                    }(dt * f(y + dy3));
                }(dt * f(y + dy2 / 2));
            }(dt * f(y + dy1 / 2));
        }(dt * f(y));
    };
}

double runge_kutta(std::function<double(double)> f, double tf, double y,
                   int N_STEP)
{
    auto dy = rk4(f);

    double dt = tf / N_STEP;

    for (int i = 0; i < N_STEP; i++)
        y += dy(y, dt);

    return y;
}

double runge_kutta_launcher(std::function<double(double)> f, double tf,
                            double y, int N_STEP)
{

    double y0 = runge_kutta(f, tf, y, N_STEP);

    while (std::isnan(y0))
    {

        N_STEP *= 2;
        y0 = runge_kutta(f, tf, y, N_STEP);

        if (N_STEP > 64)
            break;
    }
    return y0;
}