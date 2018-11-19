#include <cmath>
#include <functional>

auto rk4(std::function<double(double)> f)
{
    return [f](double y, double dt) -> double {
        return [y, dt, f](double dy1) -> double {
            return [y, dt, f, dy1](double dy2) -> double {
                return [y, dt, f, dy1, dy2](double dy3) -> double {
                    return [y, dt, f, dy1, dy2, dy3](double dy4) -> double {
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

    double t = 0.;

    double dt = tf / N_STEP;

    for (int i = 0; i < N_STEP; i++)
        y += dy(y, dt);

    return y;
}