/*

#include <fstream>
#include <utility>

#include "../../include/boost/numeric/odeint.hpp"

#include "../../etc/globals.h"


typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;


struct stiff_system
{
    void operator()( const vector_type & x, vector_type & dxdt, double)
    {
        dxdt[ 0 ] = -101.0 * x[ 0 ] - 100.0 * x[ 1 ];
        dxdt[ 1 ] = x[ 0 ];
    }
};

struct stiff_system_jacobi
{
    void operator()( const vector_type & x, matrix_type & J,
                     const double &, vector_type & dfdt )
    {
        J( 0 , 0 ) = -101.0;
        J( 0 , 1 ) = -100.0;
        J( 1 , 0 ) = 1.0;
        J( 1 , 1 ) = 0.0;

        for (int i=0; i<V; i++)
            dfdt[i] = 0.;
    }
};

void odeint_stepper(vector_type & x0, double dt)
{
    integrate_const(make_dense_output<rosenbrock4<double> >(1e-6, 1e-6),
                    make_pair(stiff_system(), stiff_system_jacobi()),
                    x0, 0., dt, dt/10);
}

void ode_stepper_numeric(VecVr Q, double dt, Par & MP)
{   // Solves the ODE analytically by linearising the distortion equations and
    // providing an analytic approximation to the thermal impulse evolution

    double r = Q(0);
    double E = Q(1) / r;
    Vec3 v = get_v(Q, true);
    Mat3_3 A = get_A(Q);
    Vec3 J = get_J(Q, true);

    Mat3_3 A1 = analyticSolver_distortion(A, dt, MP.tau1);
    set_A(Q, A1);

    Mat3_3 A2 = (A+A1)/2;
    J = analyticSolver_thermal(r, E, A2, J, v, dt, MP);
    set_J(Q, J);
}

*/
