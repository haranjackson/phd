#ifndef MATRICES_H
#define MATRICES_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"


double tr(Matr X);
double det(Matr X);
Mat2_2 inv2(Mat2_2r X);
Mat3_3 inv3(Mat3_3r X);
double L2_1D(Vec3r x);
double L2_2D(Mat3_3r X);
double dot(Vec3r u, Vec3r v);
Mat3_3 gram(Mat3_3r A);
Mat3_3 devG(Mat3_3r A);
Mat3_3 AdevG(Mat3_3r A);
Mat3_3 GdevG(Mat3_3r G);
Mat3_3 outer(Vec3r x, Vec3r y);


#endif
