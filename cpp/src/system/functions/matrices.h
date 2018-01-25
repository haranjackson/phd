#ifndef MATRICES_H
#define MATRICES_H

#include "../../etc/types.h"
#include "../objects/gpr_objects.h"

Mat2_2 inv2(Mat2_2r X);
Mat3_3 inv3(Mat3_3r X);
Mat3_3 AdevG(Mat3_3r A);
double devGsq(Mat3_3r A);

#endif
