#ifndef ROTATION_H
#define ROTATION_H

#include "../objects/gpr_objects.h"

Mat3_3 rotation_matrix(Vecr n);
void rotate_tensors(VecVr Q, Mat3_3r R);

#endif // ROTATION_H
