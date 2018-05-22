#ifndef INITIAL_GRIDS_H
#define INITIAL_GRIDS_H

#include "../etc/types.h"
#include "../system/objects/gpr_objects.h"

VecV Qvec(double ρ, double p, Vec3r v, Mat3_3r A, Vec3r J, Par &MP);

iVec heat_conduction_dims();

aVec heat_conduction_spacing();

Vec heat_conduction_IC();

iVec aluminium_plate_impact_dims();

aVec aluminium_plate_impact_spacing();

Vec aluminium_plate_impact_IC();

#endif // INITIAL_GRIDS_H
