#ifndef UTILS_H
#define UTILS_H

#include "../etc/globals.h"
#include "../system/objects/gpr_objects.h"

void renorm_distortion(Vecr u, std::vector<Par> &MPs);
void reset_distortion(Vecr u, std::vector<Par> &MPs);
bool contorted(Vecr u, double contorted_tol);

#endif // UTILS_H
