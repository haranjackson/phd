#ifndef UTILS_H
#define UTILS_H

#include "../etc/globals.h"

void renorm_distortion(Vecr u, std::vector<Par> &MPs);
void reset_distortion(Vecr u, std::vector<Par> &MPs);
bool contorted(Vecr u);

#endif // UTILS_H
