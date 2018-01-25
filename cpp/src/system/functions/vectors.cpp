#include "../../etc/types.h"
#include "../objects/gpr_objects.h"

Vec3Map get_ρv(VecVr Q) {
  // Returns the momentum vector
  return Vec3Map(Q.data() + 2);
}

Mat3_3Map get_A(VecVr Q) {
  // Returns the distortion matrix.
  return Mat3_3Map(Q.data() + 5);
}

Vec3Map get_ρJ(VecVr Q) {
  // Returns the density times the thermal impulse vector
  return Vec3Map(Q.data() + 14);
}
