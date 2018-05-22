#include "etc/globals.h"
#include "solvers/iterator.h"
#include "system/objects/gpr_objects.h"
#include "test/initial_grids.h"
#include "test/params.h"
#include <iostream>

int main() {

  double CFL = 0.6;
  bool PERIODIC = false;
  bool SPLIT = false;
  bool HALF_STEP = true;
  bool STIFF = false;
  int FLUX = 1;
  int nOut = 10;

  /*
  double tf = 0.1;
  iVec nX = heat_conduction_dims();
  aVec dX = heat_conduction_spacing();
  Vec u = heat_conduction_IC();
  std::vector<Par> MPs = {air_params()};
  */

  double tf = 5e-6 / 10;
  iVec nX = aluminium_plate_impact_dims();
  aVec dX = aluminium_plate_impact_spacing();
  Vec u = aluminium_plate_impact_IC();

  std::vector<Par> MPs = {vacuum_params(), aluminium_params(),
                          aluminium_params()};

  std::vector<Vec> ret = iterator(u, tf, nX, dX, CFL, PERIODIC, SPLIT,
                                  HALF_STEP, STIFF, FLUX, MPs, nOut);

  std::cout << "FIN";

  return 0;
}
