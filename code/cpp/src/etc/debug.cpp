#include <sstream>
#include <string>

#include "pybind11/pybind11.h"

#include "types.h"

void print(Mat M) {
  std::ostringstream oss;

  if (M.cols() == 1)
    M.transposeInPlace();
  else
    oss << "[";

  for (int i = 0; i < M.rows(); i++) {
    oss << "[";
    for (int j = 0; j < M.cols(); j++) {
      if (j > 0)
        oss << ",\t";
      oss << M(i, j);
    }
    oss << "]\n";
    if (i < M.rows() - 1)
      oss << ",";
  }

  if (M.rows() > 1)
    oss << "]";

  pybind11::print(oss.str());
}

void print(double x) { pybind11::print(x); }

void print(std::string x) { pybind11::print(x); }
