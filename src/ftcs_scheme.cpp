#include <vector>

#include "ftcs_scheme.hpp"

namespace cfd {

/**
 * @brief FTCS scheme
 * 
 * FTCS scheme for 1-D advection equation
 * @f[
 * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
 * @f]
 * can be expressed by
 * @f[
 * q_j^{n+1} = q_j^n - \Delta t \cdot c \frac{q_{j+1}^n - q_{j-1}^n}{2\Delta x}
 * @f]
 */
Eigen::SparseMatrix<double> FtcsScheme::eval(double dt, double dx, double c,
                                             int nx) noexcept {
  using triplet = Eigen::Triplet<double>;
  std::vector<triplet> coeffs;
  const auto a = dt * c / dx;
  const auto a1 = 0.5 * a;
  const auto a2 = 1.0;
  const auto a3 = -0.5 * a;
  coeffs.reserve(3 * nx);
  for (int i = 1; i < nx - 1; ++i) {
    coeffs.emplace_back(i - 1, i, a1);
    coeffs.emplace_back(i, i, a2);
    coeffs.emplace_back(i + 1, i, a3);
  }
  Eigen::SparseMatrix<double> D(nx, nx);
  D.setFromTriplets(coeffs.begin(), coeffs.end());
  return D;
}

}  // namespace cfd
