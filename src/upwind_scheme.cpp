#include "upwind_scheme.hpp"

#include <cmath>
#include <vector>

namespace cfd {

/**
 * @brief Upwind scheme
 * 
 * Upwind scheme for 1-D advection equation
 * @f[
 * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
 * @f]
 * can be expressed by
 * @f[
 * q_j^{n+1} = \frac{1}{2} (| \rho | - \rho) q_{j+1}^n + 
 *             (1 - | \rho |) q_j^n +
 *             \frac{1}{2} (| \rho | + \rho) q_{j-1}^n
 * @f]
 * where @f$ \rho = c \Delta t / \Delta x @f$.
 */
Eigen::SparseMatrix<double> UpwindScheme::eval(double dt, double dx, double c,
                                               int nx) noexcept {
  using triplet = Eigen::Triplet<double>;
  std::vector<triplet> coeffs;
  const auto a = dt * c / dx;
  const auto abs_a = std::abs(a);
  const auto a1 = 0.5 * (a + std::abs(a));
  const auto a2 = 1 - std::abs(a);
  const auto a3 = -0.5 * (a - std::abs(a));
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
