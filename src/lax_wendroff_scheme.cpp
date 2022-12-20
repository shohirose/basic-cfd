#include "lax_wendroff_scheme.hpp"

#include <vector>

namespace cfd {

/**
 * @brief Lax-Wendroff scheme
 * 
 * Lax-Wendroff scheme for 1-D advection equation
 * @f[
 * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
 * @f]
 * can be expressed by
 * @f[
 * q_j^{n+1} = \frac{1}{2} \rho (\rho - 1) q_{j+1}^n + 
 *             (1 - \rho^2) q_j^n +
 *             \frac{1}{2} \rho (\rho + 1) q_{j-1}^n
 * @f]
 * where @f$ \rho = c \Delta t / \Delta x @f$.
 */
Eigen::SparseMatrix<double> LaxWendroffScheme::eval(double dt, double dx,
                                                    double c, int nx) noexcept {
  using triplet = Eigen::Triplet<double>;
  std::vector<triplet> coeffs;
  const auto a = dt * c / dx;
  const auto a1 = 0.5 * a * (a + 1);
  const auto a2 = 1 - a * a;
  const auto a3 = 0.5 * a * (a - 1);
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