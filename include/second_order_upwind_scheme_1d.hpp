#ifndef CFD_SECOND_ORDER_UPWIND_SCHEME_1D_HPP
#define CFD_SECOND_ORDER_UPWIND_SCHEME_1D_HPP

#include <Eigen/Core>
#include <cmath>
#include <vector>

namespace cfd {

/**
 * @brief 2nd order upwind scheme
 *
 */
class SecondOrderUpwindScheme1d {
 public:
  /**
   * @brief Compute differential operator
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   *
   * 2nd order upwind scheme for 1-D advection equation
   * @f[
   * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
   * @f]
   * can be expressed by
   * @f[
   * q_j^{n+1} - q_j^n = \frac{1}{4} (\rho - | \rho |) q_{j+2}^n
   *    - (\rho - | \rho |) q_{j+1}^n - \frac{3}{2} | \rho | q_j^n
   *    + (\rho + | \rho |) q_{j-1}^n - \frac{1}{4} (\rho + | \rho |) q_{j-2}^n
   * @f]
   * where @f$ \rho = c \Delta t / \Delta x @f$.
   */
  static Eigen::SparseMatrix<double> eval(double dt, double dx, double c,
                                          int nx) noexcept {
    using triplet = Eigen::Triplet<double>;
    std::vector<triplet> coeffs;
    const auto a = dt * c / dx;
    const auto a1 = -0.25 * (a + std::abs(a));
    const auto a2 = a + std::abs(a);
    const auto a3 = -1.5 * std::abs(a);
    const auto a4 = -(a - std::abs(a));
    const auto a5 = 0.25 * (a - std::abs(a));
    coeffs.reserve(5 * nx);
    for (int i = 2; i < nx - 2; ++i) {
      coeffs.emplace_back(i, i - 2, a1);
      coeffs.emplace_back(i, i - 1, a2);
      coeffs.emplace_back(i, i, a3);
      coeffs.emplace_back(i, i + 1, a4);
      coeffs.emplace_back(i, i + 2, a5);
    }
    Eigen::SparseMatrix<double> D(nx, nx);
    D.setFromTriplets(coeffs.begin(), coeffs.end());
    return D;
  }
};

}  // namespace cfd

#endif  // CFD_SECOND_ORDER_UPWIND_SCHEME_1D_HPP