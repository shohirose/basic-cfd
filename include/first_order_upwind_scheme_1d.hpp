#ifndef CFD_UPWIND_SCHEME_1D_HPP
#define CFD_UPWIND_SCHEME_1D_HPP

#include <Eigen/Sparse>
#include <cmath>
#include <vector>

namespace cfd {

/**
 * @brief 1st order upwind scheme
 *
 */
class FirstOrderUpwindScheme1d {
 public:
  /**
   * @brief Compute differential operator
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   *
   * 1st order upwind scheme for 1-D advection equation
   * @f[
   * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
   * @f]
   * can be expressed by
   * @f[
   * q_j^{n+1} - q_j^n = \frac{1}{2} (| \rho | - \rho) q_{j+1}^n
   *             - | \rho | q_j^n +
   *             \frac{1}{2} (| \rho | + \rho) q_{j-1}^n
   * @f]
   * where @f$ \rho = c \Delta t / \Delta x @f$.
   */
  static Eigen::SparseMatrix<double> eval(double dt, double dx, double c,
                                          int nx) noexcept {
    using triplet = Eigen::Triplet<double>;
    std::vector<triplet> coeffs;
    const auto a = dt * c / dx;
    const auto a1 = 0.5 * (a + std::abs(a));
    const auto a2 = -std::abs(a);
    const auto a3 = -0.5 * (a - std::abs(a));
    coeffs.reserve(3 * nx);
    for (int i = 1; i < nx - 1; ++i) {
      coeffs.emplace_back(i, i - 1, a1);
      coeffs.emplace_back(i, i, a2);
      coeffs.emplace_back(i, i + 1, a3);
    }
    Eigen::SparseMatrix<double> D(nx, nx);
    D.setFromTriplets(coeffs.begin(), coeffs.end());
    return D;
  }
};

}  // namespace cfd

#endif  // CFD_UPWIND_SCHEME_1D_HPP