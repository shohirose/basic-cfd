#ifndef CFD_LAX_SCHEME_HPP
#define CFD_LAX_SCHEME_HPP

#include <Eigen/Sparse>
#include <vector>

namespace cfd {

/**
 * @brief Lax scheme
 *
 */
class LaxScheme {
 public:
  /**
   * @brief Compute differential operator
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   *
   * Lax scheme for 1-D advection equation
   * @f[
   * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
   * @f]
   * can be expressed by
   * @f[
   * q_j^{n+1} = \frac{1}{2} (1 - \rho) q_{j+1}^n +
   *             \frac{1}{2} (1 + \rho) q_{j-1}^n
   * @f]
   * where @f$ \rho = c \Delta t / \Delta x @f$.
   */
  static Eigen::SparseMatrix<double> eval(double dt, double dx, double c,
                                          int nx) noexcept {
    using triplet = Eigen::Triplet<double>;
    std::vector<triplet> coeffs;
    const auto a = dt * c / dx;
    const auto a1 = 0.5 * (1 + a);
    const auto a2 = 0.5 * (1 - a);
    coeffs.reserve(2 * nx);
    for (int i = 1; i < nx - 1; ++i) {
      coeffs.emplace_back(i, i - 1, a1);
      coeffs.emplace_back(i, i + 1, a2);
    }
    Eigen::SparseMatrix<double> D(nx, nx);
    D.setFromTriplets(coeffs.begin(), coeffs.end());
    return D;
  }
};

}  // namespace cfd

#endif  // CFD_LAX_SCHEME_HPP