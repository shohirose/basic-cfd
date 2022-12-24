#ifndef CFD_FTCS_SCHEME_1D_HPP
#define CFD_FTCS_SCHEME_1D_HPP

#include <Eigen/Sparse>
#include <vector>

namespace cfd {

/**
 * @brief FTCS scheme
 *
 */
class FtcsScheme1d {
 public:
  /**
   * @brief Construct a new Ftcs Scheme 1d object
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   */
  FtcsScheme1d(double dt, double dx, double c, int nx)
      : D_{eval(dt, dx, c, nx)} {}

  /**
   * @brief Compute differential operator
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   *
   * FTCS scheme for 1-D advection equation
   * @f[
   * \frac{\partial q}{\partial t} + c \frac{\partial q}{\partial x} = 0
   * @f]
   * can be expressed by
   * @f[
   * q_j^{n+1} - q_j^n = \frac{1}{2} \rho q_{j-1}^n - \frac{1}{2} \rho q_{j+1}^n
   * @f]
   * where @f$ \rho = c \Delta t / \Delta x @f$.
   */
  static Eigen::SparseMatrix<double> eval(double dt, double dx, double c,
                                          int nx) noexcept {
    using triplet = Eigen::Triplet<double>;
    std::vector<triplet> coeffs;
    const auto a = dt * c / dx;
    const auto a1 = 0.5 * a;
    const auto a2 = -0.5 * a;
    coeffs.reserve(2 * nx);
    for (int i = 1; i < nx - 1; ++i) {
      coeffs.emplace_back(i, i - 1, a1);
      coeffs.emplace_back(i, i + 1, a2);
    }
    Eigen::SparseMatrix<double> D(nx, nx);
    D.setFromTriplets(coeffs.begin(), coeffs.end());
    return D;
  }

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& q) const noexcept {
    return D_ * q;
  }

 private:
  Eigen::SparseMatrix<double> D_;
};

}  // namespace cfd

#endif  // CFD_FTCS_SCHEME_1D_HPP