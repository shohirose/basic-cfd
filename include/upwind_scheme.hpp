#ifndef CFD_UPWIND_SCHEME_HPP
#define CFD_UPWIND_SCHEME_HPP

#include <Eigen/Sparse>

namespace cfd {

/**
 * @brief Upwind scheme
 * 
 */
class UpwindScheme {
 public:
  /**
   * @brief Compute differential operator
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   */
  static Eigen::SparseMatrix<double> eval(double dt, double dx, double c,
                                          int nx) noexcept;
};

}  // namespace cfd

#endif  // CFD_UPWIND_SCHEME_HPP