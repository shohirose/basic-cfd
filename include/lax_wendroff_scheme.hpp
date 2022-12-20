#ifndef CFD_LAX_WENDROFF_SCHEME_HPP
#define CFD_LAX_WENDROFF_SCHEME_HPP

#include <Eigen/Sparse>

namespace cfd {

/**
 * @brief Lax-Wendroff scheme
 * 
 */
class LaxWendroffScheme {
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

#endif  // CFD_LAX_WENDROFF_SCHEME_HPP