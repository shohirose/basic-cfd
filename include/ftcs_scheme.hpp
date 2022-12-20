#ifndef CFD_FTCS_SCHEME_HPP
#define CFD_FTCS_SCHEME_HPP

#include <Eigen/Sparse>

namespace cfd {

/**
 * @brief FTCS scheme
 * 
 */
class FtcsScheme {
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

#endif  // CFD_FTCS_SCHEME_HPP