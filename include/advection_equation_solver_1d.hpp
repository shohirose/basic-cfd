#ifndef CFD_ADVECTION_EQUATION_SOLVER_1D_HPP
#define CFD_ADVECTION_EQUATION_SOLVER_1D_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace cfd {

/**
 * @brief Solver for 1D advection equation.
 *
 * @tparam SpacialDiscretizationPolicy Policy for spacial discretization
 */
template <typename SpacialDiscretizationPolicy>
class AdvectionEquationSolver1d {
 public:
  AdvectionEquationSolver1d() = default;

  /**
   * @brief Construct a new Ftcs Solver 1d object
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   */
  AdvectionEquationSolver1d(double dt, double dx, double c, int nx)
      : dt_{dt},
        dx_{dx},
        c_{c},
        nx_{nx},
        D_{SpacialDiscretizationPolicy::eval(dt, dx, c, nx)} {}

  AdvectionEquationSolver1d(const AdvectionEquationSolver1d&) = default;
  AdvectionEquationSolver1d(AdvectionEquationSolver1d&&) = default;

  AdvectionEquationSolver1d& operator=(const AdvectionEquationSolver1d&) =
      default;
  AdvectionEquationSolver1d& operator=(AdvectionEquationSolver1d&&) = default;

  /**
   * @brief Computes q at time step @f$ n + 1 @f$
   *
   * @tparam L
   * @tparam R
   * @param q Variable at time step @f$ n @f$
   * @return Eigen::DenseBase<L>
   */
  template <typename T>
  Eigen::VectorXd solve(const Eigen::DenseBase<T>& q) const noexcept {
    const Eigen::VectorXd q_new = D_ * q;
    return q_new;
  }

 private:
  double dt_;                      ///> Delta time
  double dx_;                      ///> Distance between neighboring grid points
  double c_;                       ///> Advection velocity
  int nx_;                         ///> Number of grids
  Eigen::SparseMatrix<double> D_;  ///> Differential operator
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP