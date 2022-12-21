#ifndef CFD_ADVECTION_EQUATION_SOLVER_1D_HPP
#define CFD_ADVECTION_EQUATION_SOLVER_1D_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

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
   * @brief Computes q at the next time step
   *
   * @tparam Derived
   * @param q Variable at the next time step
   * @return Eigen::VectorXd
   */
  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& q) const noexcept {
    Eigen::VectorXd q_new =  D_ * q;
    // Copy boundary values
    q_new.head(1) = q.head(1);
    q_new.tail(1) = q.tail(1);
    return q_new;
  }

  double dt() const noexcept { return dt_; }
  double dx() const noexcept { return dx_; }
  double c() const noexcept { return c_; }
  int nx() const noexcept { return nx_; }

 private:
  double dt_;                      ///> Delta time
  double dx_;                      ///> Distance between neighboring grid points
  double c_;                       ///> Advection velocity
  int nx_;                         ///> Number of grids
  Eigen::SparseMatrix<double> D_;  ///> Differential operator
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP