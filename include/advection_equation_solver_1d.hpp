#ifndef CFD_ADVECTION_EQUATION_SOLVER_1D_HPP
#define CFD_ADVECTION_EQUATION_SOLVER_1D_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace cfd {

/**
 * @brief Solver for 1D advection equation.
 *
 * @tparam Scheme Spacial discretization scheme
 * @tparam Writer Data writer
 * @tparam TimestepChecker Check timesteps to write data
 */
template <typename Scheme, typename Writer, typename TimestepChecker>
class AdvectionEquationSolver1d {
 public:
  AdvectionEquationSolver1d() = default;

  /**
   * @brief Construct a new FTCS solver 1d object
   *
   * @param dt Delta time
   * @param dx Distances between neighboring grid points
   * @param c Velocity
   * @param nx Number of grid points
   * @param nt Number of time steps
   * @param writer Data writer
   */
  AdvectionEquationSolver1d(double dt, double dx, double c, int nx, int nt,
                            const Writer& writer,
                            const TimestepChecker& checker)
      : dt_{dt},
        dx_{dx},
        c_{c},
        nx_{nx},
        nt_{nt},
        writer_{writer},
        checker_{checker},
        scheme_{dt, dx, c, nx} {}

  AdvectionEquationSolver1d(const AdvectionEquationSolver1d&) = default;
  AdvectionEquationSolver1d(AdvectionEquationSolver1d&&) = default;

  AdvectionEquationSolver1d& operator=(const AdvectionEquationSolver1d&) =
      default;
  AdvectionEquationSolver1d& operator=(AdvectionEquationSolver1d&&) = default;

  /**
   * @brief Solve the problem
   *
   * @tparam Derived
   * @param q0 Initial condition
   */
  template <typename Derived>
  void solve(const Eigen::MatrixBase<Derived>& q0) const noexcept {
    if (checker_(0)) {
      writer_.write(q0, 0);
    }

    Eigen::VectorXd q = q0;

    for (int i = 1; i <= nt_; ++i) {
      // Solve the equation
      const Eigen::VectorXd dq = scheme_.solve(q);
      q += dq;

      if (checker_(i)) {
        writer_.write(q, i);
      }
    }
  }

  double dt() const noexcept { return dt_; }
  double dx() const noexcept { return dx_; }
  double c() const noexcept { return c_; }
  int nx() const noexcept { return nx_; }

 private:
  double dt_;                ///> Delta time
  double dx_;                ///> Distance between neighboring grid points
  double c_;                 ///> Advection velocity
  int nx_;                   ///> Number of grids
  int nt_;                   ///> Number of time steps
  Writer writer_;            ///> Data writer
  TimestepChecker checker_;  ///> Checks timesteps to write data
  Scheme scheme_;            ///> Discretization scheme
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP