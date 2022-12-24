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
   * @param writer Data writer
   * @param checker Time step checker
   * @param scheme Discretization scheme
   */
  AdvectionEquationSolver1d(const Writer& writer,
                            const TimestepChecker& checker,
                            const Scheme& scheme)
      : writer_{writer}, checker_{checker}, scheme_{scheme} {}

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
   * @param nt Number of time steps
   */
  template <typename Derived>
  void solve(const Eigen::MatrixBase<Derived>& q0, int nt) const noexcept {
    if (checker_(0)) {
      writer_.write(q0, 0);
    }

    Eigen::VectorXd q = q0;

    for (int i = 1; i <= nt; ++i) {
      // Solve the equation
      const Eigen::VectorXd dq = scheme_.solve(q);
      q += dq;

      if (checker_(i)) {
        writer_.write(q, i);
      }
    }
  }

 private:
  Writer writer_;            ///> Data writer
  TimestepChecker checker_;  ///> Checks timesteps to write data
  Scheme scheme_;            ///> Discretization scheme
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP