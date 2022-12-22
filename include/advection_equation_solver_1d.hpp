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
 */
template <typename Scheme, typename Writer>
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
                            const Writer& writer)
      : dt_{dt},
        dx_{dx},
        c_{c},
        nx_{nx},
        nt_{nt},
        writer_{writer},
        D_{Scheme::eval(dt, dx, c, nx)} {}

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
    writer_(q0, 0);

    Eigen::VectorXd q_old = q0;
    Eigen::VectorXd q_new(nx_);

    for (int i = 0; i < nt_; ++i) {
      // Solve the equation
      q_new = D_ * q_old;

      // Copy boundary values
      q_new.head(1) = q_old.head(1);
      q_new.tail(1) = q_old.tail(1);

      q_old = q_new;

      writer_(q_new, i + 1);
    }
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
  int nt_;                         ///> Number of time steps
  Writer writer_;                  ///> Data writer
  Eigen::SparseMatrix<double> D_;  ///> Differential operator
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP