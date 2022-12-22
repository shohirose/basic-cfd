#ifndef CFD_ADVECTION_EQUATION_SOLVER_1D_HPP
#define CFD_ADVECTION_EQUATION_SOLVER_1D_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <filesystem>

namespace cfd {

/**
 * @brief Solver for 1D advection equation.
 *
 * @tparam Scheme Spacial discretization scheme
 */
template <typename Scheme>
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
   * @param dir Directory to output results
   */
  AdvectionEquationSolver1d(double dt, double dx, double c, int nx,
                            const std::filesystem::path& dir)
      : dt_{dt},
        dx_{dx},
        c_{c},
        nx_{nx},
        dir_{dir},
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
   * @param n_timesteps Number of time steps
   * @param q0 Initial condition
   */
  template <typename Derived>
  void solve(int n_timesteps,
             const Eigen::MatrixBase<Derived>& q0) const noexcept {
    print(q0, 0);

    Eigen::VectorXd q_old = q0;
    Eigen::VectorXd q_new(nx_);

    for (int i = 0; i < n_timesteps; ++i) {
      // Solve the equation
      q_new = D_ * q_old;

      // Copy boundary values
      q_new.head(1) = q_old.head(1);
      q_new.tail(1) = q_old.tail(1);

      q_old = q_new;

      print(q_new, i + 1);
    }
  }

  double dt() const noexcept { return dt_; }
  double dx() const noexcept { return dx_; }
  double c() const noexcept { return c_; }
  int nx() const noexcept { return nx_; }

 private:
  template <typename Derived>
  void print(const Eigen::MatrixBase<Derived>& q, int i) const noexcept {
    const std::string filename = fmt::format("q{}.txt", i);
    std::ofstream file(dir_ / fs::path(filename));
    file << q << std::endl;
  }

  double dt_;                      ///> Delta time
  double dx_;                      ///> Distance between neighboring grid points
  double c_;                       ///> Advection velocity
  int nx_;                         ///> Number of grids
  fs::path dir_;                   ///> Directory to output results
  Eigen::SparseMatrix<double> D_;  ///> Differential operator
};

}  // namespace cfd

#endif  // CFD_ADVECTION_EQUATION_SOLVER_1D_HPP