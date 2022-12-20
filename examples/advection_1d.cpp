#include <iostream>

#include "advection_equation_solver_1d.hpp"
#include "ftcs_scheme.hpp"
#include "lax_scheme.hpp"
#include "lax_wendroff_scheme.hpp"
#include "upwind_scheme.hpp"

using FtcsSolver = cfd::AdvectionEquationSolver1d<cfd::FtcsScheme>;
using LaxSolver = cfd::AdvectionEquationSolver1d<cfd::LaxScheme>;
using LaxWendroffSolver =
    cfd::AdvectionEquationSolver1d<cfd::LaxWendroffScheme>;
using UpwindSolver = cfd::AdvectionEquationSolver1d<cfd::UpwindScheme>;

template <typename Solver>
class Simulator {
 public:
  Simulator() = default;

  Simulator(double dt, double dx, double c, int nx) : solver_{dt, dx, c, nx} {}

  Simulator(const Simulator&) = default;
  Simulator(Simulator&&) = default;

  Simulator& operator=(const Simulator&) = default;
  Simulator& operator=(Simulator&&) = default;

  /**
   * @brief Run a simulation
   *
   * @param nt Number of time steps
   * @param q0 Initial condition
   */
  template <typename T>
  void run(int nt, const Eigen::DenseBase<T>& q0) noexcept {
    std::cout << "[0] " << q0.reshaped().transpose() << std::endl;
    Eigen::VectorXd q_old = q0;
    for (int i = 1; i <= nt; ++i) {
      Eigen::VectorXd q_new = solver_.solve(q_old);
      q_old = q_new;
      std::cout << "[" << i << "] " << q_new.reshaped().transpose()
                << std::endl;
    }
  }

 private:
  Solver solver_;
};

int main(int argc, char** argv) {
  // Parameters
  double dt = 0.05;  // Time step length
  double dx = 0.1;   // Grid size
  double c = 1;      // Velocity
  int nx = 21;       // Number of grids
  int nt = 6;        // Number of time steps

  // Initial condition
  Eigen::VectorXd q0(nx);
  for (int i = 0; i < nx / 2; ++i) {
    q0[i] = 1;
  }
  for (int i = nx / 2; i < nx; ++i) {
    q0[i] = 0;
  }

  std::cout << q0.reshaped().transpose() << std::endl;

  std::cout << "------ FTCS scheme ------" << std::endl;
  {
    Simulator<FtcsSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0);
  }

  std::cout << "------ Lax scheme ------" << std::endl;
  {
    Simulator<LaxSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0);
  }

  std::cout << "------ Lax-Wendroff scheme ------" << std::endl;
  {
    Simulator<LaxWendroffSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0);
  }

  std::cout << "------ Upwind scheme ------" << std::endl;
  {
    Simulator<UpwindSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0);
  }

  return EXIT_SUCCESS;
}
