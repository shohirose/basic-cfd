#include <filesystem>
#include <fstream>
#include <string>
#include <fmt/format.h>

namespace fs = std::filesystem;

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

  /**
   * @brief Construct a new Simulator object
   * 
   * @param dt Delta time
   * @param dx Delta x for grids
   * @param c Velocity
   * @param nx Number of grid points
   */
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
   * @param dir Directory to output results
   */
  template <typename T>
  void run(int nt, const Eigen::MatrixBase<T>& q0,
           const fs::path& dir) const noexcept {
    fs::create_directory(dir);
    {
      std::ofstream file(dir / fs::path("q0.txt"));
      file << q0 << std::endl;
    }

    Eigen::VectorXd q_old = q0;
    const auto nx = solver_.nx();
    for (int i = 1; i <= nt; ++i) {
      const Eigen::VectorXd q_new = solver_.solve(q_old);
      q_old = q_new;

      {
        const std::string filename = fmt::format("q{}.txt", i);
        std::ofstream file(dir / fs::path(filename));
        file << q_new << std::endl;
      }
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

  // Solve
  {
    Simulator<FtcsSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0, "ftcs");
  }

  {
    Simulator<LaxSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0, "lax");
  }

  {
    Simulator<LaxWendroffSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0, "lax-wendroff");
  }

  {
    Simulator<UpwindSolver> simulator(dt, dx, c, nx);
    simulator.run(nt, q0, "upwind");
  }

  return EXIT_SUCCESS;
}
