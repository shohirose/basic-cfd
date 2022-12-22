#include <filesystem>

#include "advection_equation_solver_1d.hpp"
#include "ftcs_scheme.hpp"
#include "lax_scheme.hpp"
#include "lax_wendroff_scheme.hpp"
#include "upwind_scheme.hpp"

namespace fs = std::filesystem;

using FtcsSolver = cfd::AdvectionEquationSolver1d<cfd::FtcsScheme>;
using LaxSolver = cfd::AdvectionEquationSolver1d<cfd::LaxScheme>;
using LaxWendroffSolver =
    cfd::AdvectionEquationSolver1d<cfd::LaxWendroffScheme>;
using UpwindSolver = cfd::AdvectionEquationSolver1d<cfd::UpwindScheme>;

int main(int argc, char** argv) {
  // Parameters
  double dt = 0.05;     // Time step length
  double dx = 0.1;      // Grid size
  double c = 1;         // Velocity
  int n_grids = 21;     // Number of grids
  int n_timesteps = 6;  // Number of time steps

  // Initial condition
  Eigen::VectorXd q0(n_grids);
  for (int i = 0; i < n_grids / 2; ++i) {
    q0[i] = 1;
  }
  for (int i = n_grids / 2; i < n_grids; ++i) {
    q0[i] = 0;
  }

  // Solve
  {
    FtcsSolver solver(dt, dx, c, n_grids, fs::path("ftcs"));
    solver.solve(n_timesteps, q0);
  }

  {
    LaxSolver solver(dt, dx, c, n_grids, fs::path("lax"));
    solver.solve(n_timesteps, q0);
  }

  {
    LaxWendroffSolver solver(dt, dx, c, n_grids, fs::path("lax-wendroff"));
    solver.solve(n_timesteps, q0);
  }

  {
    UpwindSolver solver(dt, dx, c, n_grids, fs::path("upwind"));
    solver.solve(n_timesteps, q0);
  }

  return EXIT_SUCCESS;
}
