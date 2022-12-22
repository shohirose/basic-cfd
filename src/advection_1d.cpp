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
    FtcsSolver solver(dt, dx, c, nx, nt, fs::path("ftcs"));
    solver.solve(q0);
  }

  {
    LaxSolver solver(dt, dx, c, nx, nt, fs::path("lax"));
    solver.solve(q0);
  }

  {
    LaxWendroffSolver solver(dt, dx, c, nx, nt, fs::path("lax-wendroff"));
    solver.solve(q0);
  }

  {
    UpwindSolver solver(dt, dx, c, nx, nt, fs::path("upwind"));
    solver.solve(q0);
  }

  return EXIT_SUCCESS;
}
