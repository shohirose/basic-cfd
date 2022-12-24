#include "advection_equation_solver_1d.hpp"
#include "file_writer.hpp"
#include "first_order_upwind_scheme_1d.hpp"
#include "ftcs_scheme_1d.hpp"
#include "harten_yee_tvd_scheme_1d.hpp"
#include "lax_scheme_1d.hpp"
#include "lax_wendroff_scheme_1d.hpp"
#include "muscl_tvd_scheme_1d.hpp"
#include "second_order_upwind_scheme_1d.hpp"

namespace fs = std::filesystem;

struct EvenTimestepChecker {
  bool operator()(int i) const noexcept { return i % 2 == 0; }
};

using FtcsSolver =
    cfd::AdvectionEquationSolver1d<cfd::FtcsScheme1d, cfd::FileWriter,
                                   EvenTimestepChecker>;

using LaxSolver =
    cfd::AdvectionEquationSolver1d<cfd::LaxScheme1d, cfd::FileWriter,
                                   EvenTimestepChecker>;

using LaxWendroffSolver =
    cfd::AdvectionEquationSolver1d<cfd::LaxWendroffScheme1d, cfd::FileWriter,
                                   EvenTimestepChecker>;

using UpwindSolver1 =
    cfd::AdvectionEquationSolver1d<cfd::FirstOrderUpwindScheme1d,
                                   cfd::FileWriter, EvenTimestepChecker>;

using UpwindSolver2 =
    cfd::AdvectionEquationSolver1d<cfd::SecondOrderUpwindScheme1d,
                                   cfd::FileWriter, EvenTimestepChecker>;

using HYTvdSolver =
    cfd::AdvectionEquationSolver1d<cfd::HartenYeeTvdScheme1d, cfd::FileWriter,
                                   EvenTimestepChecker>;

using MusclTvdSolver =
    cfd::AdvectionEquationSolver1d<cfd::MusclTvdScheme1d, cfd::FileWriter,
                                   EvenTimestepChecker>;

int main(int argc, char** argv) {
  // Parameters
  double dt = 0.05;  // Time step length
  double dx = 0.1;   // Grid size
  double c = 1;      // Velocity
  int nx = 21;       // Number of grids
  int nt = 6;        // Number of time steps

  // Initial condition
  Eigen::VectorXd q0(nx);
  for (int i = 0; i <= nx / 2; ++i) {
    q0[i] = 1;
  }
  for (int i = nx / 2 + 1; i < nx; ++i) {
    q0[i] = 0;
  }

  // Solve
  {
    FtcsSolver solver(cfd::FileWriter("q", "results/FTCS"),
                      EvenTimestepChecker(), cfd::FtcsScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    LaxSolver solver(cfd::FileWriter("q", "results/Lax"), EvenTimestepChecker(),
                     cfd::LaxScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    LaxWendroffSolver solver(cfd::FileWriter("q", "results/Lax-Wendroff"),
                             EvenTimestepChecker(),
                             cfd::LaxWendroffScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    UpwindSolver1 solver(cfd::FileWriter("q", "results/Upwind1"),
                         EvenTimestepChecker(),
                         cfd::FirstOrderUpwindScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    UpwindSolver2 solver(cfd::FileWriter("q", "results/Upwind2"),
                         EvenTimestepChecker(),
                         cfd::SecondOrderUpwindScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    HYTvdSolver solver(cfd::FileWriter("q", "results/TVD"),
                       EvenTimestepChecker(),
                       cfd::HartenYeeTvdScheme1d(dt, dx, c, nx));
    solver.solve(q0, nt);
  }

  {
    MusclTvdSolver solver(cfd::FileWriter("q", "results/MUSCL"),
                          EvenTimestepChecker(),
                          cfd::MusclTvdScheme1d(dt, dx, c, nx, 1, 1.0 / 3.0));
    solver.solve(q0, nt);
  }

  return EXIT_SUCCESS;
}
