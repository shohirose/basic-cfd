#ifndef CFD_HARTEN_YEE_TVD_SCHEME_1D_HPP
#define CFD_HARTEN_YEE_TVD_SCHEME_1D_HPP

#include <Eigen/Core>
#include <cmath>

namespace cfd {

class HartenYeeTvdScheme1d {
 public:
  HartenYeeTvdScheme1d(double dt, double dx, double c, int nx)
      : dt_{dt}, dx_{dx}, c_{c}, nx_{nx}, sigma_{calc_sigma(dt, dx, c)} {}

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& q) const noexcept {
    using Eigen::VectorXd;
    const VectorXd delta = q.tail(nx_ - 1) - q.head(nx_ - 1);
    const VectorXd g = calc_g(delta);
    const VectorXd gamma = calc_gamma(g, delta, sigma_);
    const VectorXd phi = calc_phi(g, delta, gamma, sigma_, c_);

    VectorXd dq = VectorXd::Zero(nx_);
    const auto n = nx_ - 2;
    // Boundary conditions are not implemented yet.
    dq.segment(1, n) =
        -(0.5 * dt_ / dx_) *
        ((c_ * (q.segment(2, n) + q.segment(1, n)) + phi.segment(1, n)) -
         (c_ * (q.segment(1, n) + q.head(n)) + phi.head(n)));
    return dq;
  }

 private:
  static double calc_sigma(double dt, double dx, double c) noexcept {
    return 0.5 * (std::abs(c) - dt / dx * c * c);
  }

  static double minmod(double x, double y) noexcept {
    return (std::copysign(0.5, x) + std::copysign(0.5, y)) *
           std::min(std::abs(x), std::abs(y));
  }

  static Eigen::VectorXd calc_g(const Eigen::VectorXd& delta) noexcept {
    using Eigen::VectorXd;
    VectorXd g = VectorXd::Zero(delta.size() + 1);
    for (int i = 1; i < delta.size(); ++i) {
      g(i) = minmod(delta(i), delta(i - 1));
    }
    return g;
  }

  static Eigen::VectorXd calc_gamma(const Eigen::VectorXd& g,
                                    const Eigen::VectorXd& delta,
                                    double sigma) noexcept {
    const auto n = delta.size();
    return sigma * ((g.segment(1, n) - g.head(n)).array() * delta.array() /
                    (delta.array().square() + 1e-12))
                       .matrix();
  }

  static Eigen::VectorXd calc_phi(const Eigen::VectorXd& g,
                                  const Eigen::VectorXd& delta,
                                  const Eigen::VectorXd& gamma, double sigma,
                                  double c) noexcept {
    const auto n = delta.size();
    return (sigma * (g.head(n) + g.segment(1, n)).array() -
            (c + gamma.array()).abs() * delta.array())
        .matrix();
  }

 private:
  double dt_;
  double dx_;
  double c_;
  int nx_;
  double sigma_;
};

}  // namespace cfd

#endif  // CFD_HARTEN_YEE_TVD_SCHEME_1D_HPP