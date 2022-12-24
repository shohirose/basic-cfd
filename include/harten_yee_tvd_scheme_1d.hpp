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
    const Eigen::VectorXd delta = q.tail(nx_ - 1) - q.head(nx_ - 1);
    const Eigen::VectorXd g = calc_g(delta);
    const Eigen::VectorXd gamma = calc_gamma(g, delta, sigma_);
    const Eigen::VectorXd phi = calc_phi(g, delta, gamma, sigma_, c_);

    Eigen::VectorXd dq = Eigen::VectorXd::Zero(nx_);
    for (int i = 1; i < nx_ - 1; ++i) {
      const auto f_up = c_ * (q(i + 1) + q(i)) + phi(i);
      const auto f_down = c_ * (q(i) + q(i - 1)) + phi(i - 1);
      dq(i) = -0.5 * dt_ / dx_ * (f_up - f_down);
    }
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
    const auto nx = delta.size() + 1;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(nx);
    for (int i = 1; i < nx - 1; ++i) {
      g(i) = minmod(delta(i), delta(i - 1));
    }
    return g;
  }

  static Eigen::VectorXd calc_gamma(const Eigen::VectorXd& g,
                                    const Eigen::VectorXd& delta,
                                    double sigma) noexcept {
    const auto nx = g.size();
    Eigen::VectorXd gamma(nx - 1);
    for (int i = 0; i < nx - 1; ++i) {
      gamma(i) =
          sigma * (g(i + 1) - g(i)) * delta(i) / (delta(i) * delta(i) + 1e-12);
    }
    return gamma;
  }

  static Eigen::VectorXd calc_phi(const Eigen::VectorXd& g,
                                  const Eigen::VectorXd& delta,
                                  const Eigen::VectorXd& gamma, double sigma,
                                  double c) noexcept {
    const auto nx = g.size();
    Eigen::VectorXd phi(nx - 1);
    for (int i = 0; i < nx - 1; ++i) {
      phi(i) = sigma * (g(i) + g(i + 1)) - std::abs(c + gamma(i)) * delta(i);
    }
    return phi;
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