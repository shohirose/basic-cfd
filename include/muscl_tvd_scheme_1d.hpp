#ifndef CFD_MUSCL_TVD_SCHEME_1D_HPP
#define CFD_MUSCL_TVD_SCHEME_1D_HPP

#include <Eigen/Core>
#include <cmath>

namespace cfd {

class MusclTvdScheme1d {
 public:
  /**
   * @brief Construct a new Muscl Tvd Scheme 1d object
   *
   * @param dt Time step length
   * @param dx Grid length
   * @param c Velocity
   * @param nx Number of grids
   * @param epsilon @f$ \epsilon @f$ for MUSCL TVD scheme
   * @param kappa @f$ \kappa @f$ for MUSCL TVD scheme
   */
  MusclTvdScheme1d(double dt, double dx, double c, int nx, double epsilon,
                   double kappa)
      : dt_{dt},
        dx_{dx},
        c_{c},
        nx_{nx},
        epsilon_{epsilon},
        kappa_{kappa},
        b_{calc_b(kappa)} {}

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& q) const noexcept {
    const Eigen::VectorXd delta = q.tail(nx_ - 1) - q.head(nx_ - 1);
    const Eigen::VectorXd delta_p = calc_delta_p(delta, b_);
    const Eigen::VectorXd delta_m = calc_delta_m(delta, b_);
    const Eigen::VectorXd qr = calc_qr(q, delta_p, delta_m, epsilon_, kappa_);
    const Eigen::VectorXd ql = calc_ql(q, delta_p, delta_m, epsilon_, kappa_);

    Eigen::VectorXd dq = Eigen::VectorXd::Zero(nx_);
    for (int i = 2; i < nx_ - 2; ++i) {
      const auto f_p =
          0.5 * (c_ * (qr(i + 1) + ql(i)) - std::abs(c_) * (qr(i + 1) - ql(i)));
      const auto f_m =
          0.5 * (c_ * (qr(i) + ql(i - 1)) - std::abs(c_) * (qr(i) - ql(i - 1)));
      dq(i) = -dt_ / dx_ * (f_p - f_m);
    }
    return dq;
  }

 private:
  static double calc_b(double kappa) noexcept {
    return (3 - kappa) / (1 - kappa);
  }

  static double minmod(double x, double y) noexcept {
    return (std::copysign(0.5, x) + std::copysign(0.5, y)) *
           std::min(std::abs(x), std::abs(y));
  }

  static Eigen::VectorXd calc_delta_p(const Eigen::VectorXd& delta,
                                      double b) noexcept {
    Eigen::VectorXd delta_p(delta.size() - 1);
    for (int i = 0; i < delta_p.size(); ++i) {
      delta_p(i) = minmod(delta(i + 1), b * delta(i));
    }
    return delta_p;
  }

  static Eigen::VectorXd calc_delta_m(const Eigen::VectorXd& delta,
                                      double b) noexcept {
    Eigen::VectorXd delta_m(delta.size() - 1);
    for (int i = 0; i < delta_m.size(); ++i) {
      delta_m(i) = minmod(delta(i), b * delta(i + 1));
    }
    return delta_m;
  }

  template <typename Derived>
  static Eigen::VectorXd calc_qr(const Eigen::MatrixBase<Derived>& q,
                                 const Eigen::VectorXd& delta_p,
                                 const Eigen::VectorXd& delta_m, double epsilon,
                                 double kappa) noexcept {
    return q.segment(1, delta_p.size()) -
           (0.25 * epsilon) * ((1 - kappa) * delta_p + (1 + kappa) * delta_m);
  }

  template <typename Derived>
  static Eigen::VectorXd calc_ql(const Eigen::MatrixBase<Derived>& q,
                                 const Eigen::VectorXd& delta_p,
                                 const Eigen::VectorXd& delta_m, double epsilon,
                                 double kappa) noexcept {
    return q.segment(1, delta_p.size()) +
           (0.25 * epsilon) * ((1 - kappa) * delta_m + (1 + kappa) * delta_p);
  }

  double dt_;
  double dx_;
  double c_;
  int nx_;
  double epsilon_;
  double kappa_;
  double b_;
};

}  // namespace cfd

#endif  // CFD_MUSCL_TVD_SCHEME_1D_HPP