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
   * @param epsilon Epsilon = 0 for 1st order, Epsilon = 1 for higher order
   * @param kappa Kappa = -1 for 2nd order fully upwind,
   * Kappa = 0 for 2nd order upwind biased,
   * Kappa = 1 for average of two neighbor points,
   * Kappa = 1/3 for 3rd order upwind
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
    using Eigen::VectorXd;
    const VectorXd delta = q.tail(nx_ - 1) - q.head(nx_ - 1);
    const VectorXd delta_p = calc_delta_p(delta, b_);
    const VectorXd delta_m = calc_delta_m(delta, b_);
    const VectorXd qr = calc_qr(q, delta_p, delta_m, epsilon_, kappa_);
    const VectorXd ql = calc_ql(q, delta_p, delta_m, epsilon_, kappa_);

    VectorXd dq = VectorXd::Zero(nx_);
    const auto n = nx_ - 4;
    const auto c1 = c_ - std::abs(c_);
    const auto c2 = c_ + std::abs(c_);

    // Boundary conditions are not implemented yet.
    dq.segment(2, n) = -dt_ / dx_ *
                       (0.5 * (c1 * qr.segment(3, n) + c2 * ql.segment(2, n)) -
                        0.5 * (c1 * qr.segment(2, n) + c2 * ql.segment(1, n)));
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
    using Eigen::VectorXd;
    VectorXd delta_p = VectorXd::Zero(delta.size() + 1);
    for (int i = 1; i < delta.size(); ++i) {
      delta_p(i) = minmod(delta(i), b * delta(i - 1));
    }
    return delta_p;
  }

  static Eigen::VectorXd calc_delta_m(const Eigen::VectorXd& delta,
                                      double b) noexcept {
    using Eigen::VectorXd;
    VectorXd delta_m = VectorXd::Zero(delta.size() + 1);
    for (int i = 1; i < delta.size(); ++i) {
      delta_m(i) = minmod(delta(i - 1), b * delta(i));
    }
    return delta_m;
  }

  template <typename Derived>
  static Eigen::VectorXd calc_qr(const Eigen::MatrixBase<Derived>& q,
                                 const Eigen::VectorXd& delta_p,
                                 const Eigen::VectorXd& delta_m, double epsilon,
                                 double kappa) noexcept {
    return q -
           (0.25 * epsilon) * ((1 - kappa) * delta_p + (1 + kappa) * delta_m);
  }

  template <typename Derived>
  static Eigen::VectorXd calc_ql(const Eigen::MatrixBase<Derived>& q,
                                 const Eigen::VectorXd& delta_p,
                                 const Eigen::VectorXd& delta_m, double epsilon,
                                 double kappa) noexcept {
    return q +
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