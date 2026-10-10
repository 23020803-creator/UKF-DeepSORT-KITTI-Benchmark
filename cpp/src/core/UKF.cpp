#include "core/UKF.hpp"
#include <cmath>
#include <algorithm>

namespace ukf_deepsort {

TrackUKF::TrackUKF(double fps) {
    dt_ = (fps > 0.0) ? (1.0 / fps) : 0.1;
    init_sigma_weights();
}

void TrackUKF::init_sigma_weights() {
    double n = static_cast<double>(DIM_X);
    lambda_ = alpha_ * alpha_ * (n + kappa_) - n;
    double c = n + lambda_;

    Wm_(0) = lambda_ / c;
    Wc_(0) = lambda_ / c + (1.0 - alpha_ * alpha_ + beta_);

    for (int i = 1; i < NUM_SIGMAS; ++i) {
        Wm_(i) = 0.5 / c;
        Wc_(i) = 0.5 / c;
    }
}

TrackUKF::StateCovariance TrackUKF::enforce_spd(const StateCovariance& matrix, double min_eig) {
    StateCovariance sym = (matrix + matrix.transpose()) * 0.5;
    Eigen::SelfAdjointEigenSolver<StateCovariance> es(sym);
    if (es.info() == Eigen::Success) {
        Eigen::Matrix<double, DIM_X, 1> ev = es.eigenvalues().cwiseMax(min_eig);
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
    return sym + StateCovariance::Identity() * min_eig;
}

TrackUKF::MeasurementCovariance TrackUKF::enforce_spd_meas(const MeasurementCovariance& matrix, double min_eig) {
    MeasurementCovariance sym = (matrix + matrix.transpose()) * 0.5;
    Eigen::SelfAdjointEigenSolver<MeasurementCovariance> es(sym);
    if (es.info() == Eigen::Success) {
        Eigen::Matrix<double, DIM_Z, 1> ev = es.eigenvalues().cwiseMax(min_eig);
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
    return sym + MeasurementCovariance::Identity() * min_eig;
}

TrackUKF::StateCovariance TrackUKF::get_dynamic_Q(double h) const {
    double std_pos = 0.05 * h;
    double std_vel = 0.5 * h;

    StateCovariance Q = StateCovariance::Zero();
    Q(0, 0) = std_pos * std_pos;
    Q(1, 1) = std_pos * std_pos;
    Q(3, 3) = std_pos * std_pos;
    Q(2, 2) = 1e-4;

    Q(4, 4) = std_vel * std_vel;
    Q(5, 5) = std_vel * std_vel;
    Q(6, 6) = 1e-4;
    Q(7, 7) = 1e-4;

    return enforce_spd(Q);
}

void TrackUKF::initiate(const MeasurementVector& measurement) {
    x_.setZero();
    x_.head<4>() = measurement;

    double h = measurement(3);
    P_.setZero();

    double p_pos = (0.1 * h) * (0.1 * h);
    P_(0, 0) = p_pos;
    P_(1, 1) = p_pos;
    P_(3, 3) = p_pos;
    P_(2, 2) = 1e-2;

    double p_vel = (10.0 * h) * (10.0 * h);
    P_(4, 4) = p_vel;
    P_(5, 5) = p_vel;
    P_(6, 6) = 1e-2;
    P_(7, 7) = 1e-2;

    P_ = enforce_spd(P_);
}

TrackUKF::MeasurementVector TrackUKF::measurement_function(const StateVector& x) {
    double cx = x(0);
    double cy = x(1);
    double a  = x(2);
    double h  = x(3);

    double w = a * h;
    double xmin = cx - w * 0.5;
    double xmax = cx + w * 0.5;
    double ymin = cy - h * 0.5;
    double ymax = cy + h * 0.5;

    constexpr double IMG_WIDTH = 1242.0;
    constexpr double IMG_HEIGHT = 375.0;

    double xmin_obs = std::max(0.0, xmin);
    double xmax_obs = std::min(IMG_WIDTH, xmax);
    double ymin_obs = std::max(0.0, ymin);
    double ymax_obs = std::min(IMG_HEIGHT, ymax);

    if (xmin_obs >= xmax_obs || ymin_obs >= ymax_obs) {
        MeasurementVector out;
        out << -1000.0, -1000.0, a, h;
        return out;
    }

    double w_obs = xmax_obs - xmin_obs;
    double h_obs = ymax_obs - ymin_obs;

    double cx_obs = (xmin_obs + xmax_obs) * 0.5;
    double cy_obs = (ymin_obs + ymax_obs) * 0.5;
    double a_obs = (h_obs > 0.0) ? (w_obs / h_obs) : a;

    MeasurementVector out;
    out << cx_obs, cy_obs, a_obs, h_obs;
    return out;
}

TrackUKF::StateVector TrackUKF::transition_function(const StateVector& x, double dt) {
    double cx    = x(0);
    double cy    = x(1);
    double a     = x(2);
    double h     = x(3);
    double vx    = x(4);
    double vy    = x(5);
    double omega = x(6);
    double gamma = x(7);

    double cx_new, cy_new, vx_new, vy_new;
    if (std::abs(omega) < 1e-4) {
        cx_new = cx + vx * dt;
        cy_new = cy + vy * dt;
        vx_new = vx;
        vy_new = vy;
    } else {
        double sin_w = std::sin(omega * dt);
        double cos_w = std::cos(omega * dt);
        cx_new = cx + (vx / omega) * sin_w - (vy / omega) * (1.0 - cos_w);
        cy_new = cy + (vx / omega) * (1.0 - cos_w) + (vy / omega) * sin_w;

        vx_new = vx * cos_w - vy * sin_w;
        vy_new = vx * sin_w + vy * cos_w;
    }

    double a_new = a;
    double h_new = h * std::exp(gamma * dt);

    StateVector out;
    out << cx_new, cy_new, a_new, h_new, vx_new, vy_new, omega, gamma;
    return out;
}

void TrackUKF::predict(const cv::Mat& H_camera) {
    // 1. Áp dụng biến đổi camera (CMC) nếu có
    if (!H_camera.empty() && H_camera.rows == 2 && H_camera.cols == 3) {
        double h00 = H_camera.at<double>(0, 0);
        double h01 = H_camera.at<double>(0, 1);
        double h02 = H_camera.at<double>(0, 2);
        double h10 = H_camera.at<double>(1, 0);
        double h11 = H_camera.at<double>(1, 1);
        double h12 = H_camera.at<double>(1, 2);

        // Bù trừ vị trí [cx, cy]
        double cx = x_(0);
        double cy = x_(1);
        x_(0) = h00 * cx + h01 * cy + h02;
        x_(1) = h10 * cx + h11 * cy + h12;

        Eigen::Matrix2d R2x2;
        R2x2 << h00, h01,
                h10, h11;
        double det = std::abs(R2x2.determinant());
        double scale = std::sqrt(std::max(det, 1e-6));
        Eigen::Matrix2d R_rot = R2x2 / (scale + 1e-6);

        // Bù trừ vận tốc [vx, vy]
        Eigen::Vector2d vel = x_.segment<2>(4);
        x_.segment<2>(4) = R_rot * vel;

        x_(3) *= scale;
        x_(7) *= scale;

        // Truyền biến đổi vào hiệp phương sai P
        P_.block<2, 2>(0, 0) = R2x2 * P_.block<2, 2>(0, 0) * R2x2.transpose();
        P_.block<2, 2>(4, 4) = R_rot * P_.block<2, 2>(4, 4) * R_rot.transpose();
    }

    P_ = enforce_spd(P_);
    StateCovariance Q = get_dynamic_Q(x_(3));

    // 2. Tính các điểm Sigma cho bước dự đoán
    double n = static_cast<double>(DIM_X);
    double gamma = std::sqrt(n + lambda_);

    Eigen::SelfAdjointEigenSolver<StateCovariance> es(P_);
    Eigen::Matrix<double, DIM_X, 1> ev = es.eigenvalues().cwiseMax(1e-5);
    StateCovariance sqrtP = es.eigenvectors() * ev.cwiseSqrt().asDiagonal();

    Eigen::Matrix<double, DIM_X, NUM_SIGMAS> sigmas;
    sigmas.col(0) = x_;
    for (int i = 0; i < DIM_X; ++i) {
        sigmas.col(i + 1)         = x_ + gamma * sqrtP.col(i);
        sigmas.col(i + 1 + DIM_X) = x_ - gamma * sqrtP.col(i);
    }

    // 3. Lan truyền Sigma qua hàm chuyển đổi trạng thái phi tuyến f(x, dt)
    Eigen::Matrix<double, DIM_X, NUM_SIGMAS> sigmas_f;
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        sigmas_f.col(i) = transition_function(sigmas.col(i), dt_);
    }

    // 4. Ước lượng Mean và Covariance dự đoán
    StateVector x_pred = StateVector::Zero();
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        x_pred += Wm_(i) * sigmas_f.col(i);
    }

    StateCovariance P_pred = StateCovariance::Zero();
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        StateVector diff = sigmas_f.col(i) - x_pred;
        P_pred += Wc_(i) * (diff * diff.transpose());
    }
    P_pred += Q;

    x_ = x_pred;
    P_ = enforce_spd(P_pred);
}

void TrackUKF::update(const MeasurementVector& measurement) {
    double h = std::max(x_(3), 1.0);
    double std_pos = 0.03 * h;
    double std_h   = 0.03 * h;

    MeasurementCovariance R = MeasurementCovariance::Zero();
    R(0, 0) = std_pos * std_pos;
    R(1, 1) = std_pos * std_pos;
    R(2, 2) = 1e-4;
    R(3, 3) = std_h * std_h;
    R = enforce_spd_meas(R);

    // 1. Sinh Sigma points từ trạng thái dự đoán
    double n = static_cast<double>(DIM_X);
    double gamma = std::sqrt(n + lambda_);

    Eigen::SelfAdjointEigenSolver<StateCovariance> es(P_);
    Eigen::Matrix<double, DIM_X, 1> ev = es.eigenvalues().cwiseMax(1e-5);
    StateCovariance sqrtP = es.eigenvectors() * ev.cwiseSqrt().asDiagonal();

    Eigen::Matrix<double, DIM_X, NUM_SIGMAS> sigmas;
    sigmas.col(0) = x_;
    for (int i = 0; i < DIM_X; ++i) {
        sigmas.col(i + 1)         = x_ + gamma * sqrtP.col(i);
        sigmas.col(i + 1 + DIM_X) = x_ - gamma * sqrtP.col(i);
    }

    // 2. Chiếu Sigma sang không gian đo lường h(x)
    Eigen::Matrix<double, DIM_Z, NUM_SIGMAS> sigmas_h;
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        sigmas_h.col(i) = measurement_function(sigmas.col(i));
    }

    // 3. Ước lượng Mean và Covariance của đo lường
    MeasurementVector z_pred = MeasurementVector::Zero();
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        z_pred += Wm_(i) * sigmas_h.col(i);
    }

    MeasurementCovariance Pzz = MeasurementCovariance::Zero();
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        MeasurementVector diff_z = sigmas_h.col(i) - z_pred;
        Pzz += Wc_(i) * (diff_z * diff_z.transpose());
    }
    Pzz += R;
    Pzz = enforce_spd_meas(Pzz);

    // 4. Tính Hiệp phương sai chéo Pxz
    Eigen::Matrix<double, DIM_X, DIM_Z> Pxz = Eigen::Matrix<double, DIM_X, DIM_Z>::Zero();
    for (int i = 0; i < NUM_SIGMAS; ++i) {
        StateVector diff_x = sigmas.col(i) - x_;
        MeasurementVector diff_z = sigmas_h.col(i) - z_pred;
        Pxz += Wc_(i) * (diff_x * diff_z.transpose());
    }

    // 5. Kalman Gain K = Pxz * Pzz^-1
    Eigen::Matrix<double, DIM_X, DIM_Z> K = Pxz * Pzz.inverse();

    MeasurementVector innovation = measurement - z_pred;
    x_ = x_ + K * innovation;
    P_ = P_ - K * Pzz * K.transpose();
    P_ = enforce_spd(P_);
}

} // namespace ukf_deepsort
