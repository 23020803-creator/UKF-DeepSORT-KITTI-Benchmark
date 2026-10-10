#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

namespace ukf_deepsort {

/**
 * @brief Bộ lọc Kalman Phi Tuyến (Unscented Kalman Filter - UKF) 8D với mô hình CTRV mở rộng.
 * 
 * Trạng thái 8 chiều: [cx, cy, a, h, vx, vy, omega, gamma]
 * - cx, cy: Tọa độ tâm hộp bao.
 * - a: Aspect ratio (w / h).
 * - h: Chiều cao hộp bao.
 * - vx, vy: Vận tốc di chuyển 2D.
 * - omega: Vận tốc góc quay (Turn rate).
 * - gamma: Tỷ lệ giãn nở khung hình (Perspective expansion rate).
 */
class TrackUKF {
public:
    static constexpr int DIM_X = 8;
    static constexpr int DIM_Z = 4;
    static constexpr int NUM_SIGMAS = 2 * DIM_X + 1; // 17 điểm Sigma

    using StateVector = Eigen::Matrix<double, DIM_X, 1>;
    using StateCovariance = Eigen::Matrix<double, DIM_X, DIM_X>;
    using MeasurementVector = Eigen::Matrix<double, DIM_Z, 1>;
    using MeasurementCovariance = Eigen::Matrix<double, DIM_Z, DIM_Z>;

    explicit TrackUKF(double fps = 10.0);

    /**
     * @brief Khởi tạo trạng thái ban đầu của bộ lọc từ measurement [cx, cy, a, h].
     */
    void initiate(const MeasurementVector& measurement);

    /**
     * @brief Dự đoán trạng thái bước tiếp theo và bù trừ chuyển động camera (CMC).
     * @param H_camera Ma trận Affine 2x3 từ module CMC (nếu có).
     */
    void predict(const cv::Mat& H_camera = cv::Mat());

    /**
     * @brief Cập nhật trạng thái bộ lọc từ quan sát đo lường mới [cx, cy, a, h].
     */
    void update(const MeasurementVector& measurement);

    /**
     * @brief Lấy vector trạng thái hiện tại.
     */
    inline const StateVector& mean() const { return x_; }
    inline StateVector& mean() { return x_; }

    /**
     * @brief Lấy ma trận hiệp phương sai hiện tại.
     */
    inline const StateCovariance& covariance() const { return P_; }
    inline StateCovariance& covariance() { return P_; }

    /**
     * @brief Hàm đo lường phi tuyến h(x) ánh xạ trạng thái 8D sang 4D có kẹp biên ảnh camera.
     */
    static MeasurementVector measurement_function(const StateVector& x);

    /**
     * @brief Hàm chuyển đổi trạng thái phi tuyến f(x, dt) mô hình CTRV mở rộng.
     */
    static StateVector transition_function(const StateVector& x, double dt);

private:
    double dt_{0.1};

    StateVector x_{StateVector::Zero()};
    StateCovariance P_{StateCovariance::Identity()};

    // Trọng số Unscented Transform (Merwe Scaled Sigma Points)
    double alpha_{0.3};
    double beta_{2.0};
    double kappa_{0.0};
    double lambda_{0.0};

    Eigen::Matrix<double, NUM_SIGMAS, 1> Wm_;
    Eigen::Matrix<double, NUM_SIGMAS, 1> Wc_;

    void init_sigma_weights();
    StateCovariance get_dynamic_Q(double h) const;
    static StateCovariance enforce_spd(const StateCovariance& matrix, double min_eig = 1e-5);
    static MeasurementCovariance enforce_spd_meas(const MeasurementCovariance& matrix, double min_eig = 1e-5);
};

} // namespace ukf_deepsort
