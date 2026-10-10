#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace ukf_deepsort {

/**
 * @brief Bộ ước lượng chuyển động camera (Camera Motion Compensation - CMC) 
 * sử dụng Sparse Optical Flow (Lucas-Kanade) và RANSAC Affine 2D.
 */
class SparseOpticalFlowCMC {
public:
    explicit SparseOpticalFlowCMC(int max_corners = 100, 
                                 double quality_level = 0.01, 
                                 double min_distance = 30.0, 
                                 double scale_factor = 0.35);

    /**
     * @brief Ước lượng ma trận Affine 2x3 giữa khung hình trước và khung hình hiện tại.
     * @param current_frame Khung hình ảnh màu BGR hiện tại.
     * @return cv::Mat Ma trận Affine 2x3 kiểu CV_64F. Trả về ma trận đơn vị nếu không đủ điểm inlier.
     */
    cv::Mat apply(const cv::Mat& current_frame);

    /**
     * @brief Đặt lại trạng thái bộ nhớ đệm (khi chuyển sequence video mới).
     */
    void reset();

private:
    int max_corners_{100};
    double quality_level_{0.01};
    double min_distance_{30.0};
    double scale_factor_{0.35};

    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;

    void extract_new_pts(const cv::Mat& gray_img, const std::vector<cv::Point2f>& existing_pts = {});
};

} // namespace ukf_deepsort
