#include "core/CMC.hpp"
#include <opencv2/calib3d.hpp>
#include <algorithm>

namespace ukf_deepsort {

SparseOpticalFlowCMC::SparseOpticalFlowCMC(int max_corners, double quality_level, double min_distance, double scale_factor)
    : max_corners_(max_corners), quality_level_(quality_level), min_distance_(min_distance), scale_factor_(scale_factor) {}

void SparseOpticalFlowCMC::reset() {
    prev_gray_.release();
    prev_pts_.clear();
}

void SparseOpticalFlowCMC::extract_new_pts(const cv::Mat& gray_img, const std::vector<cv::Point2f>& existing_pts) {
    cv::Mat mask = cv::Mat::ones(gray_img.size(), CV_8U) * 255;
    // Loại bỏ 20% khu vực đáy ảnh (vị trí nắp ca-pô xe)
    int hood_y = static_cast<int>(gray_img.rows * 0.8);
    mask(cv::Rect(0, hood_y, gray_img.cols, gray_img.rows - hood_y)).setTo(0);

    // Tạo vùng cấm xung quanh các điểm đã tồn tại
    for (const auto& pt : existing_pts) {
        cv::circle(mask, pt, static_cast<int>(min_distance_), 0, -1);
    }

    int needed = (existing_pts.empty()) ? max_corners_ : (max_corners_ - static_cast<int>(existing_pts.size()));
    if (needed > 0) {
        std::vector<cv::Point2f> new_pts;
        cv::goodFeaturesToTrack(gray_img, new_pts, needed, quality_level_, min_distance_, mask);
        
        if (existing_pts.empty()) {
            prev_pts_ = new_pts;
        } else {
            prev_pts_ = existing_pts;
            prev_pts_.insert(prev_pts_.end(), new_pts.begin(), new_pts.end());
        }
    } else {
        prev_pts_ = existing_pts;
    }
}

cv::Mat SparseOpticalFlowCMC::apply(const cv::Mat& current_frame) {
    cv::Mat H = cv::Mat::eye(2, 3, CV_64F);
    if (current_frame.empty()) return H;

    cv::Mat small_frame, curr_gray;
    cv::resize(current_frame, small_frame, cv::Size(), scale_factor_, scale_factor_);
    cv::cvtColor(small_frame, curr_gray, cv::COLOR_BGR2GRAY);

    if (prev_gray_.empty()) {
        extract_new_pts(curr_gray);
        prev_gray_ = curr_gray;
        return H;
    }

    if (!prev_pts_.empty()) {
        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_pts_, curr_pts, status, err);

        std::vector<cv::Point2f> valid_prev, valid_curr;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] == 1) {
                valid_prev.push_back(prev_pts_[i]);
                valid_curr.push_back(curr_pts[i]);
            }
        }

        if (valid_prev.size() >= 4) {
            std::vector<uchar> inliers;
            cv::Mat H_est = cv::estimateAffinePartial2D(valid_prev, valid_curr, inliers, cv::RANSAC);
            if (!H_est.empty() && H_est.rows == 2 && H_est.cols == 3) {
                H_est.convertTo(H, CV_64F);
                // Bù trừ hệ số scale cho phần translation
                H.at<double>(0, 2) /= scale_factor_;
                H.at<double>(1, 2) /= scale_factor_;

                std::vector<cv::Point2f> inlier_pts;
                for (size_t i = 0; i < inliers.size(); ++i) {
                    if (inliers[i]) {
                        inlier_pts.push_back(valid_curr[i]);
                    }
                }
                prev_pts_ = inlier_pts;
            } else {
                prev_pts_.clear();
            }
        } else {
            prev_pts_.clear();
        }
    }

    // Refresh điểm đặc trưng nếu số lượng giảm xuống dưới 40%
    int min_pts_thresh = static_cast<int>(max_corners_ * 0.4);
    if (static_cast<int>(prev_pts_.size()) < min_pts_thresh) {
        extract_new_pts(curr_gray, prev_pts_);
    }

    prev_gray_ = curr_gray;
    return H;
}

} // namespace ukf_deepsort
