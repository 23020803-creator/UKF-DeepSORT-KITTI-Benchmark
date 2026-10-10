#include "core/Track.hpp"
#include <cmath>
#include <numeric>

namespace ukf_deepsort {

TrackUKF::MeasurementVector Track::tlwh_to_xyah(const BBox& bbox) {
    TrackUKF::MeasurementVector meas;
    meas(0) = bbox.x + bbox.w * 0.5;
    meas(1) = bbox.y + bbox.h * 0.5;
    meas(2) = (bbox.h > 1e-4f) ? (bbox.w / bbox.h) : 1.0;
    meas(3) = bbox.h;
    return meas;
}

BBox Track::to_tlwh() const {
    const auto& mean = ukf_.mean();
    double cx = mean(0);
    double cy = mean(1);
    double a  = mean(2);
    double h  = mean(3);
    double w  = a * h;

    float x = static_cast<float>(cx - w * 0.5);
    float y = static_cast<float>(cy - h * 0.5);
    return BBox(x, y, static_cast<float>(w), static_cast<float>(h));
}

Track::Track(double fps, int track_id, int class_id, const BBox& bbox, const std::vector<float>& feature)
    : track_id_(track_id), class_id_(class_id), ukf_(fps), last_bbox_(bbox) {
    
    ukf_.initiate(tlwh_to_xyah(bbox));

    if (!feature.empty()) {
        smooth_feature_ = feature;
        // L2-normalize
        double sum_sq = 0.0;
        for (float v : smooth_feature_) sum_sq += v * v;
        double norm = std::sqrt(sum_sq);
        if (norm > 1e-6) {
            for (float& v : smooth_feature_) v /= static_cast<float>(norm);
        }
    }
}

void Track::predict(const cv::Mat& H_camera) {
    ukf_.predict(H_camera);
    age_++;
    time_since_update_++;
}

void Track::update(const BBox& bbox, const std::vector<float>& feature, int img_w, int img_h) {
    auto meas = tlwh_to_xyah(bbox);
    last_bbox_ = bbox;

    double current_w = ukf_.mean()(2) * ukf_.mean()(3);
    double new_w = bbox.w;

    double old_vx = ukf_.mean()(4);
    double old_vy = ukf_.mean()(5);

    // Soft reboot nếu chiều rộng biến đổi đột ngột hoặc mất dấu > 1 frame
    if (new_w > 1.5 * current_w || new_w < 0.6 * current_w || time_since_update_ > 1) {
        ukf_.covariance()(0, 0) = std::pow(2.0 * meas(3), 2.0);
        ukf_.covariance()(1, 1) = std::pow(2.0 * meas(3), 2.0);
        ukf_.covariance()(2, 2) = 1e-1;
        ukf_.covariance()(3, 3) = std::pow(1.0 * meas(3), 2.0);
    }

    ukf_.update(meas);

    // Ràng buộc vật lý sau update
    double new_vx = ukf_.mean()(4);
    double new_vy = ukf_.mean()(5);

    double max_acc_x = 0.15 * bbox.w;
    double max_acc_y = 0.15 * bbox.h;

    if (std::abs(new_vx - old_vx) > max_acc_x) {
        ukf_.mean()(4) = old_vx + (new_vx > old_vx ? max_acc_x : -max_acc_x);
    }
    if (std::abs(new_vy - old_vy) > max_acc_y) {
        ukf_.mean()(5) = old_vy + (new_vy > old_vy ? max_acc_y : -max_acc_y);
    }
    if (std::abs(old_vy) > 5.0 && (old_vy * ukf_.mean()(5) < 0)) {
        ukf_.mean()(5) = 0.0;
    }

    // Cập nhật ReID có cơ chế bảo vệ mép màn hình
    if (!feature.empty()) {
        float margin = 15.0f;
        bool is_safe_inside = (bbox.x > margin && bbox.y > margin && 
                               bbox.x + bbox.w < img_w - margin && 
                               bbox.y + bbox.h < img_h - margin);

        if (is_safe_inside) {
            std::vector<float> norm_feat = feature;
            double sum_sq = 0.0;
            for (float v : norm_feat) sum_sq += v * v;
            double norm = std::sqrt(sum_sq);
            if (norm > 1e-6) {
                for (float& v : norm_feat) v /= static_cast<float>(norm);
            }

            if (smooth_feature_.empty()) {
                smooth_feature_ = norm_feat;
            } else {
                float alpha = 0.9f;
                for (size_t i = 0; i < smooth_feature_.size() && i < norm_feat.size(); ++i) {
                    smooth_feature_[i] = alpha * smooth_feature_[i] + (1.0f - alpha) * norm_feat[i];
                }
                // Chuẩn hóa lại sau khi cộng dồn
                sum_sq = 0.0;
                for (float v : smooth_feature_) sum_sq += v * v;
                norm = std::sqrt(sum_sq);
                if (norm > 1e-6) {
                    for (float& v : smooth_feature_) v /= static_cast<float>(norm);
                }
            }
        }
    }

    hits_++;
    time_since_update_ = 0;
    if (state_ == TrackState::TENTATIVE && hits_ >= n_init_) {
        state_ = TrackState::CONFIRMED;
    }
}

void Track::mark_missed() {
    if (state_ == TrackState::TENTATIVE) {
        state_ = TrackState::DELETED;
    } else if (time_since_update_ > max_age_) {
        state_ = TrackState::DELETED;
    }
}

} // namespace ukf_deepsort
