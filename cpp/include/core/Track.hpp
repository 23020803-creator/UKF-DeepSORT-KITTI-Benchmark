#pragma once

#include "utils/Types.hpp"
#include "core/UKF.hpp"
#include <vector>
#include <memory>

namespace ukf_deepsort {

/**
 * @brief Đại diện cho một quỹ đạo đối tượng (Track) được quản lý trong bộ nhớ hệ thống.
 */
class Track {
public:
    Track(double fps, int track_id, int class_id, const BBox& bbox, const std::vector<float>& feature);

    void predict(const cv::Mat& H_camera = cv::Mat());
    void update(const BBox& bbox, const std::vector<float>& feature, int img_w, int img_h);
    void mark_missed();

    inline bool is_tentative() const { return state_ == TrackState::TENTATIVE; }
    inline bool is_confirmed() const { return state_ == TrackState::CONFIRMED; }
    inline bool is_deleted() const { return state_ == TrackState::DELETED; }

    BBox to_tlwh() const;

    // Getters
    inline int track_id() const { return track_id_; }
    inline int class_id() const { return class_id_; }
    inline TrackState state() const { return state_; }
    inline int hits() const { return hits_; }
    inline int age() const { return age_; }
    inline int time_since_update() const { return time_since_update_; }
    inline const std::vector<float>& smooth_feature() const { return smooth_feature_; }
    inline const TrackUKF& ukf() const { return ukf_; }
    inline TrackUKF& ukf() { return ukf_; }
    inline const BBox& last_bbox() const { return last_bbox_; }

    static TrackUKF::MeasurementVector tlwh_to_xyah(const BBox& bbox);

private:
    int track_id_{0};
    int class_id_{0};
    TrackUKF ukf_;

    BBox last_bbox_;
    std::vector<float> smooth_feature_;

    int hits_{1};
    int age_{1};
    int time_since_update_{0};
    TrackState state_{TrackState::TENTATIVE};

    int n_init_{3};
    int max_age_{30};
};

} // namespace ukf_deepsort
